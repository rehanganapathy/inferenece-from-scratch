"""Timing harness. Built in week 1 so every later number is comparable.

Runs a trace against a scheduler in-process (no HTTP in the loop), records a
timestamp for every generated token, and dumps JSON.

    python -m bench.harness trace  --mode {naive,static,continuous} --concurrency 16
    python -m bench.harness seqlen --mode {naive,cached} --lengths 128,256,...,2048
    python -m bench.harness sweep  --mode continuous --concurrency 1,2,4,8,16,32,64

Closed loop: `concurrency` requests are kept in flight; when one finishes the
next in the trace is submitted. EOS is ignored so the trace's output lengths are
what gets measured, not whatever the model felt like stopping at.

Output JSON (one file per run):
  meta       : mode, concurrency, device, dtype, versions, git sha
  requests   : per request: prompt_len, output_len, t_submit, t_admit,
               t_first, t_finish, token_times (relative to run start)
  steps      : per step: t, running, waiting, admitted, tokens, max_slots
  summary    : throughput, TTFT / TPOT / latency percentiles
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from engine.cache import KVPool
from engine.config import QWEN3_0_6B, pick_device, pick_dtype
from engine.model import Qwen3, generate_cached, generate_naive
from engine.sched import NaiveScheduler, Request, Scheduler
from engine.tokenizer import load_tokenizer
from engine.weights import resolve_model_dir

RESULTS = Path(__file__).parent / "results"
TRACE = Path(__file__).parent / "traces" / "mixed.json"

# Prompt text is sliced from this so streams are readable, not random ids.
CORPUS = (
    "The harbour town wakes slowly in winter. Fishing boats leave before dawn and "
    "return with the tide, and the market on the quay sells whatever the morning "
    "brought in. Behind the warehouses the streets climb in narrow steps toward the "
    "old fort, where a small museum keeps maps, charts, and a brass telescope that "
    "still works. Visitors who make the climb are rewarded with a view across the "
    "bay to the lighthouse, which was automated decades ago but is repainted every "
    "spring by volunteers. The town council meets on the first Tuesday of each month "
    "to argue about parking, the ferry timetable, and the cost of dredging the "
    "channel. Most decisions are deferred to the following month. In summer the "
    "population triples, the cafés extend onto the pavement, and the ferry runs "
    "hourly instead of twice a day. By October the visitors are gone, the shutters "
    "come down on the ice cream kiosk, and the town returns to its own business. "
)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def meta(device, dtype, **kw) -> dict:
    m = {
        "device": str(device), "dtype": str(dtype).replace("torch.", ""),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "git": git_sha(), "model": QWEN3_0_6B.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if device.type == "cuda":
        m["gpu"] = torch.cuda.get_device_name(0)
        m["cuda"] = torch.version.cuda
    return m | kw


def load(args):
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    model_dir = resolve_model_dir(QWEN3_0_6B, args.model_dir)
    tok = load_tokenizer(model_dir)
    model = Qwen3.from_pretrained(QWEN3_0_6B, device, dtype, str(model_dir))
    return device, dtype, tok, model


def corpus_ids(tok, n_tokens: int, offset: int) -> list[int]:
    reps = 1 + (offset + 4 * n_tokens) // len(CORPUS)
    ids = tok.encode((CORPUS * reps)[offset % len(CORPUS):])
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pct(xs, ps=(50, 90, 99)) -> dict:
    if not xs:
        return {}
    arr = np.asarray(xs, dtype=float)
    out = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    out["mean"] = float(arr.mean())
    return out


def summarize(requests: list[dict], wall: float) -> dict:
    ttft = [r["t_first"] - r["t_submit"] for r in requests]
    tpot = [
        (r["t_finish"] - r["t_first"]) / (r["output_len"] - 1)
        for r in requests if r["output_len"] > 1
    ]
    lat = [r["t_finish"] - r["t_submit"] for r in requests]
    out_toks = sum(r["output_len"] for r in requests)
    return {
        "requests": len(requests),
        "wall_s": wall,
        "output_tokens": out_toks,
        "output_tokens_per_s": out_toks / wall,
        "requests_per_s": len(requests) / wall,
        "ttft_s": pct(ttft),
        "tpot_s": pct(tpot),
        "latency_s": pct(lat),
    }


def make_sched(model, mode: str, max_batch: int, max_len: int):
    if mode == "naive":
        return NaiveScheduler(model)
    cache = KVPool(model.cfg, max_batch, max_len, model.device, model.dtype)
    return Scheduler(model, cache, max_batch, mode=mode)


# --- trace run -----------------------------------------------------------------


def run_trace(model, tok, mode: str, concurrency: int, trace: list[dict], max_len: int,
              warmup: int = 2, quiet: bool = False) -> tuple[list[dict], list[dict], float]:
    sched = make_sched(model, mode, concurrency, max_len)
    reqs = [
        Request(prompt_ids=corpus_ids(tok, t["prompt_len"], t["text_offset"]),
                max_new_tokens=t["max_new_tokens"], ignore_eos=True)
        for t in trace
    ]
    # Warm up kernels on a throwaway so the first request isn't paying for it.
    for _ in range(warmup):
        w = make_sched(model, mode, 1, max_len)
        w.add(Request(prompt_ids=reqs[0].prompt_ids[:32], max_new_tokens=4, ignore_eos=True))
        while w.has_work():
            w.step()
    sync(model.device)

    pending = list(reqs)
    in_flight = 0
    t0 = time.perf_counter()
    while pending and in_flight < concurrency:
        sched.add(pending.pop(0)); in_flight += 1
    done_before = 0
    while sched.has_work():
        sched.step()
        finished = len(sched.finished)
        newly = finished - done_before
        done_before = finished
        in_flight -= newly
        while pending and in_flight < concurrency:
            sched.add(pending.pop(0)); in_flight += 1
        if not quiet and finished and finished % 32 == 0 and newly:
            el = time.perf_counter() - t0
            toks = sum(len(r.output_ids) for r in sched.finished)
            print(f"  {finished}/{len(reqs)} done  {toks / el:8.1f} tok/s  running={len(sched.running)}", file=sys.stderr)
    sync(model.device)
    wall = time.perf_counter() - t0

    rows = [
        {"id": r.id, "prompt_len": len(r.prompt_ids), "output_len": len(r.output_ids),
         "t_submit": r.t_submit - t0, "t_admit": r.t_admit - t0,
         "t_first": r.token_times[0] - t0, "t_finish": r.t_finish - t0,
         "token_times": [t - t0 for t in r.token_times]}
        for r in sched.finished
    ]
    steps = [
        {"t": s.t - t0, "running": s.running, "waiting": s.waiting, "admitted": s.admitted,
         "tokens": s.tokens, "max_slots": s.max_slots}
        for s in sched.stats
    ]
    return rows, steps, wall


def cmd_trace(args):
    device, dtype, tok, model = load(args)
    trace = json.loads(Path(args.trace).read_text())["requests"]
    if args.num_requests:
        trace = trace[: args.num_requests]
    print(f"[{args.mode}] concurrency={args.concurrency} requests={len(trace)} device={device} dtype={dtype}", file=sys.stderr)
    rows, steps, wall = run_trace(model, tok, args.mode, args.concurrency, trace, args.max_len, quiet=args.quiet)
    summary = summarize(rows, wall)
    out = {
        "meta": meta(device, dtype, mode=args.mode, concurrency=args.concurrency, trace=str(args.trace),
                     num_requests=len(trace), max_len=args.max_len),
        "summary": summary, "requests": rows, "steps": steps,
    }
    path = Path(args.out) if args.out else RESULTS / f"trace_{args.mode}_c{args.concurrency}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"{args.mode:<11} c={args.concurrency:<3} {summary['output_tokens_per_s']:8.1f} tok/s   "
          f"TTFT p50 {summary['ttft_s']['p50'] * 1000:7.1f} ms   TPOT p50 {summary['tpot_s']['p50'] * 1000:6.2f} ms   "
          f"wall {wall:6.1f}s  -> {path}")
    return out


def cmd_sweep(args):
    device, dtype, tok, model = load(args)
    trace = json.loads(Path(args.trace).read_text())["requests"]
    if args.num_requests:
        trace = trace[: args.num_requests]
    for c in args.concurrency:
        rows, steps, wall = run_trace(model, tok, args.mode, c, trace, args.max_len, quiet=True)
        summary = summarize(rows, wall)
        out = {"meta": meta(device, dtype, mode=args.mode, concurrency=c, trace=str(args.trace),
                            num_requests=len(trace), max_len=args.max_len),
               "summary": summary, "requests": rows, "steps": steps}
        path = RESULTS / f"trace_{args.mode}_c{c}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        print(f"{args.mode:<11} c={c:<3} {summary['output_tokens_per_s']:8.1f} tok/s   "
              f"TTFT p50 {summary['ttft_s']['p50'] * 1000:7.1f} ms   TPOT p50 {summary['tpot_s']['p50'] * 1000:6.2f} ms   "
              f"wall {wall:6.1f}s")


# --- sequence-length sweep (figure 1) ----------------------------------------


def cmd_seqlen(args):
    """Decode tokens/sec as a function of prompt length, single request."""
    device, dtype, tok, model = load(args)
    cache = None if args.mode == "naive" else KVPool(model.cfg, 1, max(args.lengths) + args.new_tokens + 1, device, dtype)
    results = []
    for L in args.lengths:
        ids = corpus_ids(tok, L, 0)
        times: list[float] = []
        gen = (lambda: generate_naive(model, ids, args.new_tokens, on_token=lambda _: times.append(time.perf_counter()))) \
            if cache is None else \
            (lambda: generate_cached(model, cache, ids, args.new_tokens, on_token=lambda _: times.append(time.perf_counter())))
        gen(); times.clear()                      # warm
        sync(device); t0 = time.perf_counter()
        gen(); sync(device)
        wall = time.perf_counter() - t0
        deltas = np.diff([t0] + times)
        decode_tps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else float("nan")
        row = {"seq_len": L, "new_tokens": len(times), "wall_s": wall, "first_token_s": float(deltas[0]),
               "decode_tokens_per_s": decode_tps, "per_token_s": [float(d) for d in deltas]}
        results.append(row)
        print(f"{args.mode:<7} L={L:<5} first {deltas[0] * 1000:8.1f} ms   decode {decode_tps:8.2f} tok/s", file=sys.stderr)
    out = {"meta": meta(device, dtype, mode=args.mode, new_tokens=args.new_tokens), "results": results}
    path = Path(args.out) if args.out else RESULTS / f"seqlen_{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"-> {path}")


# --- CLI -------------------------------------------------------------------------


def int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device"); ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"]); ap.add_argument("--model-dir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="run a trace at one concurrency")
    t.add_argument("--mode", choices=["naive", "static", "continuous"], required=True)
    t.add_argument("--concurrency", type=int, default=16)
    t.add_argument("--trace", default=str(TRACE)); t.add_argument("--num-requests", type=int)
    t.add_argument("--max-len", type=int, default=2048); t.add_argument("--out"); t.add_argument("--quiet", action="store_true")
    t.set_defaults(fn=cmd_trace)

    s = sub.add_parser("sweep", help="run a trace across concurrency levels")
    s.add_argument("--mode", choices=["naive", "static", "continuous"], required=True)
    s.add_argument("--concurrency", type=int_list, default=[1, 2, 4, 8, 16, 32, 64])
    s.add_argument("--trace", default=str(TRACE)); s.add_argument("--num-requests", type=int)
    s.add_argument("--max-len", type=int, default=2048)
    s.set_defaults(fn=cmd_sweep)

    q = sub.add_parser("seqlen", help="tokens/sec vs sequence length, single request")
    q.add_argument("--mode", choices=["naive", "cached"], required=True)
    q.add_argument("--lengths", type=int_list, default=[128, 256, 512, 768, 1024, 1536, 2048])
    q.add_argument("--new-tokens", type=int, default=32); q.add_argument("--out")
    q.set_defaults(fn=cmd_seqlen)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
