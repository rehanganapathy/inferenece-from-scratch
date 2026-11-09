"""Generate bench/traces/mixed.json: a fixed request trace with realistic
length variation in both prompt and output.

Same-length requests hide the whole benefit of continuous batching, so lengths
are drawn from a wide log-normal and clipped. Seeded; regenerate with `make trace`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def lognormal_int(rng, n, median, sigma, lo, hi):
    x = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    return np.clip(np.round(x), lo, hi).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-len", type=int, default=2048, help="prompt + output must fit a slot")
    ap.add_argument("--out", default=str(Path(__file__).parent / "traces" / "mixed.json"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    prompt = lognormal_int(rng, a.n, median=256, sigma=0.9, lo=16, hi=1024)
    output = lognormal_int(rng, a.n, median=128, sigma=0.8, lo=8, hi=512)
    assert (prompt + output).max() <= a.max_len

    reqs = [
        {"id": i, "prompt_len": int(p), "max_new_tokens": int(o), "text_offset": int(rng.integers(0, 10_000))}
        for i, (p, o) in enumerate(zip(prompt, output))
    ]
    meta = {
        "seed": a.seed, "n": a.n,
        "prompt_len": {"min": int(prompt.min()), "median": float(np.median(prompt)), "max": int(prompt.max())},
        "max_new_tokens": {"min": int(output.min()), "median": float(np.median(output)), "max": int(output.max())},
        "total_output_tokens": int(output.sum()),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "requests": reqs}, indent=1))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
