"""Token-for-token gates. Every later optimization must keep these green."""

import pytest
import torch

from engine.config import QWEN3_0_6B
from engine.model import generate_cached, generate_naive
from engine.sched import NaiveScheduler, Request, Scheduler

EOS = QWEN3_0_6B.eos_token_ids
N = 64


def test_hf_parity_naive_greedy(model, prompts, model_dir, dtype):
    """Week 1 gate: our no-cache forward matches HF generate, greedy, 64 tokens."""
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(model.device).eval()
    ids = prompts[0]
    with torch.no_grad():
        ref = hf.generate(torch.tensor([ids], device=model.device), max_new_tokens=N, do_sample=False)
    ref = ref[0, len(ids):].tolist()
    ours = generate_naive(model, ids, N, EOS)
    assert ours == ref[: len(ours)]
    assert len(ours) == N or ours[-1] in EOS


def test_cached_matches_naive(model, cache, prompts):
    """Week 2 gate: KV cache path == naive path exactly. Catches off-by-one RoPE."""
    for ids in prompts[:3]:
        naive = generate_naive(model, ids, N, EOS)
        cached = generate_cached(model, cache, ids, N, EOS)
        assert cached == naive, f"cache diverges at index {next(i for i, (a, b) in enumerate(zip(cached, naive)) if a != b)}"


def test_cache_slot_reuse_is_clean(model, cache, prompts):
    """A slot that held a longer sequence must not leak stale K/V into a shorter one."""
    long_ids = prompts[0] + prompts[4]
    generate_cached(model, cache, long_ids, 16, EOS)
    short = generate_cached(model, cache, prompts[2], 16, EOS)
    assert short == generate_naive(model, prompts[2], 16, EOS)


@pytest.mark.parametrize("mode", ["continuous", "static"])
def test_batched_matches_single(model, cache, prompts, mode):
    """Week 3 gate: batched scheduler output == single-request oracle for every request."""
    oracle = {i: generate_cached(model, cache, ids, 24, EOS) for i, ids in enumerate(prompts)}
    sched = Scheduler(model, cache, max_batch_size=4, mode=mode)
    reqs = [sched.add(Request(prompt_ids=ids, max_new_tokens=24)) for ids in prompts]
    max_running = 0
    while sched.has_work():
        st = sched.step()
        max_running = max(max_running, st.running)
    assert max_running > 1, "batching never happened"
    assert len(sched.finished) == len(prompts)
    assert cache.num_free == cache.num_slots
    for i, r in enumerate(reqs):
        assert r.output_ids == oracle[i], f"request {i} diverges in {mode} mode"
        assert r.finish_reason in ("stop", "length")


def test_slot_packing_preserves_outputs(model, cache, prompts):
    """Staggered lengths force survivors to be moved down into freed slots mid-flight."""
    lens = [5, 20, 9, 16, 3, 12]
    oracle = [generate_cached(model, cache, ids, n, ()) for ids, n in zip(prompts, lens)]
    sched = Scheduler(model, cache, max_batch_size=4)
    moves = []
    orig_move = cache.move
    cache.move = lambda *a: (moves.append(a), orig_move(*a))
    try:
        reqs = [sched.add(Request(prompt_ids=ids, max_new_tokens=n, ignore_eos=True))
                for ids, n in zip(prompts, lens)]
        while sched.has_work():
            sched.step()
    finally:
        del cache.move
    assert moves, "no slot was relocated; the test is not exercising packing"
    assert [r.output_ids for r in reqs] == oracle
    assert cache.num_free == cache.num_slots


def test_static_mode_waits_for_batch_to_drain(model, cache, prompts):
    sched = Scheduler(model, cache, max_batch_size=2, mode="static")
    for ids in prompts[:4]:
        sched.add(Request(prompt_ids=ids, max_new_tokens=8, ignore_eos=True))
    admits = []
    while sched.has_work():
        admits.append(sched.step().admitted)
    # Two admission bursts of 2, nothing admitted while a batch is running.
    assert [a for a in admits if a] == [2, 2]


def test_naive_scheduler_matches_oracle(model, prompts):
    sched = NaiveScheduler(model)
    reqs = [sched.add(Request(prompt_ids=ids, max_new_tokens=8)) for ids in prompts[:2]]
    while sched.has_work():
        sched.step()
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == generate_naive(model, ids, 8, EOS)


def test_ignore_eos_and_length(model, cache, prompts):
    sched = Scheduler(model, cache, max_batch_size=2)
    r = sched.add(Request(prompt_ids=prompts[3], max_new_tokens=40, ignore_eos=True))
    while sched.has_work():
        sched.step()
    assert len(r.output_ids) == 40 and r.finish_reason == "length"
    assert len(r.token_times) == 40


def test_rejects_overlong_request(model, cache):
    sched = Scheduler(model, cache)
    with pytest.raises(ValueError):
        sched.add(Request(prompt_ids=[1] * 500, max_new_tokens=100))
