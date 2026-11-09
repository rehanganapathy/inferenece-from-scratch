# inference-engine

Phase 1: host Qwen3-0.6B from scratch, add a KV cache, then continuous batching,
behind a FastAPI server that streams tokens over SSE. See `phase-1-plan.md`.

```
engine/
  config.py     frozen model hyperparameters (the numbers the math refers to)
  weights.py    safetensors -> state dict
  model.py      RMSNorm, RoPE, GQA attention, SwiGLU MLP; naive / prefill / decode in one forward
  cache.py      per-sequence contiguous KV cache (slot pool)
  sched.py      waiting -> running -> finished, step(); continuous and static modes; naive baseline
  sampling.py   greedy, temperature, top-k, top-p, per-row in a batch
  tokenizer.py  tokenizer + incremental detokenization
  server.py     FastAPI, background step loop, SSE streaming
bench/
  harness.py    run a trace, per-token timestamps, JSON out; seqlen and concurrency sweeps
  make_trace.py generates traces/mixed.json (mixed prompt and output lengths, seeded)
  hardware.py   writes HARDWARE.md
figures/        one script per figure, reads bench/results/*.json
tests/          HF parity, cache == naive, batched == single-request oracle, server e2e
```

## Setup

```
make install            # uv sync
make test               # downloads Qwen/Qwen3-0.6B on first run; ~2 min on CPU
```

Env: `DEVICE` (cuda / mps / cpu, auto), `DTYPE` (bf16 / fp16 / fp32; bf16 on CUDA,
fp16 on MPS, fp32 on CPU), `MODEL_DIR` (local checkpoint dir, otherwise HF cache).

## Serve

```
make serve
curl -N localhost:8000/generate -d '{"prompt":"Explain KV caching.","max_tokens":128,"chat":true}'
curl -N localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
curl localhost:8000/stats
```

`MAX_BATCH` (default 32) and `MAX_LEN` (default 2048) size the KV pool:
`2 * layers * kv_heads * head_dim * bytes` per token per slot, about 112 KB in bf16.

## Benchmark

The three headline runs are the same command with the mode swapped:

```
make bench-naive
make bench-static
make bench-continuous
```

`make sweep` runs concurrency 1..64 for all three configurations, `make seqlen`
runs the single-request sequence-length sweep, and `make figures` produces all
three plots into `figures/out/`, running whatever benchmarks are missing first.
`make hardware` records the exact GPU and library versions to `HARDWARE.md`.

## Correctness gates

- `test_hf_parity_naive_greedy`: our forward pass vs HF `generate`, greedy, 64 tokens.
- `test_cached_matches_naive`: the KV cache path must reproduce the naive path exactly.
- `test_batched_matches_single`: every request in a mixed batch (continuous and
  static) must match the single-request oracle token for token.
- `test_concurrent_streams_are_all_correct`: 12 concurrent SSE streams through the server.

Tests run in fp32 by default so token-ID comparisons are exact across kernel
shapes; set `DTYPE=bf16` on a GPU to run them fast.
