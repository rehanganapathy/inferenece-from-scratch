"""Tests that do not need the model weights."""

import torch

from engine.sampling import SamplingParams, sample
from engine.tokenizer import IncrementalDetokenizer


def test_greedy_is_argmax():
    logits = torch.randn(3, 50)
    assert sample(logits, [SamplingParams()] * 3).tolist() == logits.argmax(-1).tolist()


def test_top_k_restricts_support():
    logits = torch.randn(64, 100)
    g = torch.Generator().manual_seed(0)
    toks = sample(logits, [SamplingParams(temperature=1.0, top_k=3)] * 64, g)
    top3 = logits.topk(3, dim=-1).indices
    assert all(t in row for t, row in zip(toks.tolist(), top3.tolist()))


def test_top_p_keeps_at_least_one():
    logits = torch.full((4, 10), -100.0)
    logits[:, 7] = 100.0
    toks = sample(logits, [SamplingParams(temperature=1.0, top_p=0.01)] * 4)
    assert toks.tolist() == [7] * 4


def test_mixed_greedy_and_sampled_rows():
    logits = torch.randn(2, 20)
    toks = sample(logits, [SamplingParams(), SamplingParams(temperature=1.0, top_k=1)])
    assert toks.tolist() == logits.argmax(-1).tolist()


def test_incremental_detokenizer_matches_full_decode(tokenizer):
    text = "Héllo wörld — naïve café, 日本語のテキスト, emoji 🚀🎉 and\n\tcode: x = [1, 2]"
    ids = tokenizer.encode(text)
    d = IncrementalDetokenizer(tokenizer)
    out = "".join(d.push(t) for t in ids) + d.flush()
    assert out == tokenizer.decode(ids, skip_special_tokens=True)
