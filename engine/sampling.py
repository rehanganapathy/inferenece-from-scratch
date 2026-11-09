"""Token sampling. Greedy, then temperature / top-k / top-p.

Works on a batch with per-row parameters so a decode step with mixed requests
is still one call. Rows with temperature 0 are greedy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0   # 0 => greedy
    top_k: int = 0             # 0 => disabled
    top_p: float = 1.0         # 1 => disabled
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def sample(logits: torch.Tensor, params: list[SamplingParams],
           generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]."""
    B, V = logits.shape
    assert len(params) == B
    if all(p.greedy for p in params):
        return logits.argmax(dim=-1)

    device = logits.device
    temp = torch.tensor([max(p.temperature, 1e-6) for p in params], device=device)
    top_k = torch.tensor([p.top_k if p.top_k > 0 else V for p in params], device=device)
    top_p = torch.tensor([p.top_p for p in params], device=device)

    scaled = logits.float() / temp[:, None]
    sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
    ranks = torch.arange(V, device=device)[None, :]
    probs = sorted_logits.softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    # Keep the smallest prefix whose mass reaches top_p; always keep rank 0.
    remove = (ranks >= top_k[:, None]) | ((cum - probs) > top_p[:, None])
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    choice = torch.multinomial(sorted_logits.softmax(dim=-1), 1, generator=generator)
    tokens = sorted_idx.gather(-1, choice)[:, 0]

    greedy_rows = torch.tensor([p.greedy for p in params], device=device)
    if greedy_rows.any():
        tokens = torch.where(greedy_rows, logits.argmax(dim=-1), tokens)
    return tokens
