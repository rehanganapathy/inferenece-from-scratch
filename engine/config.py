"""Frozen model hyperparameters.

The numbers live here, in source, rather than being read from HF's config.json
at runtime. Every derivation in the write-up refers to them by name.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_repo: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    pad_token_id: int

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Query heads per KV head (GQA)."""
        return self.num_heads // self.num_kv_heads

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """K and V for one token across every layer."""
        return 2 * self.num_layers * self.kv_dim * torch.tensor([], dtype=dtype).element_size()


QWEN3_0_6B = ModelConfig(
    name="qwen3-0.6b",
    hf_repo="Qwen/Qwen3-0.6B",
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_layers=28,
    num_heads=16,
    num_kv_heads=8,
    head_dim=128,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    max_position_embeddings=40960,
    tie_word_embeddings=True,
    eos_token_ids=(151645, 151643),  # <|im_end|>, <|endoftext|>
    pad_token_id=151643,
)

CONFIGS = {QWEN3_0_6B.name: QWEN3_0_6B}


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, name: str | None = None) -> torch.dtype:
    """bf16 on CUDA, fp16 on MPS, fp32 on CPU.

    MPS: fp16 and bf16 run at the same speed (~1.9x fp32 batched on an M5),
    but fp16 tracks fp32 greedy output more closely. Use DTYPE=bf16 if fp16
    ever overflows (NaN / garbage output).
    """
    if name:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    return {"cuda": torch.bfloat16, "mps": torch.float16}.get(device.type, torch.float32)
