"""Qwen3 decoder: RMSNorm, RoPE, GQA attention with q/k norm, SwiGLU MLP.

`forward` serves three call patterns with one code path:

  naive   : input_ids [1, T], no cache. Recomputes the whole prefix every token.
  prefill : input_ids [1, T], cache + slot. Writes K/V for positions 0..T-1.
  decode  : input_ids [B, 1], cache + slots. Appends one position per sequence.

It returns logits for the final position of every row, [B, vocab].

Module names match the HF checkpoint (minus the `model.` prefix) so the state
dict loads without a rename table.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVPool
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same order as HF: normalize in fp32, cast back, then scale.
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


class RotaryEmbedding(nn.Module):
    """cos/sin tables indexed by absolute position. Built once in fp32."""

    def __init__(self, cfg: ModelConfig, device: torch.device):
        super().__init__()
        hd = cfg.head_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
        pos = torch.arange(cfg.max_position_embeddings, device=device).float()
        freqs = pos[:, None] * inv_freq[None, :]          # [P, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [P, hd]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # positions [B, T] -> [B, 1, T, hd] to broadcast over heads
        return self.cos[positions].to(dtype)[:, None], self.sin[positions].to(dtype)[:, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.scale = cfg.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,           # [B, T, hidden]
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,   # [B, T]
        cache: KVPool | None,
        slots: torch.Tensor | slice | None,
        kv_len: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        # Project, split heads, per-head RMSNorm (Qwen3 specific), then RoPE.
        q = self.q_norm(self.q_proj(x).view(B, T, cfg.num_heads, cfg.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is None:
            # Naive path: attend over exactly the tokens in this call.
            keys, values, mask, causal = k, v, None, T > 1
        else:
            cache.write(self.layer_idx, slots, positions, k, v)
            keys, values, mask = cache.read(self.layer_idx, slots, positions, kv_len)
            causal = False
            if T == kv_len:
                # Prefill from position 0: the mask is plain causal, and passing
                # is_causal instead keeps the fused kernels eligible.
                mask, causal = None, T > 1

        # GQA: every KV head serves `group_size` query heads. enable_gqa does the
        # broadcast inside the kernel rather than materializing repeated K/V.
        out = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask, is_causal=causal, scale=self.scale, enable_gqa=True
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, T, cfg.q_dim))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, positions, cache, slots, kv_len):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, positions, cache, slots, kv_len)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        with torch.device("meta"):
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(cfg, device)

    @classmethod
    def from_pretrained(
        cls,
        cfg: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: str | None = None,
    ) -> "Qwen3":
        from .weights import load_state_dict, remap_hf_keys, resolve_model_dir

        sd = remap_hf_keys(load_state_dict(resolve_model_dir(cfg, model_dir), device, dtype), cfg)
        model = cls(cfg, device, dtype)
        model.load_state_dict(sd, strict=True, assign=True)
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        return model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T]
        positions: torch.Tensor,           # [B, T]
        cache: KVPool | None = None,
        slots: torch.Tensor | slice | None = None,
    ) -> torch.Tensor:
        """Logits for the last position of each row: [B, vocab]."""
        kv_len = int(positions.max().item()) + 1 if cache is not None else 0
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, positions, cache, slots, kv_len)
        x = self.norm(x[:, -1:])
        return self.lm_head(x)[:, 0]

    def param_bytes(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel() * p.element_size()
        return total


# --- single-sequence generation, the oracle for everything batched ---------


@torch.no_grad()
def generate_naive(model: Qwen3, prompt_ids: list[int], max_new_tokens: int,
                   eos_ids: tuple[int, ...] = (), on_token=None) -> list[int]:
    """No cache. Every step re-runs the forward pass over the entire sequence."""
    ids = list(prompt_ids)
    out: list[int] = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], device=model.device)
        pos = torch.arange(len(ids), device=model.device)[None]
        tok = int(model(x, pos).argmax(-1).item())
        ids.append(tok)
        out.append(tok)
        if on_token:
            on_token(tok)
        if tok in eos_ids:
            break
    return out


@torch.no_grad()
def generate_cached(model: Qwen3, cache: KVPool, prompt_ids: list[int], max_new_tokens: int,
                    eos_ids: tuple[int, ...] = (), on_token=None) -> list[int]:
    """Prefill once, then one position per step. Batch size 1."""
    slot = cache.alloc()
    slots = slice(slot, slot + 1)
    try:
        x = torch.tensor([prompt_ids], device=model.device)
        pos = torch.arange(len(prompt_ids), device=model.device)[None]
        tok = int(model(x, pos, cache, slots).argmax(-1).item())
        out = [tok]
        if on_token:
            on_token(tok)
        n = len(prompt_ids)
        while len(out) < max_new_tokens and tok not in eos_ids:
            x = torch.tensor([[tok]], device=model.device)
            pos = torch.tensor([[n]], device=model.device)
            tok = int(model(x, pos, cache, slots).argmax(-1).item())
            out.append(tok)
            n += 1
            if on_token:
                on_token(tok)
        return out
    finally:
        cache.release(slot)
