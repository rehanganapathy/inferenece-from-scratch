"""Safetensors loading into a plain state dict."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .config import ModelConfig

_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"]


def resolve_model_dir(cfg: ModelConfig, path: str | None = None) -> Path:
    """Local directory containing the checkpoint, downloading it if needed."""
    candidate = path or os.environ.get("MODEL_DIR")
    if candidate and Path(candidate).is_dir():
        return Path(candidate)
    return Path(snapshot_download(cfg.hf_repo, allow_patterns=_PATTERNS))


def load_state_dict(model_dir: Path, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    sd: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                sd[key] = fh.get_tensor(key).to(dtype=dtype, device=device)
    return sd


def remap_hf_keys(sd: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """HF names `model.layers.0...`; ours drop the `model.` prefix. Tie lm_head if absent."""
    out = {}
    for k, v in sd.items():
        out[k[len("model."):] if k.startswith("model.") else k] = v
    if "lm_head.weight" not in out:
        if not cfg.tie_word_embeddings:
            raise KeyError("lm_head.weight missing and embeddings are not tied")
        out["lm_head.weight"] = out["embed_tokens.weight"]
    return out
