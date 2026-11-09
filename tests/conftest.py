import os

import pytest
import torch

from engine.cache import KVPool
from engine.config import QWEN3_0_6B, pick_device, pick_dtype
from engine.model import Qwen3
from engine.tokenizer import load_tokenizer
from engine.weights import resolve_model_dir


@pytest.fixture(scope="session")
def device():
    return pick_device(os.environ.get("DEVICE"))


@pytest.fixture(scope="session")
def dtype(device):
    # fp32 by default: the parity tests compare token ids and bf16 near-ties
    # flip between kernel shapes. DTYPE=bf16 to run them the fast way.
    return pick_dtype(device, os.environ.get("DTYPE", "fp32"))


@pytest.fixture(scope="session")
def model_dir():
    return resolve_model_dir(QWEN3_0_6B, os.environ.get("MODEL_DIR"))


@pytest.fixture(scope="session")
def tokenizer(model_dir):
    return load_tokenizer(model_dir)


@pytest.fixture(scope="session")
def model(model_dir, device, dtype):
    torch.manual_seed(0)
    return Qwen3.from_pretrained(QWEN3_0_6B, device, dtype, str(model_dir))


@pytest.fixture
def cache(model):
    return KVPool(QWEN3_0_6B, num_slots=8, max_len=512, device=model.device, dtype=model.dtype)


@pytest.fixture(scope="session")
def prompts(tokenizer):
    def chat(s):
        return tokenizer.apply_chat_template([{"role": "user", "content": s}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    texts = [
        chat("Explain what a KV cache is in two sentences."),
        chat("Write a haiku about a lighthouse."),
        "The quick brown fox",
        chat("List three prime numbers and stop."),
        chat("What is 17 * 23? Show your work briefly."),
        "Once upon a time, in a harbour town,",
    ]
    return [tokenizer.encode(t) for t in texts]
