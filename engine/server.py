"""FastAPI front end.

One background task drives `Scheduler.step()` in a tight loop on a dedicated
worker thread so the event loop stays free to push SSE chunks. Handlers encode
the prompt in a thread pool (the tokenizer is pure Python), enqueue a Request,
then `async for` over that request's queue and yield SSE events.

    POST /generate              raw prompt, SSE stream of text deltas
    POST /v1/chat/completions   OpenAI-style, stream=true or false
    GET  /                      chat UI (engine/static/index.html)
    GET  /health, /stats

Run:  uvicorn engine.server:app --host 0.0.0.0 --port 8000
Env:  MODEL_DIR, DEVICE, DTYPE, MAX_BATCH, MAX_LEN
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .cache import KVPool
from .config import QWEN3_0_6B, pick_device, pick_dtype
from .model import Qwen3
from .sampling import SamplingParams
from .sched import Request, Scheduler
from .tokenizer import IncrementalDetokenizer, load_tokenizer
from .weights import resolve_model_dir


class Engine:
    """Model, cache, scheduler, and the loop that steps them."""

    def __init__(self) -> None:
        self.cfg = QWEN3_0_6B
        self.device = pick_device(os.environ.get("DEVICE"))
        self.dtype = pick_dtype(self.device, os.environ.get("DTYPE"))
        self.max_batch = int(os.environ.get("MAX_BATCH", "32"))
        self.max_len = int(os.environ.get("MAX_LEN", "2048"))
        model_dir = resolve_model_dir(self.cfg, os.environ.get("MODEL_DIR"))
        self.tokenizer = load_tokenizer(model_dir)
        self.model = Qwen3.from_pretrained(self.cfg, self.device, self.dtype, str(model_dir))
        self.cache = KVPool(self.cfg, self.max_batch, self.max_len, self.device, self.dtype)
        self.sched = Scheduler(self.model, self.cache, self.max_batch)
        self.wake = asyncio.Event()
        self.step_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="step")
        self.tok_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tok")
        self.steps = 0
        self.t_start = time.perf_counter()

    async def run_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.sched.has_work():
                self.wake.clear()
                await self.wake.wait()
            await loop.run_in_executor(self.step_pool, self.sched.step)
            self.steps += 1

    async def encode(self, text: str) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.tok_pool, self.tokenizer.encode, text)

    def submit(self, prompt_ids: list[int], max_new_tokens: int, sampling: SamplingParams) -> tuple[Request, asyncio.Queue]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[int | None] = asyncio.Queue()

        def on_token(_req: Request, tok: int | None) -> None:
            # Called from the step thread; hop back onto the loop.
            loop.call_soon_threadsafe(q.put_nowait, tok)

        req = Request(prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, sampling=sampling, on_token=on_token)
        self.sched.add(req)
        self.wake.set()
        return req, q

    async def stream_text(self, req: Request, q: asyncio.Queue):
        """Yield (delta_text, token_id) pairs. Detokenizes incrementally, off the loop."""
        loop = asyncio.get_running_loop()
        detok = IncrementalDetokenizer(self.tokenizer)
        while True:
            tok = await q.get()
            if tok is None:
                tail = detok.flush()
                if tail:
                    yield tail, None
                return
            delta = await loop.run_in_executor(self.tok_pool, detok.push, tok)
            if delta:
                yield delta, tok


engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = Engine()
    task = asyncio.create_task(engine.run_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="inference-engine", lifespan=lifespan)


def sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _params(temperature: float, top_k: int, top_p: float) -> SamplingParams:
    return SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)


def _check(prompt_len: int, max_tokens: int) -> None:
    assert engine is not None
    if prompt_len + max_tokens > engine.max_len:
        raise HTTPException(400, f"prompt ({prompt_len}) + max_tokens ({max_tokens}) > MAX_LEN ({engine.max_len})")


# --- /generate ---------------------------------------------------------------


class GenerateBody(BaseModel):
    prompt: str
    max_tokens: int = Field(128, ge=1)
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    chat: bool = False  # wrap the prompt in the chat template


@app.post("/generate")
async def generate(body: GenerateBody):
    assert engine is not None
    text = body.prompt
    if body.chat:
        text = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": body.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    ids = await engine.encode(text)
    _check(len(ids), body.max_tokens)
    req, q = engine.submit(ids, body.max_tokens, _params(body.temperature, body.top_k, body.top_p))

    async def events():
        async for delta, tok in engine.stream_text(req, q):
            yield sse({"text": delta, "token_id": tok})
        yield sse({"done": True, "finish_reason": req.finish_reason,
                   "usage": {"prompt_tokens": len(ids), "completion_tokens": len(req.output_ids)}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- /v1/chat/completions ----------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    messages: list[ChatMessage]
    model: str = QWEN3_0_6B.name
    max_tokens: int = Field(256, ge=1)
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stream: bool = False
    enable_thinking: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody):
    assert engine is not None
    text = engine.tokenizer.apply_chat_template(
        [m.model_dump() for m in body.messages], tokenize=False,
        add_generation_prompt=True, enable_thinking=body.enable_thinking,
    )
    ids = await engine.encode(text)
    _check(len(ids), body.max_tokens)
    req, q = engine.submit(ids, body.max_tokens, _params(body.temperature, body.top_k, body.top_p))
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": body.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    if body.stream:
        async def events():
            yield sse(chunk({"role": "assistant", "content": ""}))
            async for delta, _ in engine.stream_text(req, q):
                yield sse(chunk({"content": delta}))
            yield sse(chunk({}, req.finish_reason))
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    parts = [delta async for delta, _ in engine.stream_text(req, q)]
    return {"id": cid, "object": "chat.completion", "created": created, "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": req.finish_reason}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(req.output_ids),
                      "total_tokens": len(ids) + len(req.output_ids)}}


# --- introspection -----------------------------------------------------------


UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/health")
async def health():
    return {"ok": engine is not None}


@app.get("/stats")
async def stats():
    assert engine is not None
    s = engine.sched
    elapsed = time.perf_counter() - engine.t_start
    return {
        "device": str(engine.device), "dtype": str(engine.dtype),
        "max_batch": engine.max_batch, "max_len": engine.max_len,
        "kv_pool_gb": round(engine.cache.nbytes() / 1e9, 3),
        "running": len(s.running), "waiting": len(s.waiting), "finished": len(s.finished),
        "steps": engine.steps, "steps_per_sec": round(engine.steps / max(elapsed, 1e-9), 2),
        "uptime_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("engine.server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
