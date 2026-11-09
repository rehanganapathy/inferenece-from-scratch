"""End-to-end through FastAPI: SSE stream matches the single-request oracle."""

import asyncio
import json
import os

import httpx
import pytest

from engine.model import generate_cached


@pytest.fixture(scope="module")
def client(model_dir, dtype):
    os.environ.setdefault("MODEL_DIR", str(model_dir))
    os.environ.setdefault("DTYPE", {"torch.float32": "fp32", "torch.bfloat16": "bf16", "torch.float16": "fp16"}[str(dtype)])
    os.environ["MAX_BATCH"] = "8"
    os.environ["MAX_LEN"] = "512"
    from engine import server

    async def run():
        async with server.lifespan(server.app):
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                yield c, server.engine

    gen = run()
    loop = asyncio.new_event_loop()
    c, eng = loop.run_until_complete(gen.__anext__())
    yield loop, c, eng
    try:
        loop.run_until_complete(gen.__anext__())
    except StopAsyncIteration:
        pass
    loop.close()


def parse_sse(body: str):
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def test_generate_streams_and_matches_oracle(client, cache, tokenizer):
    loop, c, eng = client
    prompt = "The quick brown fox"

    async def go():
        r = await c.post("/generate", json={"prompt": prompt, "max_tokens": 16})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        return r.text

    events = list(parse_sse(loop.run_until_complete(go())))
    text = "".join(e.get("text", "") for e in events)
    done = events[-1]
    assert done["done"] and done["usage"]["completion_tokens"] == 16
    ids = tokenizer.encode(prompt)
    ref = generate_cached(eng.model, cache, ids, 16, eng.cfg.eos_token_ids)
    assert text == tokenizer.decode(ref, skip_special_tokens=True)


def test_concurrent_streams_are_all_correct(client, tokenizer, prompts):
    loop, c, eng = client
    texts = [tokenizer.decode(p) for p in prompts[:4]] * 3   # 12 concurrent requests

    async def one(t):
        r = await c.post("/generate", json={"prompt": t, "max_tokens": 12})
        return "".join(e.get("text", "") for e in parse_sse(r.text))

    async def go():
        return await asyncio.gather(*(one(t) for t in texts))

    outs = loop.run_until_complete(go())
    # Identical prompts must give identical output regardless of batch-mates.
    for i in range(4):
        assert outs[i] == outs[i + 4] == outs[i + 8]
    assert eng.cache.num_free == eng.cache.num_slots


def test_chat_completions_non_stream(client):
    loop, c, _ = client

    async def go():
        r = await c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8})
        return r.json()

    j = loop.run_until_complete(go())
    assert j["choices"][0]["message"]["content"]
    assert j["usage"]["completion_tokens"] <= 8


def test_stats(client):
    loop, c, _ = client
    j = loop.run_until_complete(c.get("/stats"))
    assert j.json()["max_batch"] == 8


def test_ui_is_served(client):
    loop, c, _ = client
    r = loop.run_until_complete(c.get("/"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/v1/chat/completions" in r.text
