"""Request queue, running set, step().

    waiting  --admit-->  running  --EOS / max_tokens-->  finished

`step()` runs one decode pass over the running set, then admits from waiting
into free slots, prefilling each admission. Requests join and leave the batch
mid-flight. No preemption, no chunked prefill.

`mode="static"` reproduces static batching: admit only when the running set is
empty, so a batch finishes when its longest sequence finishes and slots sit idle
in between. Same code, one branch, so the two are directly comparable.

`NaiveScheduler` is the no-cache baseline behind the same interface: one request
at a time, every token recomputes the whole prefix.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from .cache import KVPool
from .model import Qwen3
from .sampling import SamplingParams, sample

OnToken = Callable[["Request", int | None], None]  # None => finished


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int
    sampling: SamplingParams = SamplingParams()
    ignore_eos: bool = False
    on_token: OnToken | None = None
    id: int = field(default_factory=itertools.count().__next__)

    output_ids: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)
    t_submit: float = 0.0
    t_admit: float = 0.0
    t_finish: float = 0.0
    slot: int | None = None
    finish_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    @property
    def seq_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def emit(self, tok: int | None) -> None:
        if self.on_token:
            self.on_token(self, tok)


@dataclass
class StepStats:
    t: float
    running: int          # sequences in the decode pass this step
    waiting: int
    admitted: int
    tokens: int           # tokens produced this step (decode + prefill first tokens)
    max_slots: int

    @property
    def occupancy(self) -> float:
        return self.running / self.max_slots


class Scheduler:
    def __init__(
        self,
        model: Qwen3,
        cache: KVPool,
        max_batch_size: int | None = None,
        mode: Literal["continuous", "static"] = "continuous",
        eos_ids: tuple[int, ...] | None = None,
        seed: int = 0,
    ):
        self.model = model
        self.cache = cache
        self.max_batch_size = min(max_batch_size or cache.num_slots, cache.num_slots)
        self.mode = mode
        self.eos_ids = eos_ids if eos_ids is not None else model.cfg.eos_token_ids
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.stats: list[StepStats] = []
        self.lock = threading.Lock()
        self.gen = torch.Generator(device=model.device).manual_seed(seed)

    # --- queue --------------------------------------------------------------

    def add(self, req: Request) -> Request:
        if req.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if len(req.prompt_ids) + req.max_new_tokens > self.cache.max_len:
            raise ValueError(
                f"prompt {len(req.prompt_ids)} + max_new_tokens {req.max_new_tokens} "
                f"exceeds slot length {self.cache.max_len}"
            )
        req.t_submit = time.perf_counter()
        with self.lock:
            self.waiting.append(req)
        return req

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    # --- step ---------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> StepStats:
        n_running = len(self.running)
        tokens = 0
        if self.running:
            tokens += self._decode(self.running)
            self._retire()
        admitted = self._admit()
        tokens += admitted
        self._retire()
        st = StepStats(
            t=time.perf_counter(), running=n_running, waiting=len(self.waiting),
            admitted=admitted, tokens=tokens, max_slots=self.max_batch_size,
        )
        self.stats.append(st)
        return st

    def _admit(self) -> int:
        if self.mode == "static" and self.running:
            return 0
        admitted = 0
        while len(self.running) < self.max_batch_size and self.cache.num_free:
            with self.lock:
                if not self.waiting:
                    break
                req = self.waiting.popleft()
            req.slot = self.cache.alloc()
            req.t_admit = time.perf_counter()
            self._prefill(req)
            self.running.append(req)
            admitted += 1
        return admitted

    def _prefill(self, req: Request) -> None:
        dev = self.model.device
        ids = torch.tensor([req.prompt_ids], device=dev)
        pos = torch.arange(len(req.prompt_ids), device=dev)[None]
        logits = self.model(ids, pos, self.cache, slice(req.slot, req.slot + 1))
        tok = int(sample(logits, [req.sampling], self.gen).item())
        self._append(req, tok)

    def _decode(self, batch: list[Request]) -> int:
        dev = self.model.device
        # _retire keeps the running set on slots [0, B), in order, so attention
        # reads a view of the pool rather than gathering it.
        assert all(r.slot == i for i, r in enumerate(batch))
        ids = torch.tensor([[r.output_ids[-1]] for r in batch], device=dev)
        pos = torch.tensor([[r.seq_len - 1] for r in batch], device=dev)
        logits = self.model(ids, pos, self.cache, slice(0, len(batch)))
        toks = sample(logits, [r.sampling for r in batch], self.gen).tolist()
        for req, tok in zip(batch, toks):
            self._append(req, tok)
        return len(batch)

    def _append(self, req: Request, tok: int) -> None:
        req.output_ids.append(tok)
        req.token_times.append(time.perf_counter())
        req.emit(tok)
        if not req.ignore_eos and tok in self.eos_ids:
            req.finish_reason = "stop"
        elif len(req.output_ids) >= req.max_new_tokens:
            req.finish_reason = "length"

    def _retire(self) -> None:
        still = []
        for req in self.running:
            if req.finished:
                req.t_finish = time.perf_counter()
                self.cache.release(req.slot)
                req.slot = None
                self.finished.append(req)
                req.emit(None)
            else:
                still.append(req)
        # Pack survivors into slots [0, B). Slots below i are held by earlier
        # survivors, so slot i is free whenever req.slot > i. Only the positions
        # already written move: the newest token's K/V lands on the next decode.
        still.sort(key=lambda r: r.slot)
        for i, req in enumerate(still):
            if req.slot != i:
                self.cache.move(req.slot, i, req.seq_len - 1)
                req.slot = i
        self.running = still


class NaiveScheduler:
    """Week 1 baseline: no KV cache, one request at a time. Same interface."""

    def __init__(self, model: Qwen3, eos_ids: tuple[int, ...] | None = None, seed: int = 0):
        self.model = model
        self.eos_ids = eos_ids if eos_ids is not None else model.cfg.eos_token_ids
        self.max_batch_size = 1
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.stats: list[StepStats] = []
        self.lock = threading.Lock()
        self.gen = torch.Generator(device=model.device).manual_seed(seed)

    def add(self, req: Request) -> Request:
        req.t_submit = time.perf_counter()
        with self.lock:
            self.waiting.append(req)
        return req

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    @torch.no_grad()
    def step(self) -> StepStats:
        admitted = 0
        if not self.running:
            with self.lock:
                req = self.waiting.popleft() if self.waiting else None
            if req is None:
                st = StepStats(time.perf_counter(), 0, 0, 0, 0, 1)
                self.stats.append(st)
                return st
            req.t_admit = time.perf_counter()
            self.running = [req]
            admitted = 1
        req = self.running[0]
        ids = req.prompt_ids + req.output_ids
        dev = self.model.device
        x = torch.tensor([ids], device=dev)
        pos = torch.arange(len(ids), device=dev)[None]
        logits = self.model(x, pos)
        tok = int(sample(logits, [req.sampling], self.gen).item())
        req.output_ids.append(tok)
        req.token_times.append(time.perf_counter())
        req.emit(tok)
        if not req.ignore_eos and tok in self.eos_ids:
            req.finish_reason = "stop"
        elif len(req.output_ids) >= req.max_new_tokens:
            req.finish_reason = "length"
        if req.finished:
            req.t_finish = time.perf_counter()
            self.finished.append(req)
            self.running = []
            req.emit(None)
        st = StepStats(time.perf_counter(), 1, len(self.waiting), admitted, 1, 1)
        self.stats.append(st)
        return st
