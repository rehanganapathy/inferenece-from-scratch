"""Per-sequence contiguous KV cache.

A pool of `num_slots` slots. Each slot is one contiguous, preallocated region of
`max_len` positions for every layer. A sequence owns exactly one slot for its
whole life and writes K/V for position p at index p. No paging, no sharing.

Layout: k, v are [num_layers, num_slots, num_kv_heads, max_len, head_dim].

Every slot reserves `max_len` positions whether the sequence uses them or not.
That reservation waste is the thing paged attention fixes in a later phase.
"""

from __future__ import annotations

import torch

from .config import ModelConfig


class KVPool:
    def __init__(
        self,
        cfg: ModelConfig,
        num_slots: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.num_slots = num_slots
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        shape = (cfg.num_layers, num_slots, cfg.num_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self._free = set(range(num_slots))
        self._arange = torch.arange(max_len, device=device)
        self._slot_ids = torch.arange(num_slots, device=device)

    # --- slot bookkeeping ---------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        """Lowest free slot, so a busy pool stays packed at the bottom."""
        if not self._free:
            raise RuntimeError("KV pool exhausted")
        slot = min(self._free)
        self._free.remove(slot)
        return slot

    def release(self, slot: int) -> None:
        self._free.add(slot)

    def move(self, src: int, dst: int, length: int) -> None:
        """Relocate the first `length` positions of slot `src` into free slot `dst`.

        One copy per retirement, so every decode step can read a view instead of
        gathering the whole batch's K/V.
        """
        assert dst in self._free and src not in self._free
        self.k[:, dst, :, :length] = self.k[:, src, :, :length]
        self.v[:, dst, :, :length] = self.v[:, src, :, :length]
        self._free.remove(dst)
        self._free.add(src)

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    # --- hot path -----------------------------------------------------------

    def write(
        self,
        layer: int,
        slots: torch.Tensor | slice,  # [B] long, or a contiguous slot range
        positions: torch.Tensor,      # [B, T] long
        k: torch.Tensor,              # [B, kv_heads, T, head_dim]
        v: torch.Tensor,
    ) -> None:
        """Store k/v for `positions` of each slot. One indexed put per tensor."""
        if isinstance(slots, slice):
            slots = self._slot_ids[slots]
        # Advanced indices on dims 0 and 2 with a slice between them: torch moves
        # the indexed dims to the front, so the target is [B, T, kv_heads, head_dim].
        idx = (slots[:, None], slice(None), positions)
        self.k[layer][idx] = k.transpose(1, 2)
        self.v[layer][idx] = v.transpose(1, 2)

    def read(
        self,
        layer: int,
        slots: torch.Tensor | slice,  # [B], or a contiguous slot range
        positions: torch.Tensor,      # [B, T]
        length: int,                  # max(positions) + 1, computed once per step
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B, kv_heads, length, head_dim] K/V plus a boolean mask.

        A slice of slots returns a view of the pool: no copy. Arbitrary slot ids
        fall back to a gather, which costs a full read and write of the batch's
        K/V per layer per step and dominated decode time before the scheduler
        kept the running set packed.

        mask[b, 0, i, j] is True when query i of sequence b may attend to key j,
        i.e. j <= positions[b, i]. That is both the causal rule inside a prefill
        and the padding rule across a decode batch with ragged lengths.
        """
        if isinstance(slots, slice):
            k = self.k[layer, slots, :, :length]
            v = self.v[layer, slots, :, :length]
        else:
            k = self.k[layer, :, :, :length].index_select(0, slots)
            v = self.v[layer, :, :, :length].index_select(0, slots)
        mask = self._arange[:length][None, None, None, :] <= positions[:, None, :, None]
        return k, v, mask
