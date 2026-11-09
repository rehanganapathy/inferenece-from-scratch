"""Tokenizer loading and incremental detokenization.

Decoding the whole output every token is O(n^2) in pure Python and shows up in
profiles well before the GPU does. `IncrementalDetokenizer` keeps two offsets
and only re-decodes the tail needed to resolve multi-token characters.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_dir: Path | str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(model_dir))


class IncrementalDetokenizer:
    """Yields text deltas as token ids arrive.

    prefix_offset..read_offset is the already-emitted tail kept for context so
    byte-level BPE merges and spacing decode the same as a one-shot decode.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_ids: list[int] | None = None):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(full) > len(prefix) and not full.endswith("�"):
            delta = full[len(prefix):]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.ids)
            self.text += delta
            return delta
        return ""

    def flush(self) -> str:
        """Emit whatever is still pending (e.g. a trailing partial character)."""
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        delta = full[len(prefix):]
        self.prefix_offset = self.read_offset = len(self.ids)
        self.text += delta
        return delta
