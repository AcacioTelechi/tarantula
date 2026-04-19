from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    ordinal: int
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(
    text: str,
    *,
    target_tokens: int = 2000,
    overlap_tokens: int = 200,
) -> Iterator[Chunk]:
    """Split text into paragraph-boundary-aligned chunks with overlap.

    Greedy fill: accumulate paragraphs until adding the next would exceed
    target_tokens; then emit a chunk. Keep the trailing N tokens worth of
    paragraphs as overlap into the next chunk.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        if text.strip():
            yield Chunk(ordinal=0, text=text.strip(), token_count=count_tokens(text))
        return

    para_tokens = [count_tokens(p) for p in paragraphs]

    ordinal = 0
    i = 0
    n = len(paragraphs)
    while i < n:
        buf: list[str] = []
        buf_tokens = 0
        j = i
        while j < n and (buf_tokens + para_tokens[j] <= target_tokens or not buf):
            buf.append(paragraphs[j])
            buf_tokens += para_tokens[j]
            j += 1
        chunk_str = "\n\n".join(buf).strip()
        yield Chunk(ordinal=ordinal, text=chunk_str, token_count=buf_tokens)
        ordinal += 1
        if j >= n:
            break
        tail_tokens = 0
        k = j
        while k > i and tail_tokens < overlap_tokens:
            k -= 1
            tail_tokens += para_tokens[k]
        i = max(k, i + 1)
