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


def _hard_split_tokens(text: str, max_tokens: int) -> list[str]:
    """Slice text into pieces of at most max_tokens by token boundary.

    Last resort for content with no whitespace to break on (minified blobs).
    """
    tokens = _ENCODING.encode(text)
    return [
        _ENCODING.decode(tokens[i:i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _enforce_max(para: str, max_tokens: int) -> list[str]:
    """Split a paragraph so every piece is at most max_tokens.

    Prefer line boundaries; any single line still over the limit is hard
    token-sliced. Paragraphs already within the limit pass through unchanged.
    """
    if count_tokens(para) <= max_tokens:
        return [para]
    raw: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for line in para.split("\n"):
        lt = count_tokens(line)
        if lt > max_tokens:
            if buf:
                raw.append("\n".join(buf))
                buf, buf_tokens = [], 0
            raw.extend(_hard_split_tokens(line, max_tokens))
            continue
        if buf and buf_tokens + lt > max_tokens:
            raw.append("\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(line)
        buf_tokens += lt
    if buf:
        raw.append("\n".join(buf))
    # Per-line token counts underestimate the joined text (newline/merge
    # tokens are not counted per line), so a piece's true count can still
    # exceed the limit. Hard-split any piece that does — this guarantees
    # every returned piece is within max_tokens.
    pieces: list[str] = []
    for p in raw:
        if count_tokens(p) <= max_tokens:
            pieces.append(p)
        else:
            pieces.extend(_hard_split_tokens(p, max_tokens))
    return pieces


def chunk_text(
    text: str,
    *,
    target_tokens: int = 2000,
    overlap_tokens: int = 200,
    max_tokens: int = 8192,
) -> Iterator[Chunk]:
    """Split text into paragraph-boundary-aligned chunks with overlap.

    Greedy fill: accumulate paragraphs until adding the next would exceed
    target_tokens; then emit a chunk. Keep the trailing N tokens worth of
    paragraphs as overlap into the next chunk.

    No emitted chunk exceeds max_tokens (the embedding model's input limit):
    an oversized paragraph is split at line, then token, boundaries first.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        if text.strip():
            paragraphs = [text.strip()]
        else:
            return

    # Guarantee no single paragraph exceeds max_tokens, so the force-add
    # path below can never emit an over-limit chunk.
    paragraphs = [
        piece for p in paragraphs for piece in _enforce_max(p, max_tokens)
    ]

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
