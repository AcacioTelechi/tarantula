"""Curated, read-only tools an extraction agent uses to search the crawled
corpus. Pure functions over an in-memory Corpus — no DB, no network, no shell."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SNIPPET_MAX = 240


@dataclass(frozen=True)
class PageDoc:
    url: str
    title: str | None
    text: str


class Corpus:
    """A crawl's pages as (url, title, cleaned_text), searched in memory."""

    def __init__(self, docs: list[PageDoc]) -> None:
        self.docs = docs

    @classmethod
    def from_pages(cls, pages: list[tuple[str, str | None, str]]) -> "Corpus":
        return cls([PageDoc(url, title, text) for url, title, text in pages])


def grep(
    corpus: Corpus, pattern: str, ignore_case: bool = True, max_matches: int = 50
) -> dict[str, Any]:
    """Regex-search every page's cleaned text. Returns {matches, truncated} or
    {error} on an invalid pattern — never raises."""
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return {"error": f"invalid regex: {e}"}
    matches: list[dict[str, Any]] = []
    for doc in corpus.docs:
        for line_no, line in enumerate(doc.text.splitlines(), start=1):
            if rx.search(line):
                snippet = line.strip()
                if len(snippet) > _SNIPPET_MAX:
                    snippet = snippet[:_SNIPPET_MAX] + "…"
                matches.append({"url": doc.url, "line_no": line_no, "line": snippet})
                if len(matches) >= max_matches:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


def read_page(corpus: Corpus, url: str, max_chars: int = 6000) -> dict[str, Any]:
    """Return a page's cleaned text (truncation-noted), or {error} if unknown."""
    for doc in corpus.docs:
        if doc.url == url:
            if len(doc.text) > max_chars:
                return {"url": url, "text": doc.text[:max_chars], "truncated": True}
            return {"url": url, "text": doc.text, "truncated": False}
    return {"error": f"no page with url {url!r}"}


def list_pages(corpus: Corpus) -> dict[str, Any]:
    """List every page's url, title, and character count."""
    return {"pages": [
        {"url": d.url, "title": d.title, "chars": len(d.text)} for d in corpus.docs
    ]}
