from __future__ import annotations

from bs4 import BeautifulSoup
import trafilatura


_BOILERPLATE_TAGS = ("nav", "footer", "aside", "script", "style", "noscript", "header")


def _strip_boilerplate(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_BOILERPLATE_TAGS):
        tag.decompose()
    return str(soup)


def clean_html(html: str, url: str) -> str:
    """Extract readable content from HTML; returns '' if page looks empty/gated."""
    if not html or not html.strip():
        return ""
    prepped = _strip_boilerplate(html)
    text = trafilatura.extract(
        prepped,
        url=url,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        output_format="markdown",
    )
    if text and text.strip():
        return text.strip()
    try:
        from readability import Document
        doc = Document(prepped)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "lxml")
        fallback = soup.get_text(separator="\n").strip()
        lines = [ln.strip() for ln in fallback.splitlines() if ln.strip()]
        return "\n\n".join(lines)
    except Exception:
        return ""


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip() or None
    return None
