from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import tldextract

_BINARY_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    ".css", ".js", ".json", ".xml", ".rss",
}


def normalize_url(url: str, base: str | None = None) -> str:
    if base is not None:
        url = urljoin(base, url)
    parts = urlsplit(url)
    host = parts.hostname or ""
    host = host.lower()
    netloc = host
    if parts.port and not (
        (parts.scheme == "http" and parts.port == 80)
        or (parts.scheme == "https" and parts.port == 443)
    ):
        netloc = f"{host}:{parts.port}"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    path = parts.path
    if path == "/" and not query:
        path = ""
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def same_scope(candidate: str, seed: str, include_subdomains: bool) -> bool:
    cand_host = (urlsplit(candidate).hostname or "").lower()
    seed_host = (urlsplit(seed).hostname or "").lower()
    if cand_host == seed_host:
        return True
    if not include_subdomains:
        return False
    c = tldextract.extract(cand_host)
    s = tldextract.extract(seed_host)
    return bool(c.domain) and c.domain == s.domain and c.suffix == s.suffix


def is_html_like(url: str) -> bool:
    path = urlsplit(url).path.lower()
    if "/" in path:
        last = path.rsplit("/", 1)[-1]
    else:
        last = path
    if "." in last:
        ext = "." + last.rsplit(".", 1)[-1]
        if ext in _BINARY_EXTS:
            return False
    return True
