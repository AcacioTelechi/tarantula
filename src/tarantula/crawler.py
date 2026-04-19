from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from bs4 import BeautifulSoup

from .cleaner import clean_html, extract_title
from .config import SiteConfig
from .robots import RobotsCache
from .store import Store
from .urls import is_html_like, normalize_url, same_scope

log = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    crawl_id: int
    pages_fetched: int
    status: str  # ok | partial | failed


PlaywrightFetcher = Callable[[str, int], Awaitable[tuple[int, bytes, str | None]]]


async def crawl_site(
    *,
    store: Store,
    run_id: int,
    site: SiteConfig,
    cache_ttl_seconds: int = 86400,
    playwright_fetcher: PlaywrightFetcher | None = None,
) -> CrawlResult:
    crawl_id = store.start_crawl(run_id, seed_url=site.url)
    pages_fetched = 0
    status = "ok"

    headers = {"User-Agent": site.user_agent}
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    timeout = httpx.Timeout(site.request_timeout_s)

    async with httpx.AsyncClient(
        headers=headers, follow_redirects=True, timeout=timeout, limits=limits, http2=True
    ) as client:
        async def http_fetch(url: str) -> tuple[int, str]:
            try:
                r = await client.get(url)
                return r.status_code, r.text
            except Exception as e:
                log.warning("robots fetch failed %s: %s", url, e)
                return 0, ""

        robots = RobotsCache(http_fetch, user_agent=site.user_agent)
        bucket = _TokenBucket(rate_rps=site.rate_limit_rps)
        sem = asyncio.Semaphore(4)

        seed = site.url
        seen: set[str] = {seed, normalize_url(seed)}
        q: deque[tuple[str, int, str | None]] = deque([(seed, 0, None)])

        while q and pages_fetched < site.max_pages:
            url, depth, parent = q.popleft()

            if site.respect_robots_txt and not await robots.allowed(url):
                log.info("robots disallow: %s", url)
                continue

            cached = store.find_fresh_page(url, ttl_seconds=cache_ttl_seconds)
            if cached:
                page_id = cached.id
                with open(cached.raw_path, "rb") as f:
                    raw_bytes = f.read()
                html = raw_bytes.decode("utf-8", errors="replace")
            else:
                await bucket.acquire()
                try:
                    async with sem:
                        resp = await _fetch_with_retries(client, url)
                except Exception as e:
                    log.warning("fetch failed %s: %s", url, e)
                    status = "partial"
                    continue
                if resp is None or not (200 <= resp.status_code < 300):
                    continue
                html = resp.text
                raw_bytes = resp.content
                title = extract_title(html)
                page_id = store.save_page(
                    url=url,
                    raw_bytes=raw_bytes,
                    http_status=resp.status_code,
                    content_type=resp.headers.get("content-type"),
                    fetcher="http",
                    title=title,
                )

            cleaned = store.find_fresh_page(url, ttl_seconds=cache_ttl_seconds)
            cleaned_text = cleaned.cleaned_text if cleaned else None
            if cleaned_text is None:
                cleaned_text = clean_html(html, url=url)
                if not cleaned_text and playwright_fetcher is not None:
                    log.info("empty clean, retrying with playwright: %s", url)
                    try:
                        status_code, raw_bytes_pw, title_pw = await playwright_fetcher(
                            url, site.request_timeout_s
                        )
                        html_pw = raw_bytes_pw.decode("utf-8", errors="replace")
                        page_id = store.save_page(
                            url=url, raw_bytes=raw_bytes_pw, http_status=status_code,
                            content_type="text/html", fetcher="playwright",
                            title=title_pw or extract_title(html_pw),
                        )
                        cleaned_text = clean_html(html_pw, url=url)
                        html = html_pw
                    except Exception as e:
                        log.warning("playwright fetch failed %s: %s", url, e)
                store.set_cleaned_text(page_id, cleaned_text or "")

            store.link_page(crawl_id, page_id, depth=depth, parent_url=parent)
            pages_fetched += 1

            if depth >= site.max_depth:
                continue

            for link in _extract_links(html, base=url):
                if not is_html_like(link):
                    continue
                if not same_scope(link, seed=site.url, include_subdomains=site.include_subdomains):
                    continue
                norm = normalize_url(link)
                if norm in seen:
                    continue
                seen.add(norm)
                q.append((norm, depth + 1, url))

    store.finish_crawl(crawl_id, status=status, pages_fetched=pages_fetched)
    return CrawlResult(crawl_id=crawl_id, pages_fetched=pages_fetched, status=status)


class _TokenBucket:
    def __init__(self, rate_rps: float) -> None:
        self._interval = 1.0 / rate_rps if rate_rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_running_loop().time()


async def _fetch_with_retries(
    client: httpx.AsyncClient, url: str, retries: int = 3
) -> httpx.Response | None:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = await client.get(url)
            if r.status_code in (429,) or 500 <= r.status_code < 600:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            await asyncio.sleep(min(2 ** attempt, 30))
    if last_exc is not None:
        raise last_exc
    return None


def _extract_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        try:
            out.append(normalize_url(href, base=base))
        except Exception:
            continue
    return out
