from __future__ import annotations

from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from protego import Protego

Fetcher = Callable[[str], Awaitable[tuple[int, str]]]


class RobotsCache:
    """Async robots.txt cache keyed by host.

    On missing (404) or error (5xx) we allow - matches broad ecosystem
    convention when robots.txt cannot be determined.
    """

    def __init__(self, fetcher: Fetcher, user_agent: str) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._parsers: dict[str, Protego | None] = {}

    async def _get_parser(self, host_key: str) -> Protego | None:
        if host_key in self._parsers:
            return self._parsers[host_key]
        robots_url = host_key + "/robots.txt"
        status, body = await self._fetcher(robots_url)
        if status == 200 and body:
            self._parsers[host_key] = Protego.parse(body)
        else:
            self._parsers[host_key] = None
        return self._parsers[host_key]

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host_key = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        parser = await self._get_parser(host_key)
        if parser is None:
            return True
        return parser.can_fetch(url, self._user_agent)
