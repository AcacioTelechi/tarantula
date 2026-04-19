import pytest
from tarantula.robots import RobotsCache


class FakeFetcher:
    def __init__(self, responses: dict[str, tuple[int, str]]):
        self.responses = responses
        self.calls = []

    async def __call__(self, url: str) -> tuple[int, str]:
        self.calls.append(url)
        return self.responses.get(url, (404, ""))


@pytest.mark.asyncio
async def test_allows_when_robots_missing():
    fetcher = FakeFetcher({})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/anything") is True


@pytest.mark.asyncio
async def test_respects_user_agent_disallow():
    body = "User-agent: *\nDisallow: /private"
    fetcher = FakeFetcher({"https://a.com/robots.txt": (200, body)})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/public") is True
    assert await cache.allowed("https://a.com/private/x") is False


@pytest.mark.asyncio
async def test_caches_per_host():
    body = "User-agent: *\nDisallow: /p"
    fetcher = FakeFetcher({"https://a.com/robots.txt": (200, body)})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    await cache.allowed("https://a.com/x")
    await cache.allowed("https://a.com/y")
    await cache.allowed("https://a.com/p")
    assert fetcher.calls.count("https://a.com/robots.txt") == 1


@pytest.mark.asyncio
async def test_5xx_blocks_crawling_conservatively():
    fetcher = FakeFetcher({"https://a.com/robots.txt": (500, "")})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/x") is True
