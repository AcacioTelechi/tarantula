import pytest
from tarantula.config import SiteConfig, SiteDefaults
from tarantula.crawler import crawl_site, CrawlResult
from tarantula.store import Store


def _site(url: str, **overrides) -> SiteConfig:
    d = SiteDefaults().model_dump()
    d.update(overrides)
    return SiteConfig(url=url, **d)


@pytest.mark.asyncio
async def test_crawler_respects_depth(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        '<html><body><a href="/a">A</a></body></html>', content_type="text/html"
    )
    httpserver.expect_request("/a").respond_with_data(
        '<html><body><a href="/b">B</a></body></html>', content_type="text/html"
    )
    httpserver.expect_request("/b").respond_with_data(
        '<html><body>end</body></html>', content_type="text/html"
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=1, max_pages=100, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = set(_crawl_urls(store, result.crawl_id))
    assert httpserver.url_for("/") in urls
    assert httpserver.url_for("/a") in urls
    assert httpserver.url_for("/b") not in urls


@pytest.mark.asyncio
async def test_crawler_respects_same_host(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        f'<html><body><a href="https://other.example/x">out</a><a href="/a">in</a></body></html>',
        content_type="text/html",
    )
    httpserver.expect_request("/a").respond_with_data(
        '<html><body>a</body></html>', content_type="text/html"
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=2, max_pages=100, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = _crawl_urls(store, result.crawl_id)
    assert all("other.example" not in u for u in urls)


@pytest.mark.asyncio
async def test_crawler_respects_max_pages(httpserver, tmp_path):
    for i in range(20):
        links = "".join(f'<a href="/p{j}">P{j}</a>' for j in range(20))
        httpserver.expect_request(f"/p{i}").respond_with_data(
            f"<html><body>{links}</body></html>", content_type="text/html"
        )
    httpserver.expect_request("/").respond_with_data(
        "".join(f'<a href="/p{j}">P{j}</a>' for j in range(20)),
        content_type="text/html",
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=5, max_pages=5, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    assert result.pages_fetched <= 5


@pytest.mark.asyncio
async def test_crawler_honors_robots_disallow(httpserver, tmp_path):
    httpserver.expect_request("/robots.txt").respond_with_data(
        "User-agent: *\nDisallow: /private\n", content_type="text/plain"
    )
    httpserver.expect_request("/").respond_with_data(
        '<a href="/public">pub</a><a href="/private/x">priv</a>',
        content_type="text/html",
    )
    httpserver.expect_request("/public").respond_with_data("ok", content_type="text/html")

    site = _site(httpserver.url_for("/"), max_depth=2, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = _crawl_urls(store, result.crawl_id)
    assert httpserver.url_for("/public") in urls
    assert not any("/private" in u for u in urls)


@pytest.mark.asyncio
async def test_crawler_handles_4xx_without_aborting(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        '<a href="/ok">ok</a><a href="/missing">missing</a>',
        content_type="text/html",
    )
    httpserver.expect_request("/ok").respond_with_data("yay", content_type="text/html")
    httpserver.expect_request("/missing").respond_with_data("gone", status=404)
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=2, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_crawler_uses_cache_on_repeat(httpserver, tmp_path):
    request_count = {"n": 0}
    def handler(_req):
        request_count["n"] += 1
        from werkzeug.wrappers import Response
        return Response("<html>x</html>", content_type="text/html")
    httpserver.expect_request("/").respond_with_handler(handler)
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=0, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    await crawl_site(store=store, run_id=run_id, site=site)
    await crawl_site(store=store, run_id=run_id, site=site)
    assert request_count["n"] == 1


def _crawl_urls(store: Store, crawl_id: int) -> list[str]:
    return [r[0] for r in store.conn.execute(
        "SELECT p.url FROM pages p JOIN crawl_pages cp ON cp.page_id=p.id "
        "WHERE cp.crawl_id=?", (crawl_id,)
    ).fetchall()]
