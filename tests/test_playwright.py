import pytest
from tarantula.playwright_fetcher import playwright_fetch


@pytest.mark.playwright
@pytest.mark.asyncio
async def test_playwright_renders_spa(httpserver):
    spa_html = (
        "<html><head><title>SPA</title></head><body>"
        "<div id='root'></div>"
        "<script>document.getElementById('root').innerText='hello from js'</script>"
        "</body></html>"
    )
    httpserver.expect_request("/").respond_with_data(spa_html, content_type="text/html")
    status, raw, title = await playwright_fetch(httpserver.url_for("/"), timeout_s=10)
    assert status == 200
    assert b"hello from js" in raw
    assert title == "SPA"
