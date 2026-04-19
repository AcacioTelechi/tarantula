from __future__ import annotations


async def playwright_fetch(url: str, timeout_s: int) -> tuple[int, bytes, str | None]:
    """Fetch a URL with headless Chromium. Returns (status, raw_html_bytes, title)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            resp = await page.goto(url, timeout=timeout_s * 1000, wait_until="networkidle")
            status = resp.status if resp else 0
            content = await page.content()
            title = await page.title()
            return status, content.encode("utf-8"), title or None
        finally:
            await browser.close()
