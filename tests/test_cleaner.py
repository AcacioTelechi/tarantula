from tarantula.cleaner import clean_html, extract_title


def test_clean_simple_preserves_headings_and_text(fixtures_dir):
    html = (fixtures_dir / "html" / "simple.html").read_text()
    cleaned = clean_html(html, url="https://example.com/about")
    assert "Example Industries, Inc." in cleaned
    assert "founded in 1998" in cleaned
    assert "Widget Pro" in cleaned
    assert "HOME | ABOUT" not in cleaned
    assert "© 2026" not in cleaned
    assert "## Products" in cleaned or "Products" in cleaned


def test_clean_spa_shell_returns_empty(fixtures_dir):
    html = (fixtures_dir / "html" / "spa_shell.html").read_text()
    cleaned = clean_html(html, url="https://example.com/")
    assert cleaned.strip() == ""


def test_extract_title_from_head():
    html = "<html><head><title>  Hello  </title></head><body>x</body></html>"
    assert extract_title(html) == "Hello"


def test_extract_title_missing():
    html = "<html><body>x</body></html>"
    assert extract_title(html) is None
