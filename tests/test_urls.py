from tarantula.urls import normalize_url, same_scope, is_html_like


def test_normalize_strips_fragment():
    assert normalize_url("https://example.com/a#x") == "https://example.com/a"


def test_normalize_lowercases_host():
    assert normalize_url("https://Example.COM/A") == "https://example.com/A"


def test_normalize_sorts_query_params():
    assert normalize_url("https://a.com/?b=2&a=1") == "https://a.com/?a=1&b=2"


def test_normalize_removes_default_ports():
    assert normalize_url("https://a.com:443/x") == "https://a.com/x"
    assert normalize_url("http://a.com:80/x") == "http://a.com/x"


def test_normalize_strips_trailing_slash_on_root():
    assert normalize_url("https://a.com/") == "https://a.com"
    assert normalize_url("https://a.com/x/") == "https://a.com/x/"


def test_normalize_joins_relative():
    assert normalize_url("/x?a=1", base="https://a.com/y") == "https://a.com/x?a=1"


def test_same_scope_same_host():
    assert same_scope("https://a.com/x", seed="https://a.com", include_subdomains=False)
    assert not same_scope("https://b.com/x", seed="https://a.com", include_subdomains=False)


def test_same_scope_subdomain_behavior():
    assert not same_scope("https://docs.a.com/x", seed="https://a.com", include_subdomains=False)
    assert same_scope("https://docs.a.com/x", seed="https://a.com", include_subdomains=True)
    assert not same_scope("https://b.com/x", seed="https://a.com", include_subdomains=True)


def test_is_html_like_skips_binary_extensions():
    assert not is_html_like("https://a.com/doc.pdf")
    assert not is_html_like("https://a.com/img.png")
    assert not is_html_like("https://a.com/archive.zip")


def test_is_html_like_allows_html_and_bare_paths():
    assert is_html_like("https://a.com/page")
    assert is_html_like("https://a.com/page.html")
    assert is_html_like("https://a.com/")
