from tarantula.agent_tools import Corpus, grep, read_page, list_pages


def _corpus():
    return Corpus.from_pages([
        ("https://a.example/", "Home", "Welcome to A.\nCNPJ: 12.345.678/0001-90\n"),
        ("https://a.example/about", "About", "Founded in 1998.\nContact us anytime."),
    ])


def test_grep_finds_matches_with_url_and_line_no():
    out = grep(_corpus(), r"CNPJ:\s*([0-9./-]+)")
    assert out["truncated"] is False
    assert len(out["matches"]) == 1
    m = out["matches"][0]
    assert m["url"] == "https://a.example/"
    assert m["line_no"] == 2
    assert "12.345.678/0001-90" in m["line"]


def test_grep_is_case_insensitive_by_default():
    out = grep(_corpus(), "founded")
    assert [m["url"] for m in out["matches"]] == ["https://a.example/about"]


def test_grep_case_sensitive_when_requested():
    out = grep(_corpus(), "founded", ignore_case=False)
    assert out["matches"] == []


def test_grep_invalid_regex_returns_error_observation():
    out = grep(_corpus(), "(unclosed")
    assert "error" in out
    assert "matches" not in out


def test_grep_caps_matches_and_flags_truncation():
    corpus = Corpus.from_pages([("u", None, "x\n" * 100)])
    out = grep(corpus, "x", max_matches=10)
    assert len(out["matches"]) == 10
    assert out["truncated"] is True


def test_read_page_returns_text_for_known_url():
    out = read_page(_corpus(), "https://a.example/about")
    assert out["truncated"] is False
    assert "Founded in 1998." in out["text"]


def test_read_page_truncates_long_text():
    corpus = Corpus.from_pages([("u", None, "y" * 50)])
    out = read_page(corpus, "u", max_chars=10)
    assert out["truncated"] is True
    assert len(out["text"]) == 10


def test_read_page_unknown_url_returns_error():
    out = read_page(_corpus(), "https://nope")
    assert "error" in out


def test_list_pages_returns_url_title_chars():
    out = list_pages(_corpus())
    assert out["pages"][0] == {
        "url": "https://a.example/", "title": "Home",
        "chars": len("Welcome to A.\nCNPJ: 12.345.678/0001-90\n"),
    }


def test_list_pages_caps_entries_and_reports_total():
    corpus = Corpus.from_pages([(f"https://x/{i}", f"t{i}", "x") for i in range(500)])
    out = list_pages(corpus, max_pages=50)
    assert len(out["pages"]) == 50
    assert out["total"] == 500
    assert out["truncated"] is True


def test_list_pages_small_corpus_not_truncated():
    out = list_pages(Corpus.from_pages([("u", "t", "x")]))
    assert out["truncated"] is False
    assert out["total"] == 1
