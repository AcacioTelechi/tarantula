import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from tarantula.store import Store, PageRecord


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s.init_schema()
    return s


def test_init_creates_tables(store):
    tables = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"runs", "crawls", "pages", "crawl_pages", "chunks",
            "chunk_extractions", "extractions"} <= tables


def test_start_run_returns_id(store):
    run_id = store.start_run(urls_config_yaml="x", variables_config_yaml="y")
    row = store.conn.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
    assert row[0] == "running"


def test_finish_run_sets_status_and_timestamp(store):
    run_id = store.start_run("", "")
    store.finish_run(run_id, status="ok")
    row = store.conn.execute(
        "SELECT status, finished_at FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row[0] == "ok"
    assert row[1] is not None


def test_save_page_upserts_on_content_hash(store, tmp_path):
    pid1 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    pid2 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    assert pid1 == pid2

    pid3 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v2</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    assert pid3 != pid1


def test_save_page_writes_raw_file(store, tmp_path):
    store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    raw_files = list((tmp_path / "data" / "raw").rglob("*.html"))
    assert len(raw_files) == 1
    assert raw_files[0].read_bytes() == b"<html>v1</html>"


def test_find_fresh_page_respects_ttl(store):
    store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    assert store.find_fresh_page("https://a.com/x", ttl_seconds=3600) is not None
    assert store.find_fresh_page("https://a.com/x", ttl_seconds=0) is None


def test_link_page_to_crawl(store):
    run_id = store.start_run("", "")
    crawl_id = store.start_crawl(run_id, seed_url="https://a.com")
    page_id = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    store.link_page(crawl_id, page_id, depth=1, parent_url="https://a.com")
    row = store.conn.execute(
        "SELECT depth, parent_url FROM crawl_pages WHERE crawl_id=? AND page_id=?",
        (crawl_id, page_id),
    ).fetchone()
    assert row == (1, "https://a.com")


def test_cleaned_text_upsert(store):
    pid = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    store.set_cleaned_text(pid, "hello world")
    row = store.conn.execute("SELECT cleaned_text FROM pages WHERE id=?", (pid,)).fetchone()
    assert row[0] == "hello world"


def test_chunk_and_extraction_roundtrip(store):
    run_id = store.start_run("", "")
    pid = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    cid = store.save_chunk(page_id=pid, ordinal=0, text="hello", token_count=1)
    store.save_chunk_extraction(
        run_id=run_id, chunk_id=cid, variable_name="v1",
        found=True, value="hello", quote="hello",
    )
    rows = list(store.iter_chunk_extractions(run_id=run_id, crawl_id=None))
    assert len(rows) == 1
    assert rows[0].variable_name == "v1"
    assert rows[0].found is True
    assert rows[0].value == "hello"


def test_mark_orphan_runs_as_failed_on_init(tmp_path):
    s = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s.init_schema()
    rid = s.start_run("", "")
    s.conn.close()
    s2 = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s2.init_schema()
    status = s2.conn.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()[0]
    assert status == "failed"


def test_init_schema_adds_embedding_columns_and_fts(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(chunks)")}
    assert "embedding" in cols
    assert "embedding_model" in cols
    fts = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    assert fts is not None


def test_fts_is_populated_by_save_chunk(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    # Set up minimal page to reference
    page_id = store.save_page(
        url="https://example.com", raw_bytes=b"<html/>",
        http_status=200, content_type="text/html",
        fetcher="http", title="ex",
    )
    store.save_chunk(page_id=page_id, ordinal=0,
                     text="the quick brown fox jumps", token_count=5)
    hits = store.conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'quick'"
    ).fetchall()
    assert len(hits) == 1


def test_init_schema_idempotent_on_existing_db(tmp_path):
    from tarantula.store import Store
    db = tmp_path / "t.db"
    s1 = Store(db, data_dir=tmp_path / "data"); s1.init_schema(); s1.conn.close()
    s2 = Store(db, data_dir=tmp_path / "data"); s2.init_schema()  # should not raise
