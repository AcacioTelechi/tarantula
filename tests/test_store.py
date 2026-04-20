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
    s2 = Store(db, data_dir=tmp_path / "data"); s2.init_schema()
    # Embedding columns still present after re-init.
    cols = {r[1] for r in s2.conn.execute("PRAGMA table_info(chunks)")}
    assert "embedding" in cols and "embedding_model" in cols
    # FTS trigger still fires after re-init.
    page_id = s2.save_page(
        url="https://example.com", raw_bytes=b"<a/>",
        http_status=200, content_type="text/html",
        fetcher="http", title="ex",
    )
    s2.save_chunk(page_id=page_id, ordinal=0,
                  text="idempotent trigger check banana", token_count=4)
    hits = s2.conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'banana'"
    ).fetchall()
    assert len(hits) == 1


def test_init_schema_backfills_fts_for_preexisting_chunks(tmp_path):
    """Simulate upgrading a DB that has chunks but no populated FTS index.

    We create the DB, insert a chunk, then drop the FTS index and
    re-run init_schema — the chunk must become searchable again.
    """
    import sqlite3
    from tarantula.store import Store
    db = tmp_path / "t.db"
    s1 = Store(db, data_dir=tmp_path / "data")
    s1.init_schema()
    page_id = s1.save_page(
        url="https://example.com", raw_bytes=b"<a/>",
        http_status=200, content_type="text/html",
        fetcher="http", title="ex",
    )
    s1.save_chunk(page_id=page_id, ordinal=0,
                  text="pre-existing chunk with unique token zorblatt", token_count=6)
    # Simulate a pre-FTS schema by dropping the FTS table and its triggers.
    s1.conn.execute("DROP TABLE chunks_fts")
    for t in ("chunks_ai", "chunks_ad", "chunks_au"):
        s1.conn.execute(f"DROP TRIGGER IF EXISTS {t}")
    s1.conn.commit()
    s1.conn.close()

    s2 = Store(db, data_dir=tmp_path / "data")
    s2.init_schema()
    hits = s2.conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'zorblatt'"
    ).fetchall()
    assert len(hits) == 1


def _seed_two_chunks(tmp_path):
    """Helper: crawl with two pages, one chunk each, linked to a crawl."""
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("urls", "vars")
    crawl_id = store.start_crawl(run_id, "https://example.com")
    p1 = store.save_page(url="https://example.com/a", raw_bytes=b"<a/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="A")
    p2 = store.save_page(url="https://example.com/b", raw_bytes=b"<b/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="B")
    store.link_page(crawl_id, p1, depth=0, parent_url=None)
    store.link_page(crawl_id, p2, depth=1, parent_url="https://example.com/a")
    c1 = store.save_chunk(p1, 0, "The company was founded in 1998 by two engineers.", 10)
    c2 = store.save_chunk(p2, 0, "Contact us at hello@example.com for sales.", 8)
    return store, crawl_id, c1, c2


def test_save_chunk_embedding_and_read_back(tmp_path):
    from tarantula.embeddings import unpack
    store, _crawl_id, c1, _c2 = _seed_two_chunks(tmp_path)
    store.save_chunk_embedding(c1, [0.1, 0.2, 0.3], model="stub")
    row = store.conn.execute(
        "SELECT embedding, embedding_model FROM chunks WHERE id=?", (c1,)
    ).fetchone()
    assert row[1] == "stub"
    got = unpack(row[0])
    assert [round(x, 3) for x in got] == [0.1, 0.2, 0.3]


def test_bm25_top_k_scopes_to_crawl(tmp_path):
    store, crawl_id, c1, _c2 = _seed_two_chunks(tmp_path)
    hits = store.bm25_top_k(crawl_id, "founded engineers", k=5)
    assert hits and hits[0][0] == c1


def test_vector_top_k_scopes_to_crawl_and_orders_by_cosine(tmp_path):
    store, crawl_id, c1, c2 = _seed_two_chunks(tmp_path)
    # Give c1 a vector aligned with the query; c2 an orthogonal one.
    store.save_chunk_embedding(c1, [1.0, 0.0, 0.0], model="stub")
    store.save_chunk_embedding(c2, [0.0, 1.0, 0.0], model="stub")
    hits = store.vector_top_k(crawl_id, [1.0, 0.0, 0.0], k=5)
    assert [h[0] for h in hits] == [c1, c2]
    assert hits[0][1] > hits[1][1]


def test_get_chunks_for_crawl_returns_text_and_url(tmp_path):
    store, crawl_id, c1, c2 = _seed_two_chunks(tmp_path)
    rows = store.get_chunks_for_crawl(crawl_id)
    by_id = {r[0]: r for r in rows}
    assert by_id[c1][2] == "https://example.com/a"
    assert "founded" in by_id[c1][4]
    assert set(by_id) == {c1, c2}
