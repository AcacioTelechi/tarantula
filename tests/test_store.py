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


def test_sync_page_chunks_replaces_changed_chunks(store):
    from tarantula.chunker import Chunk
    page_id = store.save_page(
        url="https://example.com", raw_bytes=b"<a/>", http_status=200,
        content_type="text/html", fetcher="http", title="ex",
    )
    # Old chunker left a single oversized chunk for this page.
    store.save_chunk(page_id=page_id, ordinal=0, text="OLD oversized blob", token_count=9999)
    # New chunker produces two bounded chunks for the same page.
    new = [Chunk(ordinal=0, text="bounded part one", token_count=3),
           Chunk(ordinal=1, text="bounded part two", token_count=3)]
    store.sync_page_chunks(page_id, new)
    rows = store.conn.execute(
        "SELECT ordinal, text, token_count FROM chunks WHERE page_id=? ORDER BY ordinal",
        (page_id,),
    ).fetchall()
    assert rows == [(0, "bounded part one", 3), (1, "bounded part two", 3)]
    # FTS index reflects the replacement (old text gone, new text present).
    assert store.conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'oversized'"
    ).fetchone()[0] == 0
    assert store.conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'bounded'"
    ).fetchone()[0] == 2


def test_sync_page_chunks_preserves_embeddings_when_unchanged(store):
    from tarantula.chunker import Chunk
    page_id = store.save_page(
        url="https://example.com", raw_bytes=b"<a/>", http_status=200,
        content_type="text/html", fetcher="http", title="ex",
    )
    cid = store.save_chunk(page_id=page_id, ordinal=0, text="stable text", token_count=2)
    store.save_chunk_embedding(cid, [0.1, 0.2, 0.3], model="stub")
    # Re-syncing identical chunks must NOT delete/re-create rows (id stable,
    # embedding preserved) so cached embeddings are not needlessly recomputed.
    store.sync_page_chunks(page_id, [Chunk(ordinal=0, text="stable text", token_count=2)])
    row = store.conn.execute(
        "SELECT id, embedding IS NOT NULL FROM chunks WHERE page_id=?", (page_id,)
    ).fetchone()
    assert row[0] == cid
    assert row[1] == 1


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
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("u", "v")
    # Two separate crawls, each with a chunk containing "founded".
    crawl_a = store.start_crawl(run_id, "https://a.com")
    pa = store.save_page(url="https://a.com", raw_bytes=b"<a/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="A")
    store.link_page(crawl_a, pa, depth=0, parent_url=None)
    ca = store.save_chunk(pa, 0, "Company A was founded in 1998.", 6)

    crawl_b = store.start_crawl(run_id, "https://b.com")
    pb = store.save_page(url="https://b.com", raw_bytes=b"<b/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="B")
    store.link_page(crawl_b, pb, depth=0, parent_url=None)
    cb = store.save_chunk(pb, 0, "Company B was founded in 2001.", 6)

    hits_a = store.bm25_top_k(crawl_a, "founded", k=5)
    assert [h[0] for h in hits_a] == [ca]
    hits_b = store.bm25_top_k(crawl_b, "founded", k=5)
    assert [h[0] for h in hits_b] == [cb]


def test_vector_top_k_scopes_to_crawl_and_orders_by_cosine(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("u", "v")
    crawl_a = store.start_crawl(run_id, "https://a.com")
    pa = store.save_page(url="https://a.com", raw_bytes=b"<a/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="A")
    store.link_page(crawl_a, pa, depth=0, parent_url=None)
    ca1 = store.save_chunk(pa, 0, "aligned chunk in crawl A", 5)
    ca2 = store.save_chunk(pa, 1, "orthogonal chunk in crawl A", 5)
    store.save_chunk_embedding(ca1, [1.0, 0.0, 0.0], model="stub")
    store.save_chunk_embedding(ca2, [0.0, 1.0, 0.0], model="stub")

    crawl_b = store.start_crawl(run_id, "https://b.com")
    pb = store.save_page(url="https://b.com", raw_bytes=b"<b/>",
                         http_status=200, content_type="text/html",
                         fetcher="http", title="B")
    store.link_page(crawl_b, pb, depth=0, parent_url=None)
    cb = store.save_chunk(pb, 0, "chunk in crawl B", 4)
    store.save_chunk_embedding(cb, [1.0, 0.0, 0.0], model="stub")

    hits = store.vector_top_k(crawl_a, [1.0, 0.0, 0.0], k=5)
    # Only crawl A chunks; ca1 first (aligned), ca2 second (orthogonal).
    assert [h[0] for h in hits] == [ca1, ca2]
    assert hits[0][1] > hits[1][1]


def test_get_chunks_for_crawl_returns_text_and_url(tmp_path):
    store, crawl_id, c1, c2 = _seed_two_chunks(tmp_path)
    rows = store.get_chunks_for_crawl(crawl_id)
    # c1 is on page linked at depth=0, c2 at depth=1 — ordering asserts the ORDER BY.
    assert [r[0] for r in rows] == [c1, c2]
    by_id = {r[0]: r for r in rows}
    assert by_id[c1][2] == "https://example.com/a"
    assert "founded" in by_id[c1][4]


def test_bm25_top_k_tolerates_natural_language_with_punctuation(tmp_path):
    store, crawl_id, c1, _c2 = _seed_two_chunks(tmp_path)
    # Queries that would crash raw FTS5 MATCH: email-like, colon, trailing dot.
    hits = store.bm25_top_k(crawl_id, "contact us at hello@example.com.", k=5)
    # Top hit should be the chunk mentioning the email address.
    assert hits and hits[0][0] != c1  # c2 is the email chunk


def test_bm25_top_k_returns_empty_on_empty_or_whitespace_query(tmp_path):
    store, crawl_id, _c1, _c2 = _seed_two_chunks(tmp_path)
    assert store.bm25_top_k(crawl_id, "", k=5) == []
    assert store.bm25_top_k(crawl_id, "   ", k=5) == []
    assert store.bm25_top_k(crawl_id, "!!!", k=5) == []


def test_sanitize_fts_query_strips_reserved():
    from tarantula.store import _sanitize_fts_query
    assert _sanitize_fts_query("hello@world.com v1.0:") == "hello world com v1 0"
    assert _sanitize_fts_query("") is None
    assert _sanitize_fts_query("!!!") is None
    assert _sanitize_fts_query("  foo  bar  ") == "foo bar"


def test_init_schema_creates_chunks_page_id_index(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND tbl_name='chunks' AND name='idx_chunks_page_id'"
    ).fetchall()
    assert rows, "idx_chunks_page_id should be created on init"
