from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from . import embeddings


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


_FTS5_RESERVED = re.compile(r"[^\w\s]", re.UNICODE)


def _sanitize_fts_query(query: str) -> str | None:
    """Strip FTS5 operator/reserved characters so arbitrary natural-language
    text is safe to pass as a MATCH argument. Returns None if nothing
    indexable remains (caller should short-circuit).
    """
    cleaned = _FTS5_RESERVED.sub(" ", query)
    collapsed = " ".join(cleaned.split())
    return collapsed or None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  urls_config TEXT NOT NULL,
  variables_config TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crawls (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  seed_url TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  pages_fetched INTEGER DEFAULT 0,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  http_status INTEGER,
  content_type TEXT,
  fetched_at TEXT NOT NULL,
  fetcher TEXT NOT NULL,
  raw_path TEXT NOT NULL,
  cleaned_text TEXT,
  title TEXT,
  UNIQUE(url, content_hash)
);
CREATE TABLE IF NOT EXISTS crawl_pages (
  crawl_id INTEGER NOT NULL REFERENCES crawls(id),
  page_id INTEGER NOT NULL REFERENCES pages(id),
  discovered_at TEXT NOT NULL,
  depth INTEGER NOT NULL,
  parent_url TEXT,
  PRIMARY KEY (crawl_id, page_id)
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id),
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER,
  UNIQUE(page_id, ordinal)
);
CREATE TABLE IF NOT EXISTS chunk_extractions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  chunk_id INTEGER NOT NULL REFERENCES chunks(id),
  variable_name TEXT NOT NULL,
  found INTEGER NOT NULL,
  value_json TEXT,
  quote TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, chunk_id, variable_name)
);
CREATE TABLE IF NOT EXISTS extractions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  crawl_id INTEGER NOT NULL REFERENCES crawls(id),
  variable_name TEXT NOT NULL,
  value_json TEXT,
  source_url TEXT,
  quote TEXT,
  reasoning TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, crawl_id, variable_name)
);
CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_crawl_pages_crawl ON crawl_pages(crawl_id);
CREATE INDEX IF NOT EXISTS idx_chunk_extractions_run ON chunk_extractions(run_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
  USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


@dataclass
class PageRecord:
    id: int
    url: str
    content_hash: str
    fetched_at: str
    raw_path: str
    cleaned_text: str | None
    title: str | None


class Store:
    def __init__(self, db_path: str | Path, data_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        # Migration: add embedding columns on existing DBs.
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")}
        if "embedding" not in existing:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
        if "embedding_model" not in existing:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")
        # Backfill FTS index if chunks exist but triggers were missing
        # (covers upgrades from pre-FTS schema versions). This is safe to run
        # on every init_schema because 'rebuild' is idempotent.
        chunks_count = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if chunks_count > 0:
            self.conn.execute(
                "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
            )
        self.conn.execute(
            "UPDATE runs SET status='failed', finished_at=? "
            "WHERE status='running' AND finished_at IS NULL",
            (_now(),),
        )
        self.conn.commit()

    def start_run(self, urls_config_yaml: str, variables_config_yaml: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, urls_config, variables_config, status) "
            "VALUES (?, ?, ?, 'running')",
            (_now(), urls_config_yaml, variables_config_yaml),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=? WHERE id=?",
            (status, _now(), run_id),
        )
        self.conn.commit()

    def start_crawl(self, run_id: int, seed_url: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawls (run_id, seed_url, started_at, status) "
            "VALUES (?, ?, ?, 'running')",
            (run_id, seed_url, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_crawl(self, crawl_id: int, status: str, pages_fetched: int) -> None:
        self.conn.execute(
            "UPDATE crawls SET status=?, finished_at=?, pages_fetched=? WHERE id=?",
            (status, _now(), pages_fetched, crawl_id),
        )
        self.conn.commit()

    def _raw_path_for(self, url: str, content_hash: str) -> Path:
        host = urlsplit(url).hostname or "unknown"
        p = self.data_dir / "raw" / host
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{content_hash}.html"

    def save_page(
        self,
        url: str,
        raw_bytes: bytes,
        http_status: int,
        content_type: str | None,
        fetcher: str,
        title: str | None,
    ) -> int:
        content_hash = hashlib.sha1(raw_bytes).hexdigest()
        row = self.conn.execute(
            "SELECT id FROM pages WHERE url=? AND content_hash=?",
            (url, content_hash),
        ).fetchone()
        raw_path = self._raw_path_for(url, content_hash)
        if not raw_path.exists():
            raw_path.write_bytes(raw_bytes)
        if row:
            self.conn.execute(
                "UPDATE pages SET fetched_at=? WHERE id=?",
                (_now(), row[0]),
            )
            self.conn.commit()
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO pages (url, content_hash, http_status, content_type, "
            "fetched_at, fetcher, raw_path, title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, content_hash, http_status, content_type, _now(), fetcher,
             str(raw_path), title),
        )
        self.conn.commit()
        return cur.lastrowid

    def find_fresh_page(self, url: str, ttl_seconds: int) -> PageRecord | None:
        row = self.conn.execute(
            "SELECT id, url, content_hash, fetched_at, raw_path, cleaned_text, title "
            "FROM pages WHERE url=? "
            "AND (julianday(?) - julianday(fetched_at)) * 86400 < ? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (url, _now(), ttl_seconds),
        ).fetchone()
        if not row:
            return None
        return PageRecord(*row)

    def link_page(
        self, crawl_id: int, page_id: int, depth: int, parent_url: str | None
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO crawl_pages "
            "(crawl_id, page_id, discovered_at, depth, parent_url) "
            "VALUES (?, ?, ?, ?, ?)",
            (crawl_id, page_id, _now(), depth, parent_url),
        )
        self.conn.commit()

    def set_cleaned_text(self, page_id: int, text: str) -> None:
        self.conn.execute(
            "UPDATE pages SET cleaned_text=? WHERE id=?",
            (text, page_id),
        )
        self.conn.commit()

    def save_chunk(
        self, page_id: int, ordinal: int, text: str, token_count: int
    ) -> int:
        row = self.conn.execute(
            "SELECT id FROM chunks WHERE page_id=? AND ordinal=?",
            (page_id, ordinal),
        ).fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO chunks (page_id, ordinal, text, token_count) "
            "VALUES (?, ?, ?, ?)",
            (page_id, ordinal, text, token_count),
        )
        self.conn.commit()
        return cur.lastrowid

    def sync_page_chunks(self, page_id: int, chunks: list) -> None:
        """Make the stored chunks for a page match the given list exactly.

        If the page's chunks are unchanged (same text per ordinal) this is a
        no-op, preserving cached embeddings. Otherwise the page's chunks are
        deleted and re-inserted so a changed chunker (e.g. new token limits)
        fully replaces stale rows instead of leaving orphans. The FTS index
        stays in sync via the chunks insert/delete triggers.
        """
        existing = self.conn.execute(
            "SELECT ordinal, text FROM chunks WHERE page_id=? ORDER BY ordinal",
            (page_id,),
        ).fetchall()
        new = [(c.ordinal, c.text) for c in chunks]
        if [(o, t) for o, t in existing] == new:
            return
        self.conn.execute("DELETE FROM chunks WHERE page_id=?", (page_id,))
        for c in chunks:
            self.conn.execute(
                "INSERT INTO chunks (page_id, ordinal, text, token_count) "
                "VALUES (?, ?, ?, ?)",
                (page_id, c.ordinal, c.text, c.token_count),
            )
        self.conn.commit()

    def save_extraction(
        self,
        run_id: int,
        crawl_id: int,
        variable_name: str,
        value: Any,
        source_url: str | None,
        quote: str | None,
        reasoning: str | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO extractions "
            "(run_id, crawl_id, variable_name, value_json, source_url, quote, "
            " reasoning, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, crawl_id, variable_name,
             json.dumps(value) if value is not None else None,
             source_url, quote, reasoning, _now()),
        )
        self.conn.commit()

    def save_chunk_embedding(
        self, chunk_id: int, vec: list[float], model: str
    ) -> None:
        self.conn.execute(
            "UPDATE chunks SET embedding=?, embedding_model=? WHERE id=?",
            (embeddings.pack(vec), model, chunk_id),
        )
        self.conn.commit()

    def bm25_top_k(
        self, crawl_id: int, query: str, k: int
    ) -> list[tuple[int, float]]:
        """Returns [(chunk_id, bm25_score)] ranked best-first.

        bm25() returns a *negative* number where smaller == better. We negate so
        larger == better for downstream fusion ergonomics. The query is sanitized
        to strip FTS5 reserved characters so natural-language descriptions /
        example inputs are safe to pass verbatim. Words are joined with OR for
        inclusive matching against multi-word queries.
        """
        safe = _sanitize_fts_query(query)
        if safe is None:
            return []
        # Convert space-separated words to OR-joined for inclusive matching
        fts_query = " OR ".join(safe.split())
        rows = self.conn.execute(
            "SELECT f.rowid, -bm25(chunks_fts) AS s "
            "FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "JOIN crawl_pages cp ON cp.page_id = c.page_id "
            "WHERE cp.crawl_id = ? AND chunks_fts MATCH ? "
            "ORDER BY s DESC LIMIT ?",
            (crawl_id, fts_query, k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def vector_top_k(
        self, crawl_id: int, qvec: list[float], k: int
    ) -> list[tuple[int, float]]:
        """Brute-force cosine over all chunks in this crawl that have embeddings.

        Fine for v1: chunks per crawl are in the hundreds-to-low-thousands.
        Swap for a vector index later if profiling demands it.
        """
        rows = self.conn.execute(
            "SELECT c.id, c.embedding "
            "FROM chunks c "
            "JOIN crawl_pages cp ON cp.page_id = c.page_id "
            "WHERE cp.crawl_id = ? AND c.embedding IS NOT NULL",
            (crawl_id,),
        ).fetchall()
        scored: list[tuple[int, float]] = []
        for cid, blob in rows:
            scored.append((cid, embeddings.cosine(qvec, embeddings.unpack(blob))))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def get_chunks_for_crawl(
        self, crawl_id: int
    ) -> list[tuple[int, int, str, str | None, str]]:
        """Returns [(chunk_id, page_id, page_url, page_title, chunk_text)]."""
        rows = self.conn.execute(
            "SELECT c.id, c.page_id, p.url, p.title, c.text "
            "FROM chunks c "
            "JOIN pages p ON p.id = c.page_id "
            "JOIN crawl_pages cp ON cp.page_id = c.page_id "
            "WHERE cp.crawl_id = ? "
            "ORDER BY cp.depth, c.page_id, c.ordinal",
            (crawl_id,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
