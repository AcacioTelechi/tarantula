# Tarantula — Design Spec

**Date:** 2026-04-19
**Status:** Draft, pending user review

## 1. Purpose

Tarantula is a Python CLI that, given a list of URLs and a user-defined set of variables, crawls each site, cleans the content, and uses an LLM to extract structured values for each variable — with a verifiable source URL and quoted passage per value.

## 2. Scope

**In scope (v1):**
- Python 3.11+ CLI tool.
- Static HTML fetching via `httpx`, with Playwright fallback for JS-gated pages.
- Per-URL configurable scope, depth, page cap, rate limit, robots.txt compliance.
- OpenAI-backed extraction with JSON-schema structured outputs.
- Map-reduce extraction: per-chunk extraction, then per-site reconciliation.
- SQLite cache + audit store; raw HTML persisted on disk.
- Output JSON with per-variable source URL and quoted passage.
- User-defined variable schema (typed) with optional examples.

**Out of scope (v1):**
- Non-HTML content types (PDF, DOCX, etc.).
- Sitemap.xml-driven discovery.
- Authentication / cookies / login flows.
- Parallel processing across sites (sequential in v1).
- Resume-from-failure (cache makes re-runs cheap).
- HTTP service or library packaging (CLI only for now; core logic kept cleanly separable).

## 3. High-level architecture

Pipeline per seed URL:

```
URLs file ──┐
            ├──► [Crawler] ──► [Cleaner] ──► [Chunker] ──► [Per-page Extractor] ──► [Reducer] ──► Results
Vars config ┘          │              │             │                    │                  │
                       ▼              ▼             ▼                    ▼                  ▼
                   ┌───────────────────── SQLite (cache + audit) ─────────────────────┐
                   │ runs, crawls, pages, chunks, chunk_extractions, extractions     │
                   └──────────────────────────────────────────────────────────────────┘
                                     Raw HTML on disk: data/raw/<host>/<sha1>.html
```

Sites are processed **sequentially**; pages within a site are fetched **concurrently** with per-host rate-limiting and a small semaphore.

## 4. Modules

Each module is a focused file with a single responsibility, testable in isolation:

- `cli.py` — argument parsing, orchestration, output formatting.
- `config.py` — Pydantic models for `urls.yaml` and `variables.yaml`; validates on load.
- `crawler.py` — BFS per URL, honors scope/depth/page-cap, robots.txt, rate-limit; static fetch + Playwright fallback.
- `cleaner.py` — HTML → structured plain text via `trafilatura`, fallback `readability-lxml` + BeautifulSoup.
- `chunker.py` — splits cleaned text into ~2k-token chunks with ~200-token overlap, snapping to paragraph boundaries.
- `extractor.py` — per-chunk LLM call; all variables in one call; validates returned quotes.
- `reducer.py` — one LLM call per site to reconcile candidates into final values.
- `store.py` — SQLite access layer: cache lookup, upserts, extraction records.
- `llm.py` — `LLMClient` interface; `OpenAIClient` implementation using JSON-schema structured outputs.
- `logging_setup.py` — structured JSONL logging per run; CLI verbosity levels.

**Formal boundaries:** `LLMClient` is the only pluggable interface. Nothing else is abstracted behind an interface in v1.

## 5. Configuration

### 5.1 `urls.yaml`

```yaml
defaults:
  max_depth: 3
  max_pages: 200
  same_host_only: true          # if false, allow same registrable domain
  include_subdomains: false
  respect_robots_txt: true
  rate_limit_rps: 1.5           # requests/sec per host
  request_timeout_s: 20
  user_agent: "tarantula/0.1 (+contact@example.com)"

sites:
  - url: https://example.com
    max_depth: 4                # per-site overrides
    max_pages: 500
  - url: https://docs.other.com
    include_subdomains: true
```

Any field under `defaults` can be overridden per-site.

### 5.2 `variables.yaml`

```yaml
variables:
  - name: company_name
    type: string
    description: "The official legal name of the organization."
    required: true

  - name: founded_year
    type: integer
    description: "The year the organization was founded."
    examples:
      - input: "Founded in 1998 by Larry Page and Sergey Brin"
        output: 1998

  - name: products
    type: array
    items: string
    description: "Names of products or services the company offers."

  - name: has_careers_page
    type: boolean
    description: "Whether the site has a page listing job openings."
```

Supported types: `string`, `integer`, `number`, `boolean`, `array` (requires `items` of another supported scalar type). Optional fields per variable: `required` (bool), `examples` (list of `{input, output}`), `description` (string).

### 5.3 CLI

```bash
tarantula extract \
  --urls urls.yaml \
  --variables variables.yaml \
  --output results.json \
  --db tarantula.db \
  --data-dir ./data \
  [--no-cache | --cache-ttl 24h] \
  [--map-model gpt-4o-mini] \
  [--reduce-model gpt-4o] \
  [--max-tokens 2000000] \
  [--verbose | --debug] \
  [--log-llm-io] \
  [--quiet]
```

## 6. Data model

### 6.1 SQLite schema

```sql
CREATE TABLE runs (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  urls_config     TEXT NOT NULL,
  variables_config TEXT NOT NULL,
  status          TEXT NOT NULL       -- running | ok | failed
);

CREATE TABLE crawls (
  id              INTEGER PRIMARY KEY,
  run_id          INTEGER NOT NULL REFERENCES runs(id),
  seed_url        TEXT NOT NULL,
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  pages_fetched   INTEGER DEFAULT 0,
  status          TEXT NOT NULL       -- running | ok | partial | failed
);

CREATE TABLE pages (
  id              INTEGER PRIMARY KEY,
  url             TEXT NOT NULL,
  content_hash    TEXT NOT NULL,      -- sha1 of raw bytes
  http_status     INTEGER,
  content_type    TEXT,
  fetched_at      TEXT NOT NULL,
  fetcher         TEXT NOT NULL,      -- http | playwright
  raw_path        TEXT NOT NULL,
  cleaned_text    TEXT,
  title           TEXT,
  UNIQUE(url, content_hash)
);

-- depth and parent_url live here: the same page can be discovered at
-- different depths from different parents across different crawls
CREATE TABLE crawl_pages (
  crawl_id        INTEGER NOT NULL REFERENCES crawls(id),
  page_id         INTEGER NOT NULL REFERENCES pages(id),
  discovered_at   TEXT NOT NULL,
  depth           INTEGER NOT NULL,
  parent_url      TEXT,
  PRIMARY KEY (crawl_id, page_id)
);

CREATE TABLE chunks (
  id              INTEGER PRIMARY KEY,
  page_id         INTEGER NOT NULL REFERENCES pages(id),
  ordinal         INTEGER NOT NULL,
  text            TEXT NOT NULL,
  token_count     INTEGER,
  UNIQUE(page_id, ordinal)
);

CREATE TABLE chunk_extractions (
  id              INTEGER PRIMARY KEY,
  run_id          INTEGER NOT NULL REFERENCES runs(id),
  chunk_id        INTEGER NOT NULL REFERENCES chunks(id),
  variable_name   TEXT NOT NULL,
  found           INTEGER NOT NULL,
  value_json      TEXT,
  quote           TEXT,
  created_at      TEXT NOT NULL,
  UNIQUE(run_id, chunk_id, variable_name)
);

CREATE TABLE extractions (
  id              INTEGER PRIMARY KEY,
  run_id          INTEGER NOT NULL REFERENCES runs(id),
  crawl_id        INTEGER NOT NULL REFERENCES crawls(id),
  variable_name   TEXT NOT NULL,
  value_json      TEXT,
  source_url      TEXT,
  quote           TEXT,
  reasoning       TEXT,
  created_at      TEXT NOT NULL,
  UNIQUE(run_id, crawl_id, variable_name)
);

CREATE INDEX idx_pages_url ON pages(url);
CREATE INDEX idx_crawl_pages_crawl ON crawl_pages(crawl_id);
CREATE INDEX idx_chunk_extractions_run ON chunk_extractions(run_id);
```

### 6.2 Filesystem layout

```
data/
  raw/
    example.com/
      a1b2c3...e9.html     # raw bytes keyed by content_hash
  logs/
    run-<id>.jsonl         # structured log per run
tarantula.db
results.json               # latest --output
```

### 6.3 Cache semantics

**Two layers, different lifetimes:**

- **Page fetch cache (cross-run):** on fetch, look up `pages` by URL. If found and `fetched_at` is within the TTL (default 24h), reuse — skip network. Otherwise fetch, compute new `content_hash`; if unchanged, update `fetched_at` in place; if changed, insert a new row with the new hash. This is what makes re-runs cheap even across different variable configs.
- **Extraction cache (per-run):** `chunk_extractions` is scoped to `run_id`. A new run redoes extraction — which is necessary, because changing `variables.yaml` changes the LLM call. Within a run, extraction is idempotent: re-invoking the map step on an already-processed chunk is a no-op.

**Flags:** `--no-cache` forces refetch. `--cache-ttl <duration>` (e.g., `1h`, `7d`) overrides the page-fetch TTL.

## 7. Crawling behavior

For each seed URL, BFS starting at depth 0:

1. **robots.txt** — loaded once per host per run; cached in memory. Disallowed URLs are skipped when `respect_robots_txt: true`.
2. **Fetch (static)** — `httpx` async, up to 5 redirects, per-host token bucket enforcing `rate_limit_rps`, concurrency capped at 4 in-flight requests per host.
3. **Clean (first pass)** — via `cleaner.py` (Section 8).
4. **JS-gated fallback** — if the response is HTML but cleaned text is empty (or the body is effectively an SPA shell), refetch once with headless Playwright/Chromium and re-clean. Recorded in `pages.fetcher` (`http` or `playwright`).
5. **Link extraction** — pull `<a href>` from the raw HTML; normalize (strip fragments, sort query params, lowercase host); filter out non-HTTP(S), mailto/tel, non-HTML extensions (`.pdf`, `.zip`, images). Skipped URLs are logged for future work.
6. **Scope filter** — same host (default) or same registrable domain via `tldextract` if `include_subdomains: true`.
7. **Deduplication** — in-memory `seen` set per crawl, keyed by normalized URL.
8. **Enqueue** — links added at `depth + 1` if `depth < max_depth` and `pages_fetched < max_pages`.

### Failure handling

- **4xx (except 429):** record page with status; no retry; not a crawl failure.
- **429 / 5xx / network errors:** exponential backoff, up to 3 retries, then recorded as failed page.
- **Seed URL fails after retries:** crawl marked `failed`; extraction still runs on captured pages; crawl status set to `partial`.
- **Ctrl-C:** run status set to `failed`; DB writes flushed cleanly.

## 8. Extraction pipeline

### 8.1 Cleaning

- `trafilatura` as primary — strips nav/footer/ads, preserves headings/lists/paragraphs.
- `readability-lxml` + BeautifulSoup fallback when `trafilatura` returns empty.
- Output is plain text with `## Heading` markers so chunks retain structural context.
- Cached in `pages.cleaned_text`.

### 8.2 Chunking

- Split into ~2k-token chunks (OpenAI `tiktoken` counter) with ~200-token overlap.
- Snap boundaries to paragraph breaks when possible to avoid cutting sentences.
- Each chunk gets a stable `(page_id, ordinal)` ID; persisted in `chunks`.

### 8.3 Map — per-chunk extraction

One LLM call per `(chunk)`, extracting all variables in a single call:

- **Model:** `gpt-4o-mini` by default; configurable via `--map-model`.
- **Structured output:** JSON schema compiled from `variables.yaml`. Each variable becomes `{found: boolean, value: <typed> | null, quote: string | null}`. OpenAI's `response_format: json_schema` enforces shape.
- **Prompt:** system prompt defines task — "extract only if the chunk clearly supports the answer; `quote` must be a verbatim substring of the chunk." User prompt includes variable specs (with examples), page URL + title as context, and chunk text.
- **Quote validation:** after the call, for each `found: true` variable, verify the returned `quote` is a substring of the chunk text. If not, demote to `found: false` (no retry). This blocks hallucinated quotes.
- **Persistence:** one row per `(run_id, chunk_id, variable_name)` in `chunk_extractions`. Map step is cache-aware: chunks already extracted for this run are skipped.

### 8.4 Reduce — per-site reconciliation

After a site's map step completes, gather all `chunk_extractions` with `found=1` for that crawl:

- **Model:** `gpt-4o` by default; configurable via `--reduce-model`.
- **Structured output:** final schema `{variable_name: {value, source_url, quote, reasoning}}` compiled from the variable specs.
- **Prompt:** provide the variable specs plus, per variable, the list of candidates with `{value, quote, source_url, page_title}`. Ask for the best-supported value; when candidates disagree, prefer authoritative sources (URL paths like `/about`, higher-level pages). One-sentence `reasoning` per variable.
- **Array variables:** reducer unions and semantically de-dupes candidates.
- If a variable has zero `found` candidates, final `value` is `null`; if also `required: true`, mark `required_missing: true` in output.
- **Persistence:** one row per `(run_id, crawl_id, variable_name)` in `extractions`.

### 8.5 Determinism

- `temperature=0` on both map and reduce.
- `seed` parameter set where supported.
- Results are not bit-identical run-to-run but are stable enough for diffing.

### 8.6 Cost/safety guardrails

- `--max-tokens` per run (default 2,000,000). If exceeded mid-run, extraction stops for remaining chunks, reducer runs on what we have, crawl status set to `partial`.

## 9. Output format

`results.json`:

```json
{
  "run_id": 42,
  "started_at": "2026-04-19T17:02:11Z",
  "finished_at": "2026-04-19T17:06:48Z",
  "sites": [
    {
      "seed_url": "https://example.com",
      "crawl_status": "ok",
      "pages_fetched": 87,
      "variables": {
        "company_name": {
          "value": "Example Industries, Inc.",
          "source_url": "https://example.com/about",
          "quote": "Example Industries, Inc. is a privately held company headquartered in Chicago.",
          "reasoning": "Found on the /about page, the authoritative source for the legal name."
        },
        "founded_year": {
          "value": 1998,
          "source_url": "https://example.com/about/history",
          "quote": "The company was founded in 1998 by Jane Doe.",
          "reasoning": "Explicitly stated on the history page."
        },
        "products": {
          "value": ["Widget Pro", "Widget Lite", "Widget Cloud"],
          "sources": [
            {"value_item": "Widget Pro", "source_url": "https://example.com/products/pro", "quote": "Widget Pro is our flagship offering..."},
            {"value_item": "Widget Lite", "source_url": "https://example.com/products/lite", "quote": "Widget Lite, for small teams..."},
            {"value_item": "Widget Cloud", "source_url": "https://example.com/products", "quote": "...and Widget Cloud for enterprises."}
          ],
          "reasoning": "Union of product names found across /products pages."
        },
        "has_careers_page": {
          "value": null,
          "source_url": null,
          "quote": null,
          "reasoning": "No page in the crawl mentioned careers or hiring.",
          "required_missing": true
        }
      }
    }
  ]
}
```

**Shape rules:**
- Scalar types: single `value`, single `source_url`, single `quote`.
- Array types: `value` is the merged list; `sources` pairs each item with its source.
- `value: null` when no chunk supported the variable; add `required_missing: true` when the variable was declared `required`.
- `reasoning` is always a short single sentence.

**Stdout behavior:**
- Without `--output`: pretty-print JSON to stdout.
- With `--output path.json`: write to file; print a one-line human summary to stdout.
- `--quiet`: suppress stdout summary.

**Exit codes (highest-severity wins):**
- `0` — all crawls `ok`, no required missing.
- `2` — one or more required variables missing.
- `3` — one or more crawls `partial`.
- `4` — one or more crawls `failed`.

## 10. Error handling

**Input validation:** Pydantic validation on both YAML configs at startup. Clear field-level errors. Exit non-zero before any network/LLM call.

**Crawl errors:** retries with backoff (Section 7). Seed failures don't abort the run — other sites continue.

**LLM errors:**
- Rate limit / transient: exponential backoff, up to 5 attempts.
- Parse failures on structured output: retry once, then skip the chunk (logged warning, row not written, reducer uses remaining candidates).
- Quote validation failure: demote to `found: false`; no retry.
- Token budget exceeded: stop extracting further chunks for that site, run reducer on what we have, mark crawl `partial`.

**Persistence errors:** per-page/per-chunk transactions. Startup recovery marks any run with `status=running` and no `finished_at` as `failed`.

**Logging:**
- Default: concise progress per site.
- `--verbose`: per-page fetch logs.
- `--debug`: adds LLM request/response summaries (no full prompts unless `--log-llm-io`).
- Structured JSONL at `data/logs/run-<id>.jsonl` always written.

## 11. Testing strategy

**Layout:** `tests/` using `pytest`; fixture HTML in `tests/fixtures/`.

**Unit (majority of suite):**
- `cleaner`, `chunker`, URL normalization, robots parsing, scope filtering, quote verification — pure functions against fixtures. No network, no LLM.
- Extractor + reducer tested with a `FakeLLMClient` returning scripted JSON responses.

**Integration:**
- Crawler exercised with `pytest-httpserver` serving fixture pages, robots.txt, 429s, redirects, and a JS-gated page (Playwright path gated by `--run-playwright` marker).
- SQLite store end-to-end with a temp DB: insert, upsert on hash change, cache hit/miss, TTL expiry.

**End-to-end (opt-in):**
- Local fixture site + `variables.yaml` with knowable answers. Marked `@pytest.mark.live`; requires real OpenAI key. Excluded from default `pytest` run.

**Coverage target:** 85% line coverage excluding the real-OpenAI path in `llm.py`.

## 12. Dependencies (expected)

- `httpx[http2]`
- `playwright` (Chromium) — optional install extra
- `beautifulsoup4`, `lxml`, `trafilatura`, `readability-lxml`
- `tldextract`
- `pydantic` (v2), `pyyaml`
- `tiktoken`
- `openai` (official SDK)
- `typer` or `click` for the CLI
- `pytest`, `pytest-httpserver`, `pytest-asyncio` for tests

## 13. Future work (explicitly deferred)

- Non-HTML content types (PDF, DOCX).
- Sitemap.xml-driven discovery.
- Authentication / cookies / login flows.
- Parallel processing across seed URLs.
- Resume-from-failed-run.
- Embedding + semantic retrieval (alternative to map-reduce for very large sites).
- Confidence scores per extraction.
- Swap to Anthropic / provider-agnostic via `LLMClient`.
