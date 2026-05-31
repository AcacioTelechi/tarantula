# Tarantula

A CLI that crawls websites and extracts user-defined typed variables using an LLM,
with a per-value source URL and quoted passage for auditability.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    # Optional JS rendering:
    pip install -e ".[playwright]"
    playwright install chromium

Set `OPENAI_API_KEY` (a `.env` file at the repo root is auto-loaded):

    export OPENAI_API_KEY=sk-...

## Quick start

    tarantula extract \
      --urls urls.yaml \
      --variables variables.yaml \
      --output results.json \
      --db tarantula.db \
      --data-dir ./data

Working examples live under [`examples/`](examples/).

## Configuration

Both files are YAML mappings at the root. Unknown keys are rejected (strict
schema) — typos fail fast with a validation error.

### `urls.yaml`

Two top-level keys: `defaults` (optional) and `sites` (required, non-empty).
Every field under `defaults` can be overridden per-site; per-site fields use
identical names and bounds.

```yaml
defaults:
  max_depth: 2
  max_pages: 50
  same_host_only: true
  include_subdomains: false
  respect_robots_txt: true
  rate_limit_rps: 1.5
  request_timeout_s: 20
  user_agent: "tarantula/0.1 (+contact@example.com)"

sites:
  - url: https://www.anthropic.com
    max_depth: 3
    max_pages: 10

  - url: https://openai.com        # inherits all defaults

  - url: https://docs.python.org
    include_subdomains: true
    max_pages: 10
    rate_limit_rps: 3
```

| Field | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `url` *(site only, required)* | string | — | must start with `http://` or `https://` | Seed URL for the crawl. |
| `max_depth` | int | `3` | `0..20` | BFS depth from the seed. `0` = seed only. |
| `max_pages` | int | `200` | `1..100000` | Hard cap on pages fetched per site. |
| `same_host_only` | bool | `true` | — | If `true`, stay on the exact host of the seed. |
| `include_subdomains` | bool | `false` | — | With `same_host_only: true`, also follow subdomains of the seed host. |
| `respect_robots_txt` | bool | `true` | — | Honor `robots.txt`. Turn off only for sites you own. |
| `rate_limit_rps` | float | `1.5` | `>0, ≤50` | Requests per second per host. |
| `request_timeout_s` | int | `20` | `1..600` | Per-request timeout. |
| `user_agent` | string | `"tarantula/0.1"` | — | Sent on every HTTP request. |

Notes:
- `sites` must contain at least one entry.
- Per-site values completely override the matching default (no deep merge needed — each field is a scalar).
- `defaults` itself is optional; omit it to use the built-in defaults above.

### `variables.yaml`

One top-level key: `variables`, a list of specs. Names must be unique and
match `^[a-zA-Z_][a-zA-Z0-9_]*$` (they become JSON keys in the output).

```yaml
variables:
  - name: company_name
    type: string
    description: "The official legal name of the organization that owns the site."
    required: true

  - name: founded_year
    type: integer
    description: "The year the organization was founded."
    examples:
      - input: "Founded in 1998 by Larry Page and Sergey Brin"
        output: 1998
      - input: "Since 2015, we've been helping companies..."
        output: 2015

  - name: products
    type: array
    items: string
    description: "Names of the main products or services the organization offers."

  - name: is_open_source
    type: boolean
    description: "Whether the organization ships open-source software as a primary offering."
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | Identifier (letters, digits, underscore; must not start with a digit). |
| `type` | enum | yes | One of `string`, `integer`, `number`, `boolean`, `array`. |
| `description` | string | yes | Instruction for the LLM. Be specific — this drives extraction quality. |
| `required` | bool | no (default `false`) | If `true`, a missing value raises the run's exit code to `2`. |
| `items` | enum | only for `array` | Scalar element type: `string`, `integer`, `number`, `boolean`. Required for arrays; forbidden otherwise. |
| `examples` | list | no | Few-shot pairs `{input: "...", output: <value>}`. Output must match the variable's type. |

Tips for good extraction:
- Write descriptions as if briefing a researcher — say what counts and what doesn't.
- Add `examples` for anything format-sensitive (dates, numbers, classifications).
- Prefer `array` over comma-joined strings so downstream code doesn't re-parse.
- Use `required: true` sparingly — it only affects the exit code, not the JSON.

#### Extraction strategies (`extraction_type`)

Each variable can declare an `extraction_type` that controls how Tarantula
finds its value:

| `extraction_type` | Behaviour |
|---|---|
| `retrieval` *(default)* | Top-k chunk retrieval (BM25 + embeddings) followed by a single LLM extraction call — the standard pipeline. |
| `agent` | An LLM agent iteratively searches the crawled pages' cleaned text with read-only tools (regex search + page read) and returns the same quote-grounded value. Best for pattern-shaped data (CNPJ, e-mails, founding years). |

Two additional fields are valid **only** when `extraction_type: agent`:

| Field | Type | Default | Range | Meaning |
|---|---|---|---|---|
| `hint` | string | — | — | Free-text guidance given to the agent (e.g. expected format, nearby keywords). |
| `max_steps` | int | `8` | `1..50` | Tool-call budget per variable. Raise it for hard-to-find data; lower it to save tokens. |

```yaml
variables:
  # Default strategy: retrieval (top-k chunks + one LLM call).
  - name: missao
    type: string
    description: Mission statement.

  # Agent strategy: an LLM agent searches crawled pages with regex/read tools.
  - name: cnpj
    type: string
    description: Brazilian company tax id (CNPJ).
    extraction_type: agent          # retrieval (default) | agent
    hint: "14 digits formatted XX.XXX.XXX/XXXX-XX, often near the word 'CNPJ'"
    max_steps: 6                     # optional tool-call budget (default 8)
```

## Output

Results are written to `--output` (or stdout if omitted) as a single JSON
document: `{run_id, started_at, finished_at, sites: [...]}`. Each site entry
includes `seed_url`, `crawl_status`, `pages_fetched`, and a `variables` map
from variable name to `{value, source_url, quote, reasoning, required_missing}`.

## CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--urls` | — | Path to `urls.yaml`. |
| `--variables` | — | Path to `variables.yaml`. |
| `--output` | stdout | Write results JSON to this path. |
| `--db` | `tarantula.db` | SQLite cache + audit store. |
| `--data-dir` | `./data` | Raw HTML and run logs live here. |
| `--extract-model` | `gpt-4o-mini` | Model for the per-variable extraction call. |
| `--cache-ttl` | `24h` | Cache TTL (`30s`, `10m`, `24h`, `7d`, or seconds). |
| `--no-cache` | off | Ignore the cache; re-fetch and re-extract. |
| `--workers` | `8` | Concurrent per-variable LLM calls. |
| `--retrieval` | `hybrid` | `hybrid`, `bm25`, or `vec`. See [Retrieval + Extraction](#retrieval--extraction). |
| `--top-k` | `20` | Chunks per variable kept after retrieval. |
| `--retrieval-candidates` | `50` | BM25 and vector top-N fetched before RRF fusion. |
| `--embed-model` | `text-embedding-3-small` | Embedding model for chunks and queries. |
| `-v` / `--verbose` | `0` | Increase log verbosity (repeatable). |
| `--quiet` | off | Suppress progress output on stderr. |

## Retrieval + Extraction

Tarantula extracts variables in two stages:

1. **Hybrid retrieval** (BM25 + dense embeddings, fused by Reciprocal Rank
   Fusion) selects the top-k chunks most relevant to each variable.
2. **Contextual extraction** sends *one LLM call per variable*, passing the
   variable spec and its top-k chunks as sources. The model returns the final
   typed value plus the source URL and verbatim quote that support it; a
   post-hoc check nulls the value if the quote isn't actually a substring of
   the cited source (tolerant of whitespace differences).

That's `N_variables` LLM calls per site regardless of chunk count — a big
drop from the earlier map-per-chunk + reduce-per-variable pipeline.

Embeddings are persisted in the SQLite DB — re-runs are free. BM25 uses
SQLite's built-in FTS5 extension, so `--retrieval bm25` runs without an
`OPENAI_API_KEY` (the embedding step is skipped entirely).

### Inspecting retrieval (`tarantula retrieve`)

Run retrieval against an existing DB without re-crawling or re-extracting.
Handy for iterating on a variable's `description` or `examples`:

    tarantula retrieve \
      --db tarantula.db --data-dir ./data \
      --variables variables.yaml \
      --seed-url https://example.com \
      --variable founded_year \
      --k 10 --show-text

Useful flags:
- `--mode hybrid|bm25|vec` — isolate one retrieval leg. `bm25` never calls
  the embedding API, so no `OPENAI_API_KEY` is needed for that mode.
- `--crawl-id <N>` — pick a specific crawl (overrides `--seed-url`).
- Omit `--variable` to retrieve for every variable in the file.
- `--json` — machine-readable output for piping into other tools.

Each hit prints its RRF score plus the `bm25_rank` and `vec_rank` that
contributed, so you can see exactly which leg pulled the chunk in.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | OK |
| 2    | Some required variables missing |
| 3    | One or more crawls partial |
| 4    | One or more crawls failed |

Highest-severity wins when multiple apply.

## Tests

    pytest                 # default suite (no live LLM, no Playwright)
    pytest -m playwright   # requires Chromium installed
    pytest -m live         # requires OPENAI_API_KEY

## Further reading

Full design spec: [`docs/superpowers/specs/2026-04-19-tarantula-design.md`](docs/superpowers/specs/2026-04-19-tarantula-design.md).
