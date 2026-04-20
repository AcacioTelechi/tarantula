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
| `--map-model` | `gpt-4o-mini` | Model for per-chunk extraction. |
| `--reduce-model` | `gpt-4o` | Model for per-site reconciliation. |
| `--cache-ttl` | `24h` | Cache TTL (`30s`, `10m`, `24h`, `7d`, or seconds). |
| `--no-cache` | off | Ignore the cache; re-fetch and re-extract. |
| `--map-workers` | `8` | Concurrent LLM calls during map step. |
| `--reduce-workers` | `8` | Concurrent LLM calls during reduce step. |
| `-v` / `--verbose` | `0` | Increase log verbosity (repeatable). |
| `--quiet` | off | Suppress progress output on stderr. |

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
