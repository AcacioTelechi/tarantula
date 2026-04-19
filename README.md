# Tarantula

A CLI that crawls websites and extracts user-defined typed variables using an LLM, with per-value source URL and quoted passage.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    # Optional JS rendering:
    pip install -e ".[playwright]"
    playwright install chromium

Set `OPENAI_API_KEY`:

    export OPENAI_API_KEY=sk-...

## Usage

    tarantula extract \
      --urls urls.yaml \
      --variables variables.yaml \
      --output results.json \
      --db tarantula.db \
      --data-dir ./data

Config formats: see `docs/superpowers/specs/2026-04-19-tarantula-design.md` sections 5.1 (urls.yaml) and 5.2 (variables.yaml).

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
