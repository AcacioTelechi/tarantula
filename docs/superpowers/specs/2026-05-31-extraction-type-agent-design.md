# Design: per-variable `extraction_type` with an agent strategy

**Date:** 2026-05-31
**Status:** Approved (pre-implementation)

## Problem

Today every variable is extracted the same way: retrieve top-k chunks for the
variable, then make one LLM call to extract a typed value with a grounding
quote. This works well for prose-y facts but is flaky for pattern-shaped data
(CNPJ, e-mails, phone numbers, founding years) that a regex sweep over the whole
crawl would nail directly.

We want a second, opt-in extraction strategy: a small agent that iteratively
searches the crawled corpus with curated read-only tools (regex grep, page
read) and returns the same typed, source-grounded result.

## Goals

- Add `extraction_type: retrieval | agent` to the variables config, defaulting
  to `retrieval` so all existing configs keep working unchanged.
- Implement an agent strategy that searches **cleaned page text** with a fixed
  set of safe, read-only tools — no shell, no arbitrary code execution.
- Produce output identical in shape to the retrieval strategy, so persistence
  (`store.save_extraction`), the run JSON, and `tarantula flatten` all work
  without changes.

## Non-goals

- No raw-HTML search (cleaned text only) — keeps quote grounding consistent with
  the existing validation path. May revisit later behind a flag.
- No native LLM function-calling abstraction. The agent runs as a JSON-action
  loop over the existing `complete_json` interface.
- No shell access in any form.

## Decisions (from brainstorming)

1. **Tool surface:** curated read-only tools — `grep`, `read_page`,
   `list_pages`, `answer`. No shell.
2. **Corpus:** cleaned text only (`pages.cleaned_text`), so the `quote` grounding
   reuses the existing `_quote_in_any` check.
3. **Config knobs:** `extraction_type`, optional `hint`, optional `max_steps`.
4. **Loop mechanism:** JSON-action loop via existing `complete_json` (Approach A),
   reusing `FakeLLMClient` for tests; no new client surface.

## Design

### 1. Config schema (`config.py`)

Add three fields to `VariableSpec`:

- `extraction_type: Literal["retrieval", "agent"] = "retrieval"`
- `hint: str | None = None`
- `max_steps: int | None = None` — `Field(ge=1, le=50)`; a runtime default of
  **8** is applied when omitted.

A `model_validator` forbids `hint` / `max_steps` unless
`extraction_type == "agent"` (mirrors the existing array-only-`items` rule).

Example:

```yaml
variables:
  - name: cnpj
    type: string
    description: Brazilian company tax id
    extraction_type: agent
    hint: "14 digits formatted XX.XXX.XXX/XXXX-XX, usually near the word 'CNPJ'"
    max_steps: 6
```

### 2. Corpus + tools (new module `agent_tools.py`)

A pure, DB-free corpus so the tools are trivially testable:

- `PageDoc(url, title, text)`; `Corpus` wraps `list[PageDoc]`.
- `grep(corpus, pattern, ignore_case=True, max_matches=50)` →
  `[{url, line_no, line}]`, snippet-capped. Invalid regex returns an **error
  observation**, never raises.
- `read_page(corpus, url, max_chars=6000)` → text, truncation-noted; unknown url
  → error observation.
- `list_pages(corpus)` → `[{url, title, chars}]`.

### 3. Agent loop (new module `agent_extractor.py`)

`extract_variable_agent(client, variable, corpus, model, max_steps) -> payload`:

- **System prompt:** the variable spec (name / type / description / `hint` /
  examples), the action protocol, and the grounding rule — the `quote` must be a
  verbatim substring of a page's cleaned text; answer `null` if unsupported.
- **Action schema** (one strict-mode object): `required: [thought, action]`,
  `action` enum `grep | read_page | list_pages | answer`, plus nullable params
  (`pattern`, `ignore_case`, `url`) and the answer payload (`value`,
  `source_url`, `quote`, `reasoning`; arrays use `sources[]`) — same shape as the
  retrieval extractor.
- **Loop:** maintain a transcript; each step calls `complete_json`, executes the
  action, and appends the observation, repeating until `answer` or `max_steps`.
  On `answer`, validate with the existing `_quote_in_any` grounding check against
  the corpus. If grounding fails, feed back `"quote not found verbatim in <url>;
  try again"` and keep looping (step-bounded self-correction). Exhausting
  `max_steps` → `value=None`, reasoning notes exhaustion, `required_missing` per
  spec.
- `extract_all_agent(client, variables, corpus, model, max_workers)` mirrors
  `extract_all` (thread pool, preserves input order).

**Output payload matches the retrieval extractor exactly** — scalars
`{value, source_url, quote, reasoning, required_missing}`; arrays
`{value, sources, reasoning, required_missing}` — so all downstream code is
untouched.

### 4. CLI integration (`cli.py` `run_pipeline`)

- Split `vars_cfg.variables` by `extraction_type`.
- Retrieval path runs only for retrieval-type vars; agent vars skip embeddings +
  retrieval entirely (the embed pass is guarded on retrieval vars existing — a
  cost bonus).
- Agent path builds a `Corpus` from the already-computed `pages_with_text` and
  runs `extract_all_agent`.
- Merge both result dicts and reorder to config order → `reduced` (downstream
  persistence/output unchanged).
- Reporter shows the split, e.g. `extracting 40 retrieval + 3 agent variable(s)`.

### 5. Error handling

Invalid regex, unknown url, and malformed/unknown actions all become
observations the agent sees and recovers from (each consumes one step). Only
`max_steps` exhaustion or an explicit null answer yields a null value.

## Testing (TDD throughout)

- **config:** defaults; agent fields parse; `hint`/`max_steps` rejected for
  retrieval; `max_steps` bounds.
- **tools:** grep matching / case / invalid-regex / cap; read_page truncation /
  unknown; list_pages.
- **agent loop** (scripted `FakeLLMClient`): grep→answer happy path;
  grounding-fail→retry→answer; max_steps exhaustion→null; array variable; `hint`
  reaches the prompt; unknown-url observation.
- **CLI e2e:** mixed retrieval + agent config → merged, correctly-ordered output;
  agent vars demonstrably skip retrieval.

## Files

- **New:** `src/tarantula/agent_tools.py`, `src/tarantula/agent_extractor.py`.
- **Touched:** `src/tarantula/config.py`, `src/tarantula/cli.py`, and
  `src/tarantula/llm.py` (sequential-response support in `FakeLLMClient` for the
  loop tests, if needed).
- **Tests:** `tests/test_agent_tools.py`, `tests/test_agent_extractor.py`,
  config + CLI e2e additions.

## Backwards compatibility

`extraction_type` defaults to `retrieval`; configs without the field behave
exactly as today. No output-shape changes; no DB schema changes.
