# extraction_type Agent Strategy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-variable `extraction_type: retrieval | agent` config; the `agent` strategy iteratively searches the crawled cleaned-text corpus with curated read-only tools (regex grep, page read) and returns the same typed, source-grounded result as the retrieval strategy.

**Architecture:** Two new pure modules — `agent_tools.py` (a DB-free `Corpus` + `grep`/`read_page`/`list_pages`) and `agent_extractor.py` (a JSON-action loop over the existing `complete_json`). The CLI splits variables by `extraction_type`, runs retrieval only for retrieval vars and the agent loop for agent vars, then merges both into the existing `reduced` dict so persistence, JSON output, and `tarantula flatten` are untouched.

**Tech Stack:** Python 3.11+, Pydantic v2, Typer, pytest, the project's existing `LLMClient`/`FakeLLMClient`.

---

## File Structure

- **Create** `src/tarantula/agent_tools.py` — `PageDoc`, `Corpus`, `grep`, `read_page`, `list_pages`. Pure, no DB, no LLM.
- **Create** `src/tarantula/agent_extractor.py` — `extract_variable_agent`, `extract_all_agent`, action-schema + system-prompt builders. Depends on `agent_tools`, `config`, `llm`, and reuses `_quote_in_any` from `contextual_extractor`.
- **Modify** `src/tarantula/config.py` — add `extraction_type`, `hint`, `max_steps` to `VariableSpec` + a validator.
- **Modify** `src/tarantula/cli.py` — split/merge in `run_pipeline`; `Reporter.extract_start` shows the split.
- **Create** `tests/test_agent_tools.py`, `tests/test_agent_extractor.py`.
- **Modify** `tests/test_config.py` (add cases), `tests/test_cli_e2e.py` (add a mixed-strategy e2e).

Reused, do not reimplement: `_quote_in_any(chunk_texts, url, quote)` from `contextual_extractor` (grounding check); `build_extract_schema(v)` from `contextual_extractor` (the answer sub-schema).

---

## Task 1: Config — `extraction_type`, `hint`, `max_steps`

**Files:**
- Modify: `src/tarantula/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError
from tarantula.config import VariableSpec


def test_variable_defaults_to_retrieval_strategy():
    v = VariableSpec(name="nome", type="string", description="x")
    assert v.extraction_type == "retrieval"
    assert v.hint is None
    assert v.max_steps is None


def test_agent_variable_accepts_hint_and_max_steps():
    v = VariableSpec(
        name="cnpj", type="string", description="tax id",
        extraction_type="agent", hint="14 digits", max_steps=6,
    )
    assert v.extraction_type == "agent"
    assert v.hint == "14 digits"
    assert v.max_steps == 6


def test_hint_rejected_for_retrieval_variable():
    with pytest.raises(ValidationError):
        VariableSpec(name="nome", type="string", description="x", hint="nope")


def test_max_steps_rejected_for_retrieval_variable():
    with pytest.raises(ValidationError):
        VariableSpec(name="nome", type="string", description="x", max_steps=5)


def test_max_steps_bounds_enforced():
    with pytest.raises(ValidationError):
        VariableSpec(name="cnpj", type="string", description="x",
                     extraction_type="agent", max_steps=0)
    with pytest.raises(ValidationError):
        VariableSpec(name="cnpj", type="string", description="x",
                     extraction_type="agent", max_steps=51)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -k "extraction or hint or max_steps or retrieval_strategy" -v`
Expected: FAIL — `VariableSpec` has no `extraction_type`/`hint`/`max_steps` (pydantic `extra="forbid"` raises).

- [ ] **Step 3: Add the fields + validator**

In `src/tarantula/config.py`, add `ExtractionType` next to the existing type aliases:

```python
ExtractionType = Literal["retrieval", "agent"]
```

Add three fields to `VariableSpec` (after `examples`):

```python
    extraction_type: ExtractionType = "retrieval"
    hint: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=50)
```

Add a validator to `VariableSpec` (after the existing `_check_items`):

```python
    @model_validator(mode="after")
    def _check_agent_fields(self) -> "VariableSpec":
        if self.extraction_type != "agent":
            if self.hint is not None:
                raise ValueError(
                    f"variable {self.name!r}: 'hint' only valid when "
                    "extraction_type is 'agent'"
                )
            if self.max_steps is not None:
                raise ValueError(
                    f"variable {self.name!r}: 'max_steps' only valid when "
                    "extraction_type is 'agent'"
                )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (all config tests, new and existing).

- [ ] **Step 5: Commit**

```bash
git add src/tarantula/config.py tests/test_config.py
git commit -m "feat(config): add per-variable extraction_type, hint, max_steps"
```

---

## Task 2: `agent_tools.py` — corpus + read-only tools

**Files:**
- Create: `src/tarantula/agent_tools.py`
- Test: `tests/test_agent_tools.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_tools.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_tools.py -v`
Expected: FAIL — `No module named 'tarantula.agent_tools'`.

- [ ] **Step 3: Implement the module**

Create `src/tarantula/agent_tools.py`:

```python
"""Curated, read-only tools an extraction agent uses to search the crawled
corpus. Pure functions over an in-memory Corpus — no DB, no network, no shell."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SNIPPET_MAX = 240


@dataclass(frozen=True)
class PageDoc:
    url: str
    title: str | None
    text: str


class Corpus:
    """A crawl's pages as (url, title, cleaned_text), searched in memory."""

    def __init__(self, docs: list[PageDoc]) -> None:
        self.docs = docs

    @classmethod
    def from_pages(cls, pages: list[tuple[str, str | None, str]]) -> "Corpus":
        return cls([PageDoc(url, title, text) for url, title, text in pages])


def grep(
    corpus: Corpus, pattern: str, ignore_case: bool = True, max_matches: int = 50
) -> dict[str, Any]:
    """Regex-search every page's cleaned text. Returns {matches, truncated} or
    {error} on an invalid pattern — never raises."""
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return {"error": f"invalid regex: {e}"}
    matches: list[dict[str, Any]] = []
    for doc in corpus.docs:
        for line_no, line in enumerate(doc.text.splitlines(), start=1):
            if rx.search(line):
                snippet = line.strip()
                if len(snippet) > _SNIPPET_MAX:
                    snippet = snippet[:_SNIPPET_MAX] + "…"
                matches.append({"url": doc.url, "line_no": line_no, "line": snippet})
                if len(matches) >= max_matches:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


def read_page(corpus: Corpus, url: str, max_chars: int = 6000) -> dict[str, Any]:
    """Return a page's cleaned text (truncation-noted), or {error} if unknown."""
    for doc in corpus.docs:
        if doc.url == url:
            if len(doc.text) > max_chars:
                return {"url": url, "text": doc.text[:max_chars], "truncated": True}
            return {"url": url, "text": doc.text, "truncated": False}
    return {"error": f"no page with url {url!r}"}


def list_pages(corpus: Corpus) -> dict[str, Any]:
    """List every page's url, title, and character count."""
    return {"pages": [
        {"url": d.url, "title": d.title, "chars": len(d.text)} for d in corpus.docs
    ]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_tools.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tarantula/agent_tools.py tests/test_agent_tools.py
git commit -m "feat(agent): add read-only corpus search tools"
```

---

## Task 3: `agent_extractor.py` — scalar loop (answer, null, grounding-retry, exhaustion)

**Files:**
- Create: `src/tarantula/agent_extractor.py`
- Test: `tests/test_agent_extractor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_extractor.py`:

```python
from tarantula.agent_tools import Corpus
from tarantula.agent_extractor import extract_variable_agent
from tarantula.config import VariableSpec
from tarantula.llm import FakeLLMClient


def _corpus():
    return Corpus.from_pages([
        ("https://a.example/", "Home", "Welcome.\nCNPJ: 12.345.678/0001-90"),
        ("https://a.example/about", "About", "Founded in 1998 in Rio."),
    ])


def _var(**kw):
    base = dict(name="cnpj", type="string", description="tax id",
                extraction_type="agent")
    base.update(kw)
    return VariableSpec(**base)


def _grep_then_answer(value, source_url, quote):
    return [
        {"thought": "search", "action": "grep",
         "pattern": "CNPJ", "ignore_case": True, "url": None, "answer": None},
        {"thought": "found", "action": "answer",
         "pattern": None, "ignore_case": None, "url": None,
         "answer": {"value": value, "source_url": source_url,
                    "quote": quote, "reasoning": "matched"}},
    ]


def test_agent_greps_then_answers_with_grounded_quote():
    fake = FakeLLMClient(responses=_grep_then_answer(
        "12.345.678/0001-90", "https://a.example/", "CNPJ: 12.345.678/0001-90"))
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=8)
    assert out["value"] == "12.345.678/0001-90"
    assert out["source_url"] == "https://a.example/"
    assert out["required_missing"] is False
    # First call grepped, second answered: two LLM calls.
    assert len(fake.calls) == 2


def test_agent_explicit_null_answer_returns_immediately():
    fake = FakeLLMClient(responses=[
        {"thought": "give up", "action": "answer",
         "pattern": None, "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None,
                    "quote": None, "reasoning": "not present"}},
    ])
    out = extract_variable_agent(
        client=fake, variable=_var(required=True), corpus=_corpus(),
        model="fake", max_steps=8)
    assert out["value"] is None
    assert out["required_missing"] is True
    assert len(fake.calls) == 1


def test_agent_retries_when_quote_not_grounded():
    # First answer cites a quote that is NOT in the corpus -> retry -> good answer.
    bad = {"thought": "guess", "action": "answer",
           "pattern": None, "ignore_case": None, "url": None,
           "answer": {"value": "00.000.000/0000-00",
                      "source_url": "https://a.example/",
                      "quote": "CNPJ: 00.000.000/0000-00", "reasoning": "hallucinated"}}
    good = {"thought": "fixed", "action": "answer",
            "pattern": None, "ignore_case": None, "url": None,
            "answer": {"value": "12.345.678/0001-90",
                       "source_url": "https://a.example/",
                       "quote": "CNPJ: 12.345.678/0001-90", "reasoning": "real"}}
    fake = FakeLLMClient(responses=[bad, good])
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=8)
    assert out["value"] == "12.345.678/0001-90"
    assert len(fake.calls) == 2


def test_agent_returns_null_when_step_budget_exhausted():
    # Always greps, never answers.
    grep_forever = {"thought": "search", "action": "grep",
                    "pattern": "x", "ignore_case": True, "url": None, "answer": None}
    fake = FakeLLMClient(responses=[grep_forever, grep_forever, grep_forever])
    out = extract_variable_agent(
        client=fake, variable=_var(required=True), corpus=_corpus(),
        model="fake", max_steps=3)
    assert out["value"] is None
    assert out["required_missing"] is True
    assert "exhausted" in out["reasoning"]
    assert len(fake.calls) == 3


def test_agent_hint_appears_in_system_prompt():
    fake = FakeLLMClient(responses=[
        {"thought": "x", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None, "quote": None,
                    "reasoning": "n/a"}},
    ])
    extract_variable_agent(
        client=fake, variable=_var(hint="14 digits XX.XXX.XXX/XXXX-XX"),
        corpus=_corpus(), model="fake", max_steps=4)
    assert "14 digits XX.XXX.XXX/XXXX-XX" in fake.calls[0].system


def test_agent_unknown_url_observation_does_not_crash():
    fake = FakeLLMClient(responses=[
        {"thought": "read", "action": "read_page", "pattern": None,
         "ignore_case": None, "url": "https://nope", "answer": None},
        {"thought": "done", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None, "quote": None,
                    "reasoning": "n/a"}},
    ])
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=5)
    assert out["value"] is None
    assert len(fake.calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_extractor.py -v`
Expected: FAIL — `No module named 'tarantula.agent_extractor'`.

- [ ] **Step 3: Implement the module (scalar + array; array exercised in Task 4)**

Create `src/tarantula/agent_extractor.py`:

```python
"""Agent extraction strategy: a JSON-action loop that searches the crawled
corpus with read-only tools (grep/read_page/list_pages) and returns the same
typed, source-grounded payload as the retrieval extractor."""
from __future__ import annotations

import concurrent.futures
import json
from typing import Any

from .agent_tools import Corpus, grep, list_pages, read_page
from .config import VariableSpec
from .contextual_extractor import _quote_in_any, build_extract_schema
from .llm import LLMClient, LLMRequest

DEFAULT_MAX_STEPS = 8

SYSTEM_HEADER = (
    "You extract ONE typed variable from a crawled website by searching its "
    "pages with tools. Each turn, return a JSON object with a 'thought', an "
    "'action', and that action's parameters; all unused parameters MUST be null.\n"
    "Actions:\n"
    "  grep:       set 'pattern' (a Python regex) and optionally 'ignore_case' "
    "(default true). Returns matching lines with their page url.\n"
    "  read_page:  set 'url'. Returns that page's cleaned text.\n"
    "  list_pages: no params. Returns every page's url, title, and size.\n"
    "  answer:     set 'answer'. Provide the typed value plus a 'quote' that is "
    "a VERBATIM substring of the cited page's text and the matching 'source_url'.\n"
    "Only answer a non-null value when a quote clearly supports it; if the data "
    "is absent, answer with value null. Keep searches focused."
)


def _system_prompt(v: VariableSpec) -> str:
    lines = [
        SYSTEM_HEADER,
        "",
        f"Variable: {v.name} "
        f"({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}",
    ]
    if v.hint:
        lines.append(f"Hint: {v.hint}")
    if v.examples:
        lines.append("Examples:")
        for ex in v.examples:
            lines.append(f"  - input: {ex.input!r} -> output: {ex.output!r}")
    return "\n".join(lines)


def _action_schema(v: VariableSpec) -> dict[str, Any]:
    answer_schema = dict(build_extract_schema(v))
    answer_schema["type"] = ["object", "null"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["thought", "action", "pattern", "ignore_case", "url", "answer"],
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string",
                       "enum": ["grep", "read_page", "list_pages", "answer"]},
            "pattern": {"type": ["string", "null"]},
            "ignore_case": {"type": ["boolean", "null"]},
            "url": {"type": ["string", "null"]},
            "answer": answer_schema,
        },
    }


def _grounding_map(corpus: Corpus) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for doc in corpus.docs:
        out.setdefault(doc.url, []).append(doc.text)
    return out


def _null_payload(v: VariableSpec, reason: str) -> dict[str, Any]:
    if v.type == "array":
        return {"value": None, "sources": [], "reasoning": reason,
                "required_missing": bool(v.required)}
    return {"value": None, "source_url": None, "quote": None,
            "reasoning": reason, "required_missing": bool(v.required)}


def _finalize_answer(
    v: VariableSpec, ans: dict[str, Any], grounding: dict[str, list[str]]
) -> dict[str, Any] | None:
    """Return a finalized payload, or None if the agent claimed a value that
    is not grounded (signals the loop to let the agent try again)."""
    if v.type == "array":
        sources = ans.get("sources") or []
        valid = [
            s for s in sources
            if s.get("quote") and s.get("source_url")
            and _quote_in_any(grounding, s["source_url"], s["quote"])
        ]
        claimed = ans.get("value") not in (None, [])
        if claimed and not valid:
            return None
        value = [s["value_item"] for s in valid] if valid else None
        return {"value": value, "sources": valid,
                "reasoning": ans.get("reasoning", ""),
                "required_missing": bool(v.required and value in (None, []))}

    value = ans.get("value")
    if value is not None:
        quote = ans.get("quote")
        url = ans.get("source_url")
        if not quote or not url or not _quote_in_any(grounding, url, quote):
            return None
    return {
        "value": value,
        "source_url": ans.get("source_url") if value is not None else None,
        "quote": ans.get("quote") if value is not None else None,
        "reasoning": ans.get("reasoning", ""),
        "required_missing": bool(v.required and value is None),
    }


def _run_tool(action: str, raw: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    if action == "grep":
        ic = raw.get("ignore_case")
        return grep(corpus, raw.get("pattern") or "",
                    ignore_case=True if ic is None else bool(ic))
    if action == "read_page":
        return read_page(corpus, raw.get("url") or "")
    if action == "list_pages":
        return list_pages(corpus)
    return {"error": f"unknown action {action!r}"}


def extract_variable_agent(
    *, client: LLMClient, variable: VariableSpec, corpus: Corpus,
    model: str, max_steps: int,
) -> dict[str, Any]:
    """Run the agent loop for one variable. Output shape matches the retrieval
    extractor (scalars and arrays)."""
    if not corpus.docs:
        return _null_payload(variable, "no pages crawled")

    system = _system_prompt(variable)
    schema = _action_schema(variable)
    grounding = _grounding_map(corpus)
    transcript: list[str] = [
        "Begin. Use the tools to locate the value, then answer."
    ]

    for _step in range(max_steps):
        req = LLMRequest(
            system=system,
            user="\n\n".join(transcript),
            json_schema=schema,
            model=model,
            temperature=0.0,
            schema_name=f"agent_{variable.name}",
        )
        raw = client.complete_json(req)
        action = raw.get("action")
        if action == "answer":
            payload = _finalize_answer(variable, raw.get("answer") or {}, grounding)
            if payload is not None:
                return payload
            transcript.append(
                "OBSERVATION: the cited quote was not found verbatim in that "
                "page. Re-check the source text and answer again, or keep searching."
            )
            continue
        obs = _run_tool(action, raw, corpus)
        transcript.append(f"OBSERVATION: {json.dumps(obs, ensure_ascii=False)}")

    return _null_payload(variable, f"step budget ({max_steps}) exhausted")


def extract_all_agent(
    *, client: LLMClient, variables: list[VariableSpec], corpus: Corpus,
    model: str, max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """Run the agent loop for each variable in parallel. Preserves input order."""
    if not variables:
        return {}
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                extract_variable_agent,
                client=client, variable=v, corpus=corpus, model=model,
                max_steps=v.max_steps or DEFAULT_MAX_STEPS,
            ): v
            for v in variables
        }
        for fut in concurrent.futures.as_completed(futures):
            v = futures[fut]
            results[v.name] = fut.result()
    return {v.name: results[v.name] for v in variables}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_extractor.py -v`
Expected: PASS (all scalar-loop tests).

- [ ] **Step 5: Commit**

```bash
git add src/tarantula/agent_extractor.py tests/test_agent_extractor.py
git commit -m "feat(agent): add JSON-action agent extraction loop"
```

---

## Task 4: `agent_extractor.py` — array variable support

**Files:**
- Modify: `tests/test_agent_extractor.py` (add cases; implementation already supports arrays)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_extractor.py`:

```python
def _array_corpus():
    return Corpus.from_pages([
        ("https://a.example/products", "Products",
         "We sell Widget Pro and Widget Lite to everyone."),
    ])


def test_agent_array_collects_grounded_sources():
    fake = FakeLLMClient(responses=[
        {"thought": "answer", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {
             "value": ["Widget Pro", "Widget Lite"],
             "sources": [
                 {"value_item": "Widget Pro",
                  "source_url": "https://a.example/products",
                  "quote": "Widget Pro"},
                 {"value_item": "Widget Lite",
                  "source_url": "https://a.example/products",
                  "quote": "Widget Lite"},
             ],
             "reasoning": "listed",
         }},
    ])
    v = VariableSpec(name="products", type="array", items="string",
                     description="products", extraction_type="agent")
    out = extract_variable_agent(
        client=fake, variable=v, corpus=_array_corpus(), model="fake", max_steps=5)
    assert out["value"] == ["Widget Pro", "Widget Lite"]
    assert len(out["sources"]) == 2


def test_agent_array_drops_ungrounded_items_and_retries():
    bad = {"thought": "guess", "action": "answer", "pattern": None,
           "ignore_case": None, "url": None,
           "answer": {"value": ["Phantom"],
                      "sources": [{"value_item": "Phantom",
                                   "source_url": "https://a.example/products",
                                   "quote": "Phantom Device"}],
                      "reasoning": "hallucinated"}}
    good = {"thought": "fix", "action": "answer", "pattern": None,
            "ignore_case": None, "url": None,
            "answer": {"value": ["Widget Pro"],
                       "sources": [{"value_item": "Widget Pro",
                                    "source_url": "https://a.example/products",
                                    "quote": "Widget Pro"}],
                       "reasoning": "real"}}
    fake = FakeLLMClient(responses=[bad, good])
    v = VariableSpec(name="products", type="array", items="string",
                     description="products", extraction_type="agent")
    out = extract_variable_agent(
        client=fake, variable=v, corpus=_array_corpus(), model="fake", max_steps=5)
    assert out["value"] == ["Widget Pro"]
    assert len(fake.calls) == 2
```

- [ ] **Step 2: Run tests to verify they pass (implementation already covers arrays)**

Run: `.venv/bin/python -m pytest tests/test_agent_extractor.py -k array -v`
Expected: PASS. If `test_agent_array_drops_ungrounded_items_and_retries` fails, confirm `_finalize_answer`'s array branch returns `None` when `claimed and not valid` — that is the retry trigger.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_extractor.py
git commit -m "test(agent): cover array extraction and ungrounded retry"
```

---

## Task 5: CLI integration — split by strategy, run both, merge

**Files:**
- Modify: `src/tarantula/cli.py` (`Reporter.extract_start`, and the per-site block in `run_pipeline`)
- Test: `tests/test_cli_e2e.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_e2e.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_mixes_retrieval_and_agent_strategies(
    httpserver, tmp_path, fixtures_dir
):
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Company name.}\n"
        "  - {name: founded_year, type: integer, description: Founded year., "
        "extraction_type: agent, hint: 'a four digit year'}\n"
    )

    about_url = httpserver.url_for("/about")
    fake = FakeLLMClient(responses_by_schema={
        # retrieval strategy for company_name. The quote MUST be a verbatim
        # substring of about.html's cleaned text: "ACME Inc. was founded in 1998."
        "extract_company_name": {
            "value": "ACME Inc.", "source_url": about_url,
            "quote": "ACME Inc.", "reasoning": "from about",
        },
        # agent strategy for founded_year — single-step answer (same response
        # is returned for every step under this schema key).
        "agent_founded_year": {
            "thought": "found it", "action": "answer",
            "pattern": None, "ignore_case": None, "url": None,
            "answer": {"value": 1998, "source_url": about_url,
                       "quote": "founded in 1998", "reasoning": "on /about"},
        },
    })

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        extract_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        quiet=True,
        retrieval="bm25",
    )
    exit_code = await run_pipeline(opts)

    out = json.loads((tmp_path / "out.json").read_text())
    by_var = out["sites"][0]["variables"]
    # Order preserved: company_name first, founded_year second.
    assert list(by_var.keys()) == ["company_name", "founded_year"]
    assert by_var["company_name"]["value"] == "ACME Inc."
    assert by_var["founded_year"]["value"] == 1998
    assert exit_code == 0

    # The agent variable used the agent schema, not the retrieval schema.
    schema_names = {c.schema_name for c in fake.calls}
    assert "agent_founded_year" in schema_names
    assert "extract_founded_year" not in schema_names
```

The `/about` fixture (`tests/fixtures/sample_site/about.html`) cleaned text is `ACME Inc. was founded in 1998.`, so both quotes above ground verbatim. Grounding is case-sensitive — do not change `ACME` to `Acme`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_e2e.py -k mixes -v`
Expected: FAIL — `founded_year` is currently extracted via the retrieval path (`extract_founded_year`), so `agent_founded_year` is never called and `extract_founded_year` is.

- [ ] **Step 3: Update `Reporter.extract_start`**

In `src/tarantula/cli.py`, replace the `extract_start` method:

```python
    def extract_start(self, n_retrieval: int, n_agent: int) -> None:
        self._emit(
            f"  [dim]extracting {n_retrieval} retrieval + {n_agent} agent "
            f"variable(s)...[/]"
        )
```

- [ ] **Step 4: Wire the split/merge in `run_pipeline`**

In `src/tarantula/cli.py`, add the import near the other local imports at the top of the file:

```python
from .agent_extractor import extract_all_agent
from .agent_tools import Corpus
```

Then replace the block that currently starts at the embed comment
`# --- Embed any chunks that don't yet have a vector (hybrid/vec only). ---`
and runs through the `extract_done` call. Replace it with:

```python
        # Split variables by extraction strategy. Agent variables never touch
        # retrieval or embeddings.
        retrieval_vars = [v for v in vars_cfg.variables
                          if v.extraction_type == "retrieval"]
        agent_vars = [v for v in vars_cfg.variables
                      if v.extraction_type == "agent"]

        # --- Embed any chunks that don't yet have a vector (hybrid/vec only). ---
        if retrieval_vars and opts.retrieval in ("hybrid", "vec"):
            chunk_ids = [r[0] for r in store.conn.execute(
                "SELECT c.id FROM chunks c "
                "JOIN crawl_pages cp ON cp.page_id = c.page_id "
                "WHERE cp.crawl_id = ? AND c.embedding IS NULL",
                (result.crawl_id,),
            )]
            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                rows = store.conn.execute(
                    f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                for i in range(0, len(rows), 64):
                    batch = rows[i:i + 64]
                    vecs = client.embed(
                        [t for _cid, t in batch], model=opts.embed_model,
                    )
                    for (cid, _text), vec in zip(batch, vecs):
                        store.save_chunk_embedding(cid, vec, model=opts.embed_model)

        # --- Retrieve top-k per retrieval variable. ---
        hits_by_var: dict[str, list[Hit]] = {}
        for v in retrieval_vars:
            hits = retrieve_for_variable(
                store=store, crawl_id=result.crawl_id, variable=v,
                embed_fn=lambda texts: client.embed(
                    texts, model=opts.embed_model
                ),
                k=opts.top_k, mode=opts.retrieval,
                fts_candidates=opts.retrieval_candidates,
                vec_candidates=opts.retrieval_candidates,
            )
            hits_by_var[v.name] = hits

        # --- Extract: retrieval strategy + agent strategy. ---
        reporter.extract_start(n_retrieval=len(retrieval_vars),
                               n_agent=len(agent_vars))
        reduced_retrieval = extract_all(
            client=client,
            variables=retrieval_vars,
            hits_by_var=hits_by_var,
            model=opts.extract_model,
            max_workers=opts.workers,
        )
        corpus = Corpus.from_pages([
            (url, title, cleaned)
            for _page_id, url, title, cleaned in pages_with_text
        ])
        reduced_agent = extract_all_agent(
            client=client,
            variables=agent_vars,
            corpus=corpus,
            model=opts.extract_model,
            max_workers=opts.workers,
        )
        merged = {**reduced_retrieval, **reduced_agent}
        reduced = {v.name: merged[v.name] for v in vars_cfg.variables}
        reporter.extract_done(n_vars=len(reduced))
```

Note: the existing code below this block already iterates `reduced.items()` to persist and build `sites_out` — leave that untouched. This replacement removes the old single `extract_all(... vars_cfg.variables ...)` call and the old retrieve loop over `vars_cfg.variables`; make sure no duplicate of either remains.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_e2e.py -k mixes -v`
Expected: PASS.

- [ ] **Step 6: Run the full e2e + extractor suites**

Run: `.venv/bin/python -m pytest tests/test_cli_e2e.py tests/test_agent_extractor.py tests/test_agent_tools.py -v`
Expected: PASS (including the pre-existing e2e tests — they have no agent vars, so `n_agent=0` and behavior is unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/tarantula/cli.py tests/test_cli_e2e.py
git commit -m "feat(cli): route variables to retrieval or agent extraction"
```

---

## Task 6: Full regression + lint + docs

**Files:**
- Modify: `README.md` (document `extraction_type`/`hint`/`max_steps`)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — all pre-existing tests plus the new ones.

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check src/tarantula tests`
Expected: `All checks passed!` (fix any unused-import/line-length issues in the new modules).

- [ ] **Step 3: Document the feature**

In `README.md`, in the variables-config section, add a short subsection with this example and explanation:

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

Explain: `extraction_type` defaults to `retrieval`; `agent` is best for
pattern-shaped data (CNPJ, e-mails, years). `hint` and `max_steps` are only
valid for agent variables. The agent searches cleaned page text only and every
returned value is quote-grounded, exactly like the retrieval strategy.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document extraction_type agent strategy"
```

---

## Self-Review

**Spec coverage:**
- Config `extraction_type`/`hint`/`max_steps` + validation → Task 1. ✓
- Curated read-only tools over cleaned text → Task 2. ✓
- JSON-action agent loop, grounding via `_quote_in_any`, grounding-fail retry, max_steps exhaustion → Task 3. ✓
- Same output shape (scalar + array) → Tasks 3–4. ✓
- CLI split/merge, agent vars skip retrieval+embeddings, order preserved → Task 5. ✓
- Reporter split line → Task 5. ✓
- Error handling (invalid regex, unknown url, unknown action as observations) → Task 2 (regex/url) + Task 3 (`_run_tool` unknown action, `test_agent_unknown_url_observation_does_not_crash`). ✓
- Tests for config/tools/loop/e2e → Tasks 1–5. ✓
- Docs → Task 6. ✓

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `Corpus.from_pages(list[tuple[url, title, text]])`, `grep(corpus, pattern, ignore_case, max_matches)`, `read_page(corpus, url, max_chars)`, `list_pages(corpus)`, `extract_variable_agent(*, client, variable, corpus, model, max_steps)`, `extract_all_agent(*, client, variables, corpus, model, max_workers)` — used consistently across Tasks 2–5. Schema name `agent_<name>` used in both the loop (Task 3) and the e2e assertion (Task 5). Reporter `extract_start(n_retrieval, n_agent)` defined and called once (Task 5).

**Known integration risk to verify during execution:** Task 5 replaces a specific existing block in `run_pipeline`. The executor must confirm the old `extract_all(... vars_cfg.variables ...)` call and the old `for v in vars_cfg.variables:` retrieve loop are fully removed (not duplicated). The full-suite run in Task 6 catches a mistake here.
