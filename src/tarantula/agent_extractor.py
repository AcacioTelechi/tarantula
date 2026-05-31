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
