from __future__ import annotations

import concurrent.futures
import re
from typing import Any

from .config import VariableSpec
from .llm import LLMClient, LLMRequest
from .retriever import Hit

_JSON_TYPE = {"string": "string", "integer": "integer",
              "number": "number", "boolean": "boolean"}


def _value_schema(spec: VariableSpec) -> dict[str, Any]:
    if spec.type == "array":
        return {
            "type": ["array", "null"],
            "items": {"type": _JSON_TYPE[spec.items]},  # type: ignore[index]
        }
    return {"type": [_JSON_TYPE[spec.type], "null"]}


def build_extract_schema(v: VariableSpec) -> dict[str, Any]:
    if v.type == "array":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "sources", "reasoning"],
            "properties": {
                "value": _value_schema(v),
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["value_item", "source_url", "quote"],
                        "properties": {
                            "value_item": {"type": _JSON_TYPE[v.items]},  # type: ignore[index]
                            "source_url": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                    },
                },
                "reasoning": {"type": "string"},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "source_url", "quote", "reasoning"],
        "properties": {
            "value": _value_schema(v),
            "source_url": {"type": ["string", "null"]},
            "quote": {"type": ["string", "null"]},
            "reasoning": {"type": "string"},
        },
    }


SYSTEM = (
    "You extract ONE typed variable from a set of retrieved sources. "
    "Only return a non-null value when at least one source clearly supports it. "
    "The quote MUST be a verbatim substring of one of the sources' content. "
    "The source_url MUST match the URL of the source that contains the quote. "
    "Prefer authoritative pages (e.g., /about, /sobre, /quem-somos, /institucional) "
    "over blog posts. For array variables, include only items explicitly mentioned "
    "in the sources and cite each item separately."
)


def _render_user_prompt(v: VariableSpec, hits: list[Hit]) -> str:
    lines = [
        f"Variable: {v.name} "
        f"({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}"
    ]
    if v.examples:
        lines.append("Examples:")
        for ex in v.examples:
            lines.append(f"  - input: {ex.input!r} -> output: {ex.output!r}")
    lines.append("")
    lines.append(f"Sources ({len(hits)}):")
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] URL: {h.url}")
        if h.title:
            lines.append(f"    Title: {h.title}")
        lines.append(f"    Content: {h.text}")
        lines.append("")
    return "\n".join(lines)


_WS = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _quote_in_any(chunk_texts: dict[str, list[str]], url: str, quote: str) -> bool:
    """True if `quote` appears verbatim — or whitespace-normalized — inside
    any chunk served from `url`. Verifies source grounding, not that `value`
    was actually derived from `quote` (the system prompt enforces that)."""
    nq = _normalize_ws(quote) if quote else ""
    for text in chunk_texts.get(url, []):
        if quote and quote in text:
            return True
        if nq and nq in _normalize_ws(text):
            return True
    return False


def _validate_and_finalize(
    v: VariableSpec, raw: dict[str, Any], hits: list[Hit]
) -> dict[str, Any]:
    chunk_texts: dict[str, list[str]] = {}
    for h in hits:
        chunk_texts.setdefault(h.url, []).append(h.text)

    if v.type == "array":
        sources = raw.get("sources", []) or []
        valid = [
            s for s in sources
            if s.get("quote") and s.get("source_url")
            and _quote_in_any(chunk_texts, s["source_url"], s["quote"])
        ]
        # Rebuild value from validated sources so value and sources stay in sync.
        value = [s["value_item"] for s in valid] if valid else None
        return {
            "value": value,
            "sources": valid,
            "reasoning": raw.get("reasoning", ""),
            "required_missing": bool(v.required and value in (None, [])),
        }

    value = raw.get("value")
    source_url = raw.get("source_url")
    quote = raw.get("quote")
    if value is not None:
        if not quote or not source_url or not _quote_in_any(
            chunk_texts, source_url, quote
        ):
            value = None
            source_url = None
            quote = None
    return {
        "value": value,
        "source_url": source_url,
        "quote": quote,
        "reasoning": raw.get("reasoning", ""),
        "required_missing": bool(v.required and value is None),
    }


def extract_variable(
    *,
    client: LLMClient,
    variable: VariableSpec,
    hits: list[Hit],
    model: str,
) -> dict[str, Any]:
    """Extract one variable from its retrieved sources.

    Output shape matches the legacy reducer:
    - scalars: {value, source_url, quote, reasoning, required_missing}
    - arrays:  {value, sources: [{value_item, source_url, quote}, ...],
                reasoning, required_missing}
    """
    if not hits:
        if variable.type == "array":
            return {
                "value": None, "sources": [],
                "reasoning": "no chunks retrieved",
                "required_missing": bool(variable.required),
            }
        return {
            "value": None, "source_url": None, "quote": None,
            "reasoning": "no chunks retrieved",
            "required_missing": bool(variable.required),
        }
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(variable, hits),
        json_schema=build_extract_schema(variable),
        model=model,
        temperature=0.0,
        schema_name=f"extract_{variable.name}",
    )
    raw = client.complete_json(req)
    return _validate_and_finalize(variable, raw, hits)


def extract_all(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    hits_by_var: dict[str, list[Hit]],
    model: str,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    """Run one LLM call per variable in parallel. Output preserves input order."""
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                extract_variable,
                client=client, variable=v,
                hits=hits_by_var.get(v.name, []),
                model=model,
            ): v
            for v in variables
        }
        for fut in concurrent.futures.as_completed(futures):
            v = futures[fut]
            results[v.name] = fut.result()
    # Preserve input variable order in the returned dict.
    return {v.name: results[v.name] for v in variables}
