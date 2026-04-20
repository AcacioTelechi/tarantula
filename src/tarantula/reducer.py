from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from typing import Any

from .config import VariableSpec
from .llm import LLMClient, LLMRequest


@dataclass
class Candidate:
    value: Any
    quote: str | None
    source_url: str
    page_title: str | None


_JSON_TYPE = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def _value_schema(spec: VariableSpec) -> dict[str, Any]:
    if spec.type == "array":
        return {"type": ["array", "null"], "items": {"type": _JSON_TYPE[spec.items]}}  # type: ignore[index]
    return {"type": [_JSON_TYPE[spec.type], "null"]}


def _variable_entry_schema(v: VariableSpec) -> dict[str, Any]:
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


def build_reduce_schema(variables: list[VariableSpec]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [v.name for v in variables],
        "properties": {v.name: _variable_entry_schema(v) for v in variables},
    }


SYSTEM = (
    "You are reconciling candidate extractions from multiple pages of one website "
    "into the final value for ONE variable. Choose the best-supported candidate and "
    "prefer authoritative pages (e.g., /about, /sobre, /quem-somos, /institucional) "
    "over blog posts. For array variables, union and deduplicate candidate items and "
    "keep the source page for each item. Return null when no candidate supports the "
    "variable. Give a brief one-sentence reasoning."
)


def _render_user_prompt(v: VariableSpec, candidates: list[Candidate]) -> str:
    header = (
        f"Variable:\n- {v.name} "
        f"({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}"
    )
    if not candidates:
        return header + "\n\nCandidates:\n  (none)"
    lines = [header, "", "Candidates:"]
    for c in candidates:
        title = f" [{c.page_title}]" if c.page_title else ""
        lines.append(
            f"  - value={json.dumps(c.value)} "
            f"source_url={c.source_url}{title} "
            f"quote={json.dumps(c.quote)}"
        )
    return "\n".join(lines)


def _finalize_entry(v: VariableSpec, entry: dict[str, Any]) -> dict[str, Any]:
    if v.type == "array":
        return {
            "value": entry.get("value"),
            "sources": entry.get("sources", []) or [],
            "reasoning": entry.get("reasoning", ""),
            "required_missing": bool(v.required and entry.get("value") in (None, [])),
        }
    return {
        "value": entry.get("value"),
        "source_url": entry.get("source_url"),
        "quote": entry.get("quote"),
        "reasoning": entry.get("reasoning", ""),
        "required_missing": bool(v.required and entry.get("value") is None),
    }


def _reduce_one(
    *,
    client: LLMClient,
    spec: VariableSpec,
    candidates: list[Candidate],
    model: str,
) -> dict[str, Any]:
    schema = build_reduce_schema([spec])
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(spec, candidates),
        json_schema=schema,
        model=model,
        temperature=0.0,
        schema_name=f"site_reduction_{spec.name}",
    )
    raw = client.complete_json(req)
    entry = raw.get(spec.name, {}) or {}
    return _finalize_entry(spec, entry)


def reduce_candidates(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    candidates_by_var: dict[str, list[Candidate]],
    model: str,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _reduce_one,
                client=client,
                spec=v,
                candidates=candidates_by_var.get(v.name, []),
                model=model,
            ): v
            for v in variables
        }
        for fut in concurrent.futures.as_completed(futures):
            v = futures[fut]
            results[v.name] = fut.result()
    return {v.name: results[v.name] for v in variables}
