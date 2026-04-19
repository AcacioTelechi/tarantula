from __future__ import annotations

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


def build_reduce_schema(variables: list[VariableSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for v in variables:
        if v.type == "array":
            props[v.name] = {
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
        else:
            props[v.name] = {
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
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [v.name for v in variables],
        "properties": props,
    }


SYSTEM = (
    "You are reconciling candidate extractions from multiple pages of one website "
    "into final values. For scalar variables, choose the best-supported candidate "
    "and prefer authoritative pages (e.g., /about, /company) over blog posts. "
    "For array variables, union and deduplicate candidate items and keep the "
    "source page for each item. Return null when no candidate supports a variable. "
    "Give a brief one-sentence reasoning per variable."
)


def _render_user_prompt(
    variables: list[VariableSpec],
    candidates_by_var: dict[str, list[Candidate]],
) -> str:
    lines = ["Variables:"]
    for v in variables:
        lines.append(f"- {v.name} ({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}")
    lines.append("")
    lines.append("Candidates per variable:")
    for v in variables:
        cs = candidates_by_var.get(v.name, [])
        lines.append(f"  {v.name}:")
        if not cs:
            lines.append("    (none)")
            continue
        for c in cs:
            title = f" [{c.page_title}]" if c.page_title else ""
            lines.append(
                f"    - value={json.dumps(c.value)} "
                f"source_url={c.source_url}{title} "
                f"quote={json.dumps(c.quote)}"
            )
    return "\n".join(lines)


def reduce_candidates(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    candidates_by_var: dict[str, list[Candidate]],
    model: str,
) -> dict[str, dict[str, Any]]:
    schema = build_reduce_schema(variables)
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(variables, candidates_by_var),
        json_schema=schema,
        model=model,
        temperature=0.0,
        schema_name="site_reduction",
    )
    raw = client.complete_json(req)
    results: dict[str, dict[str, Any]] = {}
    for v in variables:
        entry = raw.get(v.name, {}) or {}
        if v.type == "array":
            results[v.name] = {
                "value": entry.get("value"),
                "sources": entry.get("sources", []) or [],
                "reasoning": entry.get("reasoning", ""),
                "required_missing": bool(v.required and entry.get("value") in (None, [])),
            }
        else:
            results[v.name] = {
                "value": entry.get("value"),
                "source_url": entry.get("source_url"),
                "quote": entry.get("quote"),
                "reasoning": entry.get("reasoning", ""),
                "required_missing": bool(v.required and entry.get("value") is None),
            }
    return results
