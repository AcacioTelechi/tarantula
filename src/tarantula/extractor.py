from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VariableSpec
from .llm import LLMClient, LLMRequest

_JSON_TYPE = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def _value_schema(spec: VariableSpec) -> dict[str, Any]:
    if spec.type == "array":
        return {
            "type": ["array", "null"],
            "items": {"type": _JSON_TYPE[spec.items]},  # type: ignore[index]
        }
    return {"type": [_JSON_TYPE[spec.type], "null"]}


def build_map_schema(variables: list[VariableSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for v in variables:
        props[v.name] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["found", "value", "quote"],
            "properties": {
                "found": {"type": "boolean"},
                "value": _value_schema(v),
                "quote": {"type": ["string", "null"]},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [v.name for v in variables],
        "properties": props,
    }


SYSTEM = (
    "You extract typed variables from a single chunk of cleaned web content. "
    "Only answer found=true when the chunk CLEARLY supports it. "
    "The quote MUST be a verbatim substring of the chunk text (copy it exactly). "
    "If the chunk does not clearly support a variable, set found=false, value=null, quote=null. "
    "For array variables, include only items explicitly mentioned in the chunk."
)


def _render_user_prompt(
    variables: list[VariableSpec], chunk_text: str, page_url: str, page_title: str | None
) -> str:
    lines = [f"Page URL: {page_url}"]
    if page_title:
        lines.append(f"Page title: {page_title}")
    lines.append("")
    lines.append("Variables to extract:")
    for v in variables:
        lines.append(f"- {v.name} ({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}")
        for ex in v.examples:
            lines.append(f"  example input: {ex.input!r} -> output: {ex.output!r}")
    lines += ["", "Chunk text:", chunk_text]
    return "\n".join(lines)


@dataclass
class MapInput:
    chunk_text: str
    page_url: str
    page_title: str | None


@dataclass
class ExtractionRecord:
    variable_name: str
    found: bool
    value: Any
    quote: str | None


def extract_from_chunk(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    inp: MapInput,
    model: str,
) -> list[ExtractionRecord]:
    schema = build_map_schema(variables)
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(variables, inp.chunk_text, inp.page_url, inp.page_title),
        json_schema=schema,
        model=model,
        temperature=0.0,
        schema_name="per_chunk_extraction",
    )
    resp = client.complete_json(req)
    out: list[ExtractionRecord] = []
    for v in variables:
        entry = resp.get(v.name, {}) or {}
        found = bool(entry.get("found"))
        value = entry.get("value")
        quote = entry.get("quote")
        if found and (not quote or quote not in inp.chunk_text):
            found = False
            value = None
            quote = None
        out.append(ExtractionRecord(
            variable_name=v.name, found=found, value=value, quote=quote
        ))
    return out
