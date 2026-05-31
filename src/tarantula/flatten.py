"""Flatten a run payload (the JSON emitted by `tarantula extract`) into wide
rows — one row per site, each variable expanded into
``<var>_value / _source_url / _quote / _reasoning / _required_missing`` columns
— writable as CSV or XLSX."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Sub-fields of each variable object, in the order they become columns.
_VAR_FIELDS = ("value", "source_url", "quote", "reasoning", "required_missing")

# Leading columns describing the run and the crawled site.
_RUN_FIELDS = ("run_id", "started_at", "finished_at")
_SITE_FIELDS = ("seed_url", "crawl_status", "pages_fetched")


def flatten_run(payload: dict) -> tuple[list[str], list[dict]]:
    """Return ``(fieldnames, rows)`` for a run payload.

    Variable columns are the union of variable names across all sites, in
    first-seen order. A site missing a variable (or a null sub-field) gets an
    empty cell for it.
    """
    sites = payload.get("sites", [])

    # Collect variable names in first-seen order across sites.
    var_names: list[str] = []
    seen: set[str] = set()
    for site in sites:
        for name in site.get("variables", {}):
            if name not in seen:
                seen.add(name)
                var_names.append(name)

    fields = list(_RUN_FIELDS) + list(_SITE_FIELDS)
    for name in var_names:
        fields += [f"{name}_{sub}" for sub in _VAR_FIELDS]

    rows: list[dict] = []
    for site in sites:
        row: dict = {f: payload.get(f) for f in _RUN_FIELDS}
        for f in _SITE_FIELDS:
            row[f] = site.get(f)
        variables = site.get("variables", {})
        for name in var_names:
            var = variables.get(name)
            for sub in _VAR_FIELDS:
                col = f"{name}_{sub}"
                value = var.get(sub) if var is not None else None
                row[col] = "" if value is None else value
        rows.append(row)

    return fields, rows


def _xlsx_cell(value: Any) -> Any:
    """openpyxl only accepts scalar cell types; stringify anything else (e.g. an
    array variable whose ``_value`` column holds a list)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_xlsx(fields: list[str], rows: list[dict], path: str | Path) -> None:
    """Write flattened rows to an .xlsx workbook: a header row plus one row per
    site, columns in ``fields`` order."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(list(fields))
    for row in rows:
        ws.append([_xlsx_cell(row.get(f, "")) for f in fields])
    wb.save(str(path))
