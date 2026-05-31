"""Flatten a run payload (the JSON emitted by `tarantula extract`) into wide
CSV rows: one row per site, each variable expanded into
``<var>_value / _source_url / _quote / _reasoning / _required_missing`` columns."""
from __future__ import annotations

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
