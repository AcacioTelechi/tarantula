from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from .chunker import chunk_text
from .config import load_urls_config, load_variables_config
from .crawler import crawl_site
from .extractor import MapInput, extract_from_chunk
from .llm import LLMClient, OpenAIClient
from .logging_setup import configure as configure_logging
from .playwright_fetcher import playwright_fetch
from .reducer import Candidate, reduce_candidates
from .store import Store

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class PipelineOptions:
    urls_path: Path
    variables_path: Path
    output_path: Optional[Path]
    db_path: Path
    data_dir: Path
    map_model: str = "gpt-4o-mini"
    reduce_model: str = "gpt-4o"
    cache_ttl_seconds: int = 86400
    max_tokens: int = 2_000_000
    no_cache: bool = False
    quiet: bool = False
    llm_client: Optional[LLMClient] = None


async def run_pipeline(opts: PipelineOptions) -> int:
    urls_cfg = load_urls_config(opts.urls_path)
    vars_cfg = load_variables_config(opts.variables_path)

    store = Store(opts.db_path, data_dir=opts.data_dir)
    store.init_schema()

    urls_yaml_inline = opts.urls_path.read_text()
    vars_yaml_inline = opts.variables_path.read_text()
    run_id = store.start_run(urls_yaml_inline, vars_yaml_inline)

    client = opts.llm_client or OpenAIClient()

    sites_out = []
    run_status_bits = 0  # bit 0=required_missing, bit 1=partial, bit 2=failed
    started_at = _iso_now()

    for site in urls_cfg.sites:
        ttl = 0 if opts.no_cache else opts.cache_ttl_seconds
        try:
            result = await crawl_site(
                store=store, run_id=run_id, site=site,
                cache_ttl_seconds=ttl,
                playwright_fetcher=playwright_fetch,
            )
        except Exception as e:
            log.exception("crawl of %s failed: %s", site.url, e)
            run_status_bits |= 0b100
            sites_out.append({
                "seed_url": site.url,
                "crawl_status": "failed",
                "pages_fetched": 0,
                "variables": {},
            })
            continue

        if result.status == "partial":
            run_status_bits |= 0b010
        elif result.status == "failed":
            run_status_bits |= 0b100

        candidates_by_var: dict[str, list[Candidate]] = {}
        page_rows = list(store.conn.execute(
            "SELECT p.id, p.url, p.title, p.cleaned_text FROM pages p "
            "JOIN crawl_pages cp ON cp.page_id = p.id "
            "WHERE cp.crawl_id = ? ORDER BY cp.depth, p.id",
            (result.crawl_id,),
        ).fetchall())

        for page_id, url, title, cleaned in page_rows:
            if not cleaned:
                continue
            for chunk in chunk_text(cleaned):
                chunk_id = store.save_chunk(
                    page_id=page_id, ordinal=chunk.ordinal,
                    text=chunk.text, token_count=chunk.token_count,
                )
                recs = extract_from_chunk(
                    client=client,
                    variables=vars_cfg.variables,
                    inp=MapInput(chunk_text=chunk.text, page_url=url, page_title=title),
                    model=opts.map_model,
                )
                for r in recs:
                    store.save_chunk_extraction(
                        run_id=run_id, chunk_id=chunk_id,
                        variable_name=r.variable_name,
                        found=r.found, value=r.value, quote=r.quote,
                    )
                    if r.found:
                        candidates_by_var.setdefault(r.variable_name, []).append(
                            Candidate(value=r.value, quote=r.quote,
                                      source_url=url, page_title=title)
                        )

        reduced = reduce_candidates(
            client=client, variables=vars_cfg.variables,
            candidates_by_var=candidates_by_var, model=opts.reduce_model,
        )
        for name, payload in reduced.items():
            store.save_extraction(
                run_id=run_id,
                crawl_id=result.crawl_id,
                variable_name=name,
                value=payload["value"],
                source_url=payload.get("source_url"),
                quote=payload.get("quote"),
                reasoning=payload.get("reasoning"),
            )
            if payload.get("required_missing"):
                run_status_bits |= 0b001

        sites_out.append({
            "seed_url": site.url,
            "crawl_status": result.status,
            "pages_fetched": result.pages_fetched,
            "variables": reduced,
        })

    store.finish_run(run_id, status="ok")

    finished_at = _iso_now()
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "sites": sites_out,
    }
    out_text = json.dumps(payload, indent=2, default=str)
    if opts.output_path:
        opts.output_path.write_text(out_text)
        if not opts.quiet:
            print(_summary(sites_out))
    else:
        print(out_text)

    if run_status_bits & 0b100:
        return 4
    if run_status_bits & 0b010:
        return 3
    if run_status_bits & 0b001:
        return 2
    return 0


def _summary(sites: list[dict]) -> str:
    n_sites = len(sites)
    extracted = sum(
        1 for s in sites for v in s["variables"].values() if v.get("value") is not None
    )
    total_vars = sum(len(s["variables"]) for s in sites)
    missing_required = sum(
        1 for s in sites for v in s["variables"].values() if v.get("required_missing")
    )
    return (
        f"{n_sites} sites, {extracted}/{total_vars} variables extracted, "
        f"{missing_required} required missing"
    )


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@app.command()
def extract(
    urls: Path = typer.Option(..., "--urls", exists=True, readable=True, help="Path to urls.yaml"),
    variables: Path = typer.Option(..., "--variables", exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, "--output"),
    db: Path = typer.Option(Path("tarantula.db"), "--db"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    map_model: str = typer.Option("gpt-4o-mini", "--map-model"),
    reduce_model: str = typer.Option("gpt-4o", "--reduce-model"),
    cache_ttl: str = typer.Option("24h", "--cache-ttl"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    max_tokens: int = typer.Option(2_000_000, "--max-tokens"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Crawl sites and extract typed variables via LLM."""
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "logs" / "run.jsonl"
    configure_logging(verbosity=verbose, log_path=log_path)

    opts = PipelineOptions(
        urls_path=urls,
        variables_path=variables,
        output_path=output,
        db_path=db,
        data_dir=data_dir,
        map_model=map_model,
        reduce_model=reduce_model,
        cache_ttl_seconds=_parse_duration(cache_ttl),
        max_tokens=max_tokens,
        no_cache=no_cache,
        quiet=quiet,
    )
    exit_code = asyncio.run(run_pipeline(opts))
    raise typer.Exit(code=exit_code)


def _parse_duration(s: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    s = s.strip().lower()
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    return int(s)


if __name__ == "__main__":
    app()
