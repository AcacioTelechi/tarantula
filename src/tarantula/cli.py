from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import time

import typer
from dotenv import load_dotenv
from rich.console import Console

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


@app.callback()
def _main() -> None:
    """Tarantula: crawl sites and extract typed variables via LLM."""
    load_dotenv()


class Reporter:
    """Writes human-friendly progress to stderr. Safe to use in pipelines
    that write JSON to stdout — nothing here touches stdout."""

    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self._console = Console(stderr=True, highlight=False)
        self._run_started: float = 0.0
        self._site_started: float = 0.0

    def _emit(self, msg: str) -> None:
        if not self.quiet:
            self._console.print(msg)

    def run_start(self, n_sites: int, run_id: int) -> None:
        self._run_started = time.monotonic()
        self._emit(f"[bold cyan]tarantula[/] run #{run_id} — {n_sites} site(s)")

    def site_start(self, idx: int, total: int, url: str, cfg_depth: int, cfg_pages: int) -> None:
        self._site_started = time.monotonic()
        self._emit(
            f"\n[bold]({idx}/{total})[/] [cyan]{url}[/]  "
            f"[dim]max_depth={cfg_depth} max_pages={cfg_pages}[/]"
        )
        self._emit("  [dim]crawling...[/]")

    def page_fetched(self, pages: int, depth: int, url: str, source: str) -> None:
        tag = {"http": "[green]http[/]", "playwright": "[yellow]play[/]",
               "cache": "[blue]cache[/]"}.get(source, source)
        # Truncate very long URLs for readability.
        shown = url if len(url) <= 80 else url[:77] + "..."
        self._emit(f"    [dim]{pages:>3}.[/] {tag} d={depth} {shown}")

    def crawl_done(self, pages: int, status: str) -> None:
        color = {"ok": "green", "partial": "yellow", "failed": "red"}.get(status, "white")
        self._emit(f"  [dim]crawl[/] [{color}]{status}[/]: {pages} page(s)")

    def extract_start(self, n_pages: int) -> None:
        self._emit(f"  [dim]extracting (map) across {n_pages} page(s)...[/]")

    def extract_progress(self, chunk_i: int, chunk_n: int, url: str) -> None:
        shown = url if len(url) <= 60 else url[:57] + "..."
        # Carriage-return for in-place update; don't use emit which adds \n
        if not self.quiet:
            self._console.print(
                f"    [dim]chunk {chunk_i}/{chunk_n}[/] {shown}",
                end="\r", soft_wrap=True,
            )

    def extract_done(self, chunks: int, candidates: int) -> None:
        # Clear the in-place line then write the done line.
        if not self.quiet:
            self._console.print(" " * 100, end="\r")
        self._emit(f"  [dim]extract done: {chunks} chunk(s), {candidates} candidate(s)[/]")

    def reduce_start(self) -> None:
        self._emit("  [dim]reducing (per-site reconciliation)...[/]")

    def site_done(self, extracted: int, total: int, required_missing: int) -> None:
        elapsed = time.monotonic() - self._site_started
        miss = f" [red]{required_missing} required missing[/]" if required_missing else ""
        self._emit(
            f"  [bold green]done[/] {extracted}/{total} variables extracted "
            f"in {elapsed:.1f}s{miss}"
        )

    def run_done(self, exit_code: int, output_path: Path | None) -> None:
        elapsed = time.monotonic() - self._run_started
        code_color = "green" if exit_code == 0 else ("yellow" if exit_code == 2 else "red")
        where = f" -> {output_path}" if output_path else ""
        self._emit(
            f"\n[bold]finished[/] in {elapsed:.1f}s, "
            f"exit [{code_color}]{exit_code}[/]{where}"
        )


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
    map_workers: int = 8
    reduce_workers: int = 8
    llm_client: Optional[LLMClient] = None
    embed_model: str = "text-embedding-3-small"
    top_k: int = 20
    retrieval: str = "hybrid"  # "hybrid" | "bm25" | "vec" | "off"
    retrieval_candidates: int = 50  # BM25 and vector top-N before fusion


async def run_pipeline(opts: PipelineOptions) -> int:
    urls_cfg = load_urls_config(opts.urls_path)
    vars_cfg = load_variables_config(opts.variables_path)

    store = Store(opts.db_path, data_dir=opts.data_dir)
    store.init_schema()

    urls_yaml_inline = opts.urls_path.read_text()
    vars_yaml_inline = opts.variables_path.read_text()
    run_id = store.start_run(urls_yaml_inline, vars_yaml_inline)

    client = opts.llm_client or OpenAIClient()
    reporter = Reporter(quiet=opts.quiet)
    reporter.run_start(n_sites=len(urls_cfg.sites), run_id=run_id)

    sites_out = []
    run_status_bits = 0  # bit 0=required_missing, bit 1=partial, bit 2=failed
    started_at = _iso_now()

    for idx, site in enumerate(urls_cfg.sites, start=1):
        reporter.site_start(idx, len(urls_cfg.sites), site.url,
                            cfg_depth=site.max_depth, cfg_pages=site.max_pages)
        ttl = 0 if opts.no_cache else opts.cache_ttl_seconds
        try:
            result = await crawl_site(
                store=store, run_id=run_id, site=site,
                cache_ttl_seconds=ttl,
                playwright_fetcher=playwright_fetch,
                on_page=reporter.page_fetched,
            )
        except Exception as e:
            log.exception("crawl of %s failed: %s", site.url, e)
            reporter.crawl_done(pages=0, status="failed")
            run_status_bits |= 0b100
            sites_out.append({
                "seed_url": site.url,
                "crawl_status": "failed",
                "pages_fetched": 0,
                "variables": {},
            })
            continue

        reporter.crawl_done(pages=result.pages_fetched, status=result.status)
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

        pages_with_text = [r for r in page_rows if r[3]]
        reporter.extract_start(n_pages=len(pages_with_text))

        # Pre-compute chunks per page so we can report accurate progress.
        chunks_per_page = [
            (page_id, url, title, list(chunk_text(cleaned)))
            for page_id, url, title, cleaned in pages_with_text
        ]
        total_chunks = sum(len(cs) for _, _, _, cs in chunks_per_page)

        # Persist chunks serially (SQLite, single-threaded) and build the
        # task list for the parallel map step.
        tasks: list[tuple[int, str, str | None, str]] = []
        for page_id, url, title, chunks in chunks_per_page:
            for chunk in chunks:
                chunk_id = store.save_chunk(
                    page_id=page_id, ordinal=chunk.ordinal,
                    text=chunk.text, token_count=chunk.token_count,
                )
                tasks.append((chunk_id, url, title, chunk.text))

        # --- Embed any chunks that don't yet have a vector (retrieval step) ---
        if opts.retrieval != "off":
            ids = [t[0] for t in tasks]
            rows_to_embed: list[tuple[int, str]] = []
            if ids:
                placeholders = ",".join("?" * len(ids))
                existing_null = {
                    r[0] for r in store.conn.execute(
                        f"SELECT id FROM chunks "
                        f"WHERE id IN ({placeholders}) AND embedding IS NULL",
                        ids,
                    )
                }
                rows_to_embed = [
                    (cid, text) for (cid, _url, _title, text) in tasks
                    if cid in existing_null
                ]
            if rows_to_embed:
                # Batch to keep requests small; 64 chunks/req is a safe default.
                for i in range(0, len(rows_to_embed), 64):
                    batch = rows_to_embed[i:i + 64]
                    vecs = client.embed([t for _cid, t in batch], model=opts.embed_model)
                    for (cid, _text), vec in zip(batch, vecs):
                        store.save_chunk_embedding(cid, vec, model=opts.embed_model)

            # --- Retrieve top-k per variable; union the chunk IDs ---
            from .retriever import retrieve_for_variable
            keep_ids: set[int] = set()
            for v in vars_cfg.variables:
                hits = retrieve_for_variable(
                    store=store, crawl_id=result.crawl_id, variable=v,
                    embed_fn=lambda texts: client.embed(texts, model=opts.embed_model),
                    k=opts.top_k, mode=opts.retrieval,
                    fts_candidates=opts.retrieval_candidates,
                    vec_candidates=opts.retrieval_candidates,
                )
                keep_ids.update(h.chunk_id for h in hits)
            before = len(tasks)
            tasks = [t for t in tasks if t[0] in keep_ids]
            total_chunks = len(tasks)
            log.info("retrieval: %d/%d chunks kept for map", total_chunks, before)

        def _extract(task: tuple[int, str, str | None, str]):
            chunk_id, url, title, text = task
            recs = extract_from_chunk(
                client=client,
                variables=vars_cfg.variables,
                inp=MapInput(chunk_text=text, page_url=url, page_title=title),
                model=opts.map_model,
            )
            return chunk_id, url, title, recs

        done_i = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=opts.map_workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(_extract, t) for t in tasks]):
                chunk_id, url, title, recs = fut.result()
                done_i += 1
                reporter.extract_progress(done_i, total_chunks, url)
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

        n_candidates = sum(len(v) for v in candidates_by_var.values())
        reporter.extract_done(chunks=total_chunks, candidates=n_candidates)

        reporter.reduce_start()
        reduced = reduce_candidates(
            client=client, variables=vars_cfg.variables,
            candidates_by_var=candidates_by_var, model=opts.reduce_model,
            max_workers=opts.reduce_workers,
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

        extracted = sum(1 for v in reduced.values() if v.get("value") is not None)
        missing_required = sum(1 for v in reduced.values() if v.get("required_missing"))
        reporter.site_done(extracted, len(reduced), missing_required)

        sites_out.append({
            "seed_url": site.url,
            "crawl_status": result.status,
            "pages_fetched": result.pages_fetched,
            "variables": reduced,
        })

    run_status = "failed" if run_status_bits & 0b100 else "ok"
    store.finish_run(run_id, status=run_status)

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
    else:
        print(out_text)

    if run_status_bits & 0b100:
        exit_code = 4
    elif run_status_bits & 0b010:
        exit_code = 3
    elif run_status_bits & 0b001:
        exit_code = 2
    else:
        exit_code = 0

    reporter.run_done(exit_code, opts.output_path)
    return exit_code


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
    map_workers: int = typer.Option(8, "--map-workers", help="Concurrent LLM calls for the map step."),
    reduce_workers: int = typer.Option(8, "--reduce-workers", help="Concurrent LLM calls for the reduce step."),
    embed_model: str = typer.Option("text-embedding-3-small", "--embed-model"),
    top_k: int = typer.Option(20, "--top-k",
        help="Top-k chunks per variable after retrieval."),
    retrieval: str = typer.Option("hybrid", "--retrieval",
        help="hybrid | bm25 | vec | off (off = legacy full scan)."),
    retrieval_candidates: int = typer.Option(50, "--retrieval-candidates",
        help="BM25 and vector top-N before RRF fusion."),
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
        map_workers=map_workers,
        reduce_workers=reduce_workers,
        embed_model=embed_model,
        top_k=top_k,
        retrieval=retrieval,
        retrieval_candidates=retrieval_candidates,
    )
    exit_code = asyncio.run(run_pipeline(opts))
    raise typer.Exit(code=exit_code)


@app.command()
def retrieve(
    db: Path = typer.Option(Path("tarantula.db"), "--db", exists=True),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    variables: Path = typer.Option(..., "--variables", exists=True),
    seed_url: Optional[str] = typer.Option(None, "--seed-url",
        help="Filter to the latest crawl of this seed URL."),
    crawl_id: Optional[int] = typer.Option(None, "--crawl-id",
        help="Specific crawl to query. Overrides --seed-url."),
    variable: Optional[str] = typer.Option(None, "--variable",
        help="Retrieve for one variable name. Default: all."),
    k: int = typer.Option(10, "--k"),
    mode: str = typer.Option("hybrid", "--mode",
        help="hybrid | bm25 | vec"),
    retrieval_candidates: int = typer.Option(50, "--retrieval-candidates"),
    embed_model: str = typer.Option("text-embedding-3-small", "--embed-model"),
    show_text: bool = typer.Option(False, "--show-text",
        help="Print the full chunk text under each hit."),
    as_json: bool = typer.Option(False, "--json",
        help="Emit machine-readable JSON instead of a pretty table."),
) -> None:
    """Inspect retrieval for an existing crawl without running extraction."""
    from .retriever import retrieve_for_variable

    vars_cfg = load_variables_config(variables)
    store = Store(db, data_dir=data_dir)
    store.init_schema()

    if crawl_id is None:
        if not seed_url:
            raise typer.BadParameter("Pass --seed-url or --crawl-id.")
        row = store.conn.execute(
            "SELECT id FROM crawls WHERE seed_url=? "
            "ORDER BY started_at DESC LIMIT 1", (seed_url,),
        ).fetchone()
        if not row:
            raise typer.BadParameter(f"No crawl found for {seed_url!r}.")
        crawl_id = row[0]

    chosen = [v for v in vars_cfg.variables
              if variable is None or v.name == variable]
    if not chosen:
        raise typer.BadParameter(f"No variable named {variable!r}.")

    client = OpenAIClient()
    embed_fn = lambda texts: client.embed(texts, model=embed_model)

    console = Console(stderr=False, highlight=False)
    out: list[dict] = []
    for v in chosen:
        hits = retrieve_for_variable(
            store=store, crawl_id=crawl_id, variable=v,
            embed_fn=embed_fn, k=k, mode=mode,
            fts_candidates=retrieval_candidates,
            vec_candidates=retrieval_candidates,
        )
        out.append({
            "variable": v.name,
            "hits": [{
                "chunk_id": h.chunk_id, "score": h.score,
                "bm25_rank": h.bm25_rank, "vec_rank": h.vec_rank,
                "url": h.url, "title": h.title,
                "text": h.text if show_text else h.text[:160],
            } for h in hits],
        })

    if as_json:
        if len(out) == 1 and variable is not None:
            print(json.dumps(out[0], indent=2))
        else:
            print(json.dumps(out, indent=2))
        return

    for entry in out:
        console.print(f"\n[bold cyan]{entry['variable']}[/]")
        for i, h in enumerate(entry["hits"], start=1):
            ranks = (
                f"bm25={h['bm25_rank'] if h['bm25_rank'] is not None else '-'}"
                f" vec={h['vec_rank'] if h['vec_rank'] is not None else '-'}"
            )
            preview = h["text"][:160].replace("\n", " ")
            console.print(
                f"  [bold]{i:>2}.[/] score={h['score']:.4f} {ranks}  "
                f"[blue]{h['url']}[/]"
            )
            console.print(f"      [dim]{preview}[/]")
            if show_text and len(h["text"]) > 160:
                console.print(f"      [dim]{h['text'][160:]}[/]")


def _parse_duration(s: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    s = s.strip().lower()
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    return int(s)


if __name__ == "__main__":
    app()
