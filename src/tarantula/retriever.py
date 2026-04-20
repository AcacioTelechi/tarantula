from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .config import VariableSpec
from .store import Store

Mode = Literal["hybrid", "bm25", "vec"]


@dataclass
class Hit:
    chunk_id: int
    page_id: int
    url: str
    title: str | None
    text: str
    score: float
    bm25_rank: int | None
    vec_rank: int | None


def build_query(spec: VariableSpec) -> str:
    """Compose a retrieval query from a variable spec.

    Uses name + description + the *input* side of each example. Example outputs
    are deliberately omitted because they're often numeric/categorical values
    that hurt lexical matching.
    """
    parts = [spec.name, spec.description]
    for ex in spec.examples:
        parts.append(ex.input)
    return " ".join(p for p in parts if p)


def rrf_fuse(
    bm25_ids: list[int], vec_ids: list[int], k: int = 60
) -> list[tuple[int, float, int | None, int | None]]:
    """Reciprocal Rank Fusion. Returns [(chunk_id, score, bm25_rank, vec_rank)].

    score(id) = sum over lists of 1/(k + rank_in_that_list).
    Ranks are 1-based. Missing from a list → no contribution from that list.
    """
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}
    vec_rank = {cid: i + 1 for i, cid in enumerate(vec_ids)}
    all_ids = set(bm25_rank) | set(vec_rank)
    scored: list[tuple[int, float, int | None, int | None]] = []
    for cid in all_ids:
        s = 0.0
        if cid in bm25_rank:
            s += 1.0 / (k + bm25_rank[cid])
        if cid in vec_rank:
            s += 1.0 / (k + vec_rank[cid])
        scored.append((cid, s, bm25_rank.get(cid), vec_rank.get(cid)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def retrieve_for_variable(
    *,
    store: Store,
    crawl_id: int,
    variable: VariableSpec,
    embed_fn: Callable[[list[str]], list[list[float]]],
    k: int = 20,
    mode: Mode = "hybrid",
    fts_candidates: int = 50,
    vec_candidates: int = 50,
) -> list[Hit]:
    """Retrieve top-k chunks for a single variable inside a single crawl.

    `embed_fn` takes a list of query strings and returns their vectors. We pass
    it in (rather than taking an LLMClient) so the retriever is easy to test
    and easy to repurpose for HyDE later.
    """
    query = build_query(variable)
    bm25_ids: list[int] = []
    vec_ids: list[int] = []

    if mode in ("hybrid", "bm25"):
        bm25_ids = [cid for cid, _score in
                    store.bm25_top_k(crawl_id, query, fts_candidates)]
    if mode in ("hybrid", "vec"):
        qvec = embed_fn([query])[0]
        vec_ids = [cid for cid, _score in
                   store.vector_top_k(crawl_id, qvec, vec_candidates)]

    # Always use RRF so scores are comparable across modes. Passing an empty
    # list for the absent leg degenerates to 1/(k + rank) from the present leg.
    if mode == "bm25":
        fused = rrf_fuse(bm25_ids, [])
    elif mode == "vec":
        fused = rrf_fuse([], vec_ids)
    else:
        fused = rrf_fuse(bm25_ids, vec_ids)

    fused = fused[:k]
    if not fused:
        return []

    # Hydrate chunks with text + url in one SQL call.
    ids = [f[0] for f in fused]
    rows = store.conn.execute(
        f"SELECT c.id, c.page_id, p.url, p.title, c.text "
        f"FROM chunks c JOIN pages p ON p.id = c.page_id "
        f"WHERE c.id IN ({','.join('?' * len(ids))})",
        ids,
    ).fetchall()
    by_id = {r[0]: r for r in rows}

    hits: list[Hit] = []
    for cid, score, br, vr in fused:
        if cid not in by_id:
            continue
        _cid, page_id, url, title, text = by_id[cid]
        hits.append(Hit(
            chunk_id=cid, page_id=page_id, url=url, title=title,
            text=text, score=score, bm25_rank=br, vec_rank=vr,
        ))
    return hits
