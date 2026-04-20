import pytest
from tarantula.config import VariableSpec, VariableExample
from tarantula.retriever import Hit, build_query, rrf_fuse, retrieve_for_variable
from tarantula.llm import FakeLLMClient


def _spec(**kw) -> VariableSpec:
    base = dict(name="founded_year", type="integer",
                description="The year the organization was founded.")
    base.update(kw)
    return VariableSpec(**base)


def test_build_query_uses_name_description_and_example_inputs():
    spec = _spec(examples=[
        VariableExample(input="Founded in 1998", output=1998),
        VariableExample(input="Since 2015", output=2015),
    ])
    q = build_query(spec)
    assert "founded_year" in q
    assert "year the organization was founded" in q
    assert "Founded in 1998" in q
    assert "Since 2015" in q


def test_rrf_fuse_combines_ranks():
    # chunk 1: bm25 rank 1, no vec; chunk 2: bm25 rank 2, vec rank 1.
    bm25 = [1, 2, 3]
    vec = [2, 4]
    fused = rrf_fuse(bm25, vec, k=60)
    ids = [f[0] for f in fused]
    # Chunk 2 should outrank chunk 1 because it appears in both lists.
    assert ids.index(2) < ids.index(1)
    # All four distinct ids present.
    assert set(ids) == {1, 2, 3, 4}
    # Scores must be strictly decreasing (or equal with stable tie-break).
    scores = [f[1] for f in fused]
    assert scores == sorted(scores, reverse=True)
    # bm25_rank and vec_rank correctly populated.
    by_id = {f[0]: (f[2], f[3]) for f in fused}
    assert by_id[2] == (2, 1)
    assert by_id[3] == (3, None)
    assert by_id[4] == (None, 2)
    assert by_id[1] == (1, None)


def test_rrf_fuse_empty_inputs_returns_empty():
    assert rrf_fuse([], [], k=60) == []


def test_retrieve_for_variable_returns_hybrid_hits(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("urls", "vars")
    crawl_id = store.start_crawl(run_id, "https://ex.com")
    p = store.save_page(url="https://ex.com/a", raw_bytes=b"<a/>",
                        http_status=200, content_type="text/html",
                        fetcher="http", title="A")
    store.link_page(crawl_id, p, depth=0, parent_url=None)
    c1 = store.save_chunk(p, 0, "The company was founded in 1998.", 8)
    c2 = store.save_chunk(p, 1, "We sell widgets.", 4)
    # Script fake embeddings so c1 is vector-closer to the query than c2.
    fake = FakeLLMClient(embeddings_by_text={
        "The company was founded in 1998.": [1.0, 0.0],
        "We sell widgets.": [0.0, 1.0],
    })
    # Pre-populate chunk embeddings so retriever can score.
    store.save_chunk_embedding(c1, [1.0, 0.0], model="stub")
    store.save_chunk_embedding(c2, [0.0, 1.0], model="stub")
    spec = _spec()
    # Build a query string that, when embedded by fake, aligns with c1.
    fake.embeddings_by_text[build_query(spec)] = [1.0, 0.0]
    hits = retrieve_for_variable(
        store=store, crawl_id=crawl_id, variable=spec,
        embed_fn=lambda texts: fake.embed(texts, model="stub"),
        k=5, mode="hybrid",
    )
    assert [h.chunk_id for h in hits][0] == c1
    assert hits[0].url == "https://ex.com/a"
    assert hits[0].text.startswith("The company")


def test_retrieve_for_variable_mode_bm25_skips_embedding(tmp_path):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("urls", "vars")
    crawl_id = store.start_crawl(run_id, "https://ex.com")
    p = store.save_page(url="https://ex.com/a", raw_bytes=b"<a/>",
                        http_status=200, content_type="text/html",
                        fetcher="http", title="A")
    store.link_page(crawl_id, p, depth=0, parent_url=None)
    c1 = store.save_chunk(p, 0, "Founded in 1998 by two engineers.", 8)
    called = []
    hits = retrieve_for_variable(
        store=store, crawl_id=crawl_id, variable=_spec(),
        embed_fn=lambda texts: called.append(texts) or [[0.0]],
        k=5, mode="bm25",
    )
    assert [h.chunk_id for h in hits] == [c1]
    assert called == []  # mode=bm25 must not call the embedder
