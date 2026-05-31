import json
from pathlib import Path

import pytest
from tarantula.cli import run_pipeline, PipelineOptions
from tarantula.llm import FakeLLMClient


def _serve_fixture(httpserver, fixtures_dir: Path):
    for name, path in [("/", "index.html"), ("/about", "about.html"), ("/products", "products.html")]:
        body = (fixtures_dir / "sample_site" / path).read_text()
        httpserver.expect_request(name).respond_with_data(body, content_type="text/html")
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)


@pytest.mark.asyncio
async def test_pipeline_extracts_from_sample_site(httpserver, tmp_path, fixtures_dir):
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Company name., required: true}\n"
        "  - {name: founded_year, type: integer, description: Founded year.}\n"
        "  - {name: products, type: array, items: string, description: Products listed.}\n"
    )
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": None,
            "source_url": None,
            "quote": None,
            "reasoning": "No company name found in available chunks.",
        },
        "extract_founded_year": {
            "value": 1998,
            "source_url": httpserver.url_for("/about"),
            "quote": "founded in 1998",
            "reasoning": "Explicit on /about.",
        },
        "extract_products": {
            "value": ["Widget Pro", "Widget Lite"],
            "sources": [
                {"value_item": "Widget Pro",
                 "source_url": httpserver.url_for("/products"),
                 "quote": "Widget Pro"},
                {"value_item": "Widget Lite",
                 "source_url": httpserver.url_for("/products"),
                 "quote": "Widget Lite"},
            ],
            "reasoning": "Listed on /products.",
        },
    })

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        extract_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        retrieval="bm25",  # avoid embed calls in the test
    )
    exit_code = await run_pipeline(opts)

    out = json.loads((tmp_path / "out.json").read_text())
    site = out["sites"][0]
    by_var = site["variables"]
    assert by_var["company_name"]["value"] is None
    assert by_var["founded_year"]["value"] == 1998
    assert by_var["products"]["value"] == ["Widget Pro", "Widget Lite"]
    assert exit_code == 2  # required_missing for company_name


@pytest.mark.asyncio
async def test_extract_runs_one_llm_call_per_variable(httpserver, tmp_path, fixtures_dir):
    """Exactly one extraction call per variable, regardless of chunk count."""
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Company name., required: true}\n"
        "  - {name: founded_year, type: integer, description: Founded year.}\n"
    )
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": None, "source_url": None, "quote": None,
            "reasoning": "skip",
        },
        "extract_founded_year": {
            "value": None, "source_url": None, "quote": None,
            "reasoning": "skip",
        },
    })

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        extract_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        retrieval="bm25",
        top_k=5,
    )
    await run_pipeline(opts)

    schemas = [c.schema_name for c in fake.calls]
    assert schemas.count("extract_company_name") == 1
    assert schemas.count("extract_founded_year") == 1
    # Exactly two LLM calls total — no map step, no reduce step.
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_retrieval_bm25_does_not_embed(httpserver, tmp_path, fixtures_dir):
    """With --retrieval bm25, chunks should never be embedded (no API calls)."""
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Legal name of the company., required: true}\n"
    )
    fake = FakeLLMClient(
        responses_by_schema={
            "extract_company_name": {
                "value": None, "source_url": None, "quote": None,
                "reasoning": "nothing found",
            },
        },
    )

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        extract_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        quiet=True,
        retrieval="bm25",
        top_k=1,
    )
    await run_pipeline(opts)

    # bm25 mode must not hit the embedding API at all.
    assert fake.embed_calls == []


@pytest.mark.asyncio
async def test_pipeline_mixes_retrieval_and_agent_strategies(
    httpserver, tmp_path, fixtures_dir
):
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Company name.}\n"
        "  - {name: founded_year, type: integer, description: Founded year., "
        "extraction_type: agent, hint: 'a four digit year'}\n"
    )

    about_url = httpserver.url_for("/about")
    index_url = httpserver.url_for("/")
    fake = FakeLLMClient(responses_by_schema={
        # retrieval strategy for company_name. The quote MUST be a verbatim
        # substring of a chunk that bm25 retrieves for this variable — that is
        # the index page ("ACME Inc. is a privately held company."), not /about.
        "extract_company_name": {
            "value": "ACME Inc.", "source_url": index_url,
            "quote": "ACME Inc.", "reasoning": "from home page",
        },
        # agent strategy for founded_year — single-step answer (same response
        # is returned for every step under this schema key).
        "agent_founded_year": {
            "thought": "found it", "action": "answer",
            "pattern": None, "ignore_case": None, "url": None,
            "answer": {"value": 1998, "source_url": about_url,
                       "quote": "founded in 1998", "reasoning": "on /about"},
        },
    })

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        extract_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        quiet=True,
        retrieval="bm25",
    )
    exit_code = await run_pipeline(opts)

    out = json.loads((tmp_path / "out.json").read_text())
    by_var = out["sites"][0]["variables"]
    # Order preserved: company_name first, founded_year second.
    assert list(by_var.keys()) == ["company_name", "founded_year"]
    assert by_var["company_name"]["value"] == "ACME Inc."
    assert by_var["founded_year"]["value"] == 1998
    assert exit_code == 0

    # The agent variable used the agent schema, not the retrieval schema.
    schema_names = {c.schema_name for c in fake.calls}
    assert "agent_founded_year" in schema_names
    assert "extract_founded_year" not in schema_names
