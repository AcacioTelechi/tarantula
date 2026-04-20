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
        "  - {name: company_name, type: string, description: Legal name., required: true}\n"
        "  - {name: founded_year, type: integer, description: Year.}\n"
        "  - {name: products, type: array, items: string, description: Products.}\n"
    )

    map_resp_index = {
        "company_name": {"found": True, "value": "ACME Inc.", "quote": "ACME Inc. is a privately held company."},
        "founded_year": {"found": False, "value": None, "quote": None},
        "products": {"found": False, "value": None, "quote": None},
    }
    map_resp_about = {
        "company_name": {"found": True, "value": "ACME Inc.", "quote": "ACME Inc. was founded in 1998."},
        "founded_year": {"found": True, "value": 1998, "quote": "ACME Inc. was founded in 1998."},
        "products": {"found": False, "value": None, "quote": None},
    }
    map_resp_products = {
        "company_name": {"found": False, "value": None, "quote": None},
        "founded_year": {"found": False, "value": None, "quote": None},
        "products": {"found": True, "value": ["Widget Pro", "Widget Lite"], "quote": "Widget Pro"},
    }
    reduce_by_schema = {
        "site_reduction_company_name": {
            "company_name": {
                "value": "ACME Inc.",
                "source_url": httpserver.url_for("/about"),
                "quote": "ACME Inc. was founded in 1998.",
                "reasoning": "About page states legal name.",
            },
        },
        "site_reduction_founded_year": {
            "founded_year": {
                "value": 1998,
                "source_url": httpserver.url_for("/about"),
                "quote": "ACME Inc. was founded in 1998.",
                "reasoning": "Explicit on /about.",
            },
        },
        "site_reduction_products": {
            "products": {
                "value": ["Widget Pro", "Widget Lite"],
                "sources": [
                    {"value_item": "Widget Pro", "source_url": httpserver.url_for("/products"), "quote": "Widget Pro"},
                    {"value_item": "Widget Lite", "source_url": httpserver.url_for("/products"), "quote": "Widget Lite"},
                ],
                "reasoning": "Listed on /products.",
            },
        },
    }
    fake = FakeLLMClient(
        responses=[map_resp_index, map_resp_about, map_resp_products],
        responses_by_schema=reduce_by_schema,
    )

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        map_model="fake",
        reduce_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        retrieval="off",
    )
    exit_code = await run_pipeline(opts)

    out = json.loads((tmp_path / "out.json").read_text())
    assert len(out["sites"]) == 1
    site = out["sites"][0]
    by_var = site["variables"]
    assert by_var["company_name"]["value"] == "ACME Inc."
    assert by_var["founded_year"]["value"] == 1998
    assert by_var["products"]["value"] == ["Widget Pro", "Widget Lite"]
    assert exit_code == 0


@pytest.mark.asyncio
async def test_retrieval_narrows_map_calls(httpserver, tmp_path, fixtures_dir):
    """With hybrid retrieval, map should only run on retrieved chunks."""
    _serve_fixture(httpserver, fixtures_dir)

    urls_yaml = tmp_path / "urls.yaml"
    urls_yaml.write_text(
        "defaults:\n  max_depth: 2\n  rate_limit_rps: 50\n"
        "sites:\n"
        f"  - url: {httpserver.url_for('/')}\n"
    )
    vars_yaml = tmp_path / "vars.yaml"
    # Single variable so top_k=1 yields exactly 1 retrieved chunk,
    # which drives exactly 1 map call.
    vars_yaml.write_text(
        "variables:\n"
        "  - {name: company_name, type: string, description: Legal name of the company., required: true}\n"
    )
    fake = FakeLLMClient(
        responses_by_schema={
            "per_chunk_extraction": {
                "company_name": {"found": False, "value": None, "quote": None},
            },
            "site_reduction_company_name": {
                "company_name": {
                    "value": None, "source_url": None, "quote": None,
                    "reasoning": "nothing found", "required_missing": True,
                },
            },
        },
    )

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        map_model="fake",
        reduce_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        quiet=True,
        top_k=1,
        retrieval="hybrid",
    )
    await run_pipeline(opts)

    n_map = sum(1 for c in fake.calls if c.schema_name == "per_chunk_extraction")
    assert n_map == 1, f"expected exactly 1 map call, got {n_map}"


@pytest.mark.asyncio
async def test_retrieval_off_preserves_full_scan(httpserver, tmp_path, fixtures_dir):
    """With --retrieval off, every chunk should hit the map step (legacy behavior)."""
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
        "  - {name: company_name, type: string, description: Legal name., required: true}\n"
    )
    fake = FakeLLMClient(
        responses_by_schema={
            "per_chunk_extraction": {
                "company_name": {"found": False, "value": None, "quote": None},
            },
            "site_reduction_company_name": {
                "company_name": {
                    "value": None, "source_url": None, "quote": None,
                    "reasoning": "nothing found", "required_missing": True,
                },
            },
        },
    )

    opts = PipelineOptions(
        urls_path=urls_yaml,
        variables_path=vars_yaml,
        output_path=tmp_path / "out.json",
        db_path=tmp_path / "t.db",
        data_dir=tmp_path / "data",
        map_model="fake",
        reduce_model="fake",
        cache_ttl_seconds=3600,
        max_tokens=10_000_000,
        no_cache=False,
        llm_client=fake,
        quiet=True,
        retrieval="off",
    )
    await run_pipeline(opts)

    n_map = sum(1 for c in fake.calls if c.schema_name == "per_chunk_extraction")
    # 3 pages in the fixture → 3 chunks (pages are small) → 3 map calls.
    assert n_map >= 3, f"expected at least 3 map calls (full scan), got {n_map}"
