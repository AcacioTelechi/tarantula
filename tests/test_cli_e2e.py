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
    reduce_resp = {
        "company_name": {
            "value": "ACME Inc.",
            "source_url": httpserver.url_for("/about"),
            "quote": "ACME Inc. was founded in 1998.",
            "reasoning": "About page states legal name.",
        },
        "founded_year": {
            "value": 1998,
            "source_url": httpserver.url_for("/about"),
            "quote": "ACME Inc. was founded in 1998.",
            "reasoning": "Explicit on /about.",
        },
        "products": {
            "value": ["Widget Pro", "Widget Lite"],
            "sources": [
                {"value_item": "Widget Pro", "source_url": httpserver.url_for("/products"), "quote": "Widget Pro"},
                {"value_item": "Widget Lite", "source_url": httpserver.url_for("/products"), "quote": "Widget Lite"},
            ],
            "reasoning": "Listed on /products.",
        },
    }
    fake = FakeLLMClient(responses=[map_resp_index, map_resp_about, map_resp_products, reduce_resp])

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
