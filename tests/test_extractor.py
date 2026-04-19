from tarantula.config import VariableSpec
from tarantula.extractor import (
    build_map_schema, extract_from_chunk, MapInput,
)
from tarantula.llm import FakeLLMClient


def _vars():
    return [
        VariableSpec(name="company_name", type="string", description="Legal name.", required=True),
        VariableSpec(name="founded_year", type="integer", description="Year founded."),
        VariableSpec(name="products", type="array", items="string", description="Products."),
    ]


def test_build_map_schema_shape():
    schema = build_map_schema(_vars())
    assert schema["type"] == "object"
    props = schema["properties"]
    assert set(props) == {"company_name", "founded_year", "products"}
    assert props["company_name"]["properties"]["found"]["type"] == "boolean"
    assert props["founded_year"]["properties"]["value"]["type"] in ("integer", ["integer", "null"])
    assert props["products"]["properties"]["value"]["type"] in ("array", ["array", "null"])


def test_extract_returns_found_records_with_valid_quote():
    fake = FakeLLMClient(responses=[{
        "company_name": {"found": True, "value": "ACME Inc.",
                         "quote": "ACME Inc. is headquartered in Chicago."},
        "founded_year": {"found": False, "value": None, "quote": None},
        "products": {"found": True, "value": ["Widget"], "quote": "We make Widget."},
    }])
    chunk_text = "ACME Inc. is headquartered in Chicago. We make Widget."
    results = extract_from_chunk(
        client=fake,
        variables=_vars(),
        inp=MapInput(chunk_text=chunk_text, page_url="https://a.com", page_title="About"),
        model="fake",
    )
    by_name = {r.variable_name: r for r in results}
    assert by_name["company_name"].found is True
    assert by_name["company_name"].value == "ACME Inc."
    assert by_name["founded_year"].found is False
    assert by_name["products"].value == ["Widget"]


def test_extract_demotes_fabricated_quote():
    fake = FakeLLMClient(responses=[{
        "company_name": {"found": True, "value": "ACME",
                         "quote": "THIS QUOTE NEVER APPEARS"},
        "founded_year": {"found": False, "value": None, "quote": None},
        "products": {"found": False, "value": None, "quote": None},
    }])
    chunk_text = "ACME Inc. is in Chicago."
    results = extract_from_chunk(
        client=fake,
        variables=_vars(),
        inp=MapInput(chunk_text=chunk_text, page_url="https://a.com", page_title=None),
        model="fake",
    )
    by_name = {r.variable_name: r for r in results}
    assert by_name["company_name"].found is False
