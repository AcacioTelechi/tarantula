from tarantula.config import VariableSpec
from tarantula.contextual_extractor import (
    extract_variable, extract_all, build_extract_model,
)
from tarantula.llm import FakeLLMClient, _response_format
from tarantula.retriever import Hit


def _scalar_spec(**kw) -> VariableSpec:
    base = dict(name="company_name", type="string",
                description="Legal name of the company.")
    base.update(kw)
    return VariableSpec(**base)


def _array_spec(**kw) -> VariableSpec:
    base = dict(name="products", type="array", items="string",
                description="Products offered.")
    base.update(kw)
    return VariableSpec(**base)


def _hit(chunk_id: int, url: str, text: str, title: str | None = None) -> Hit:
    return Hit(
        chunk_id=chunk_id, page_id=chunk_id, url=url, title=title,
        text=text, score=1.0, bm25_rank=1, vec_rank=None,
    )


def test_scalar_model_shape_and_validation():
    Model = build_extract_model(_scalar_spec())
    assert set(Model.model_fields) == {"value", "source_url", "quote", "reasoning"}
    inst = Model.model_validate(
        {"value": "ACME", "source_url": "u", "quote": "q", "reasoning": "r"})
    assert inst.value == "ACME"
    # value is nullable (variable not found).
    null = Model.model_validate(
        {"value": None, "source_url": None, "quote": None, "reasoning": "r"})
    assert null.value is None


def test_array_model_shape_and_validation():
    Model = build_extract_model(_array_spec())
    assert set(Model.model_fields) == {"value", "sources", "reasoning"}
    inst = Model.model_validate({
        "value": ["a", "b"],
        "sources": [{"value_item": "a", "source_url": "u", "quote": "q"}],
        "reasoning": "r",
    })
    assert inst.value == ["a", "b"]
    assert inst.sources[0].value_item == "a"


def test_scalar_model_builds_strict_response_format():
    rf = _response_format(build_extract_model(_scalar_spec()))
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"value", "source_url", "quote", "reasoning"}


def test_array_model_builds_strict_response_format():
    # Guards the SDK converter against a nested list[submodel] (array variables).
    rf = _response_format(build_extract_model(_array_spec()))
    assert rf["json_schema"]["strict"] is True


def test_extract_scalar_returns_value_when_quote_valid():
    hits = [_hit(1, "https://ex.com/about", "ACME Inc. is a private company.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.",
            "source_url": "https://ex.com/about",
            "quote": "ACME Inc. is a private company.",
            "reasoning": "from about page",
        }
    })
    result = extract_variable(client=fake, variable=_scalar_spec(),
                              hits=hits, model="fake")
    assert result["value"] == "ACME Inc."
    assert result["source_url"] == "https://ex.com/about"
    assert result["quote"] == "ACME Inc. is a private company."
    assert result["required_missing"] is False


def test_extract_scalar_nulls_value_when_quote_fabricated():
    hits = [_hit(1, "https://ex.com/about", "Real content here only.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.",
            "source_url": "https://ex.com/about",
            "quote": "Fabricated quote not in content.",
            "reasoning": "hallucinated",
        }
    })
    result = extract_variable(client=fake, variable=_scalar_spec(),
                              hits=hits, model="fake")
    assert result["value"] is None
    assert result["quote"] is None
    assert result["source_url"] is None


def test_extract_scalar_nulls_value_when_no_citation_provided():
    hits = [_hit(1, "https://ex.com/about", "Some text.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.",
            "source_url": None,
            "quote": None,
            "reasoning": "claimed without citation",
        }
    })
    result = extract_variable(client=fake, variable=_scalar_spec(),
                              hits=hits, model="fake")
    assert result["value"] is None


def test_extract_skips_llm_call_when_no_hits():
    fake = FakeLLMClient()
    result = extract_variable(client=fake, variable=_scalar_spec(required=True),
                              hits=[], model="fake")
    assert result["value"] is None
    assert result["required_missing"] is True
    assert fake.calls == []  # no LLM call


def test_extract_array_drops_sources_with_bad_quotes():
    hits = [_hit(1, "https://ex.com/products", "We sell Widget Pro and Widget Lite.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_products": {
            "value": ["Widget Pro", "Widget Lite", "Ghost Item"],
            "sources": [
                {"value_item": "Widget Pro",
                 "source_url": "https://ex.com/products",
                 "quote": "Widget Pro"},
                {"value_item": "Widget Lite",
                 "source_url": "https://ex.com/products",
                 "quote": "Widget Lite"},
                {"value_item": "Ghost Item",
                 "source_url": "https://ex.com/products",
                 "quote": "This quote is fake"},
            ],
            "reasoning": "products page",
        }
    })
    result = extract_variable(client=fake, variable=_array_spec(),
                              hits=hits, model="fake")
    assert len(result["sources"]) == 2
    assert {s["value_item"] for s in result["sources"]} == {"Widget Pro", "Widget Lite"}
    # Value list must stay in sync with filtered sources — no uncited items.
    assert result["value"] == ["Widget Pro", "Widget Lite"]


def test_extract_array_nulls_value_when_all_sources_invalid():
    hits = [_hit(1, "https://ex.com/products", "Real text.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_products": {
            "value": ["Ghost"],
            "sources": [
                {"value_item": "Ghost",
                 "source_url": "https://ex.com/products",
                 "quote": "fabricated quote"},
            ],
            "reasoning": "all bad",
        }
    })
    result = extract_variable(client=fake, variable=_array_spec(),
                              hits=hits, model="fake")
    assert result["value"] is None
    assert result["sources"] == []


def test_extract_required_missing_flag_on_null():
    hits = [_hit(1, "https://ex.com/a", "nothing relevant")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": None, "source_url": None, "quote": None,
            "reasoning": "not found",
        }
    })
    result = extract_variable(client=fake,
                              variable=_scalar_spec(required=True),
                              hits=hits, model="fake")
    assert result["required_missing"] is True


def test_extract_all_calls_llm_once_per_variable():
    hits_by_var = {
        "company_name": [_hit(1, "https://ex.com/a", "ACME Inc. founded.")],
        "products": [_hit(2, "https://ex.com/b", "We sell Widget Pro.")],
    }
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.", "source_url": "https://ex.com/a",
            "quote": "ACME Inc. founded.", "reasoning": "x",
        },
        "extract_products": {
            "value": ["Widget Pro"],
            "sources": [{"value_item": "Widget Pro",
                         "source_url": "https://ex.com/b",
                         "quote": "Widget Pro"}],
            "reasoning": "x",
        },
    })
    variables = [_scalar_spec(), _array_spec()]
    results = extract_all(client=fake, variables=variables,
                          hits_by_var=hits_by_var, model="fake")
    assert results["company_name"]["value"] == "ACME Inc."
    assert results["products"]["value"] == ["Widget Pro"]
    assert len(fake.calls) == 2
    # Result ordering matches input variable order (stable for JSON output).
    assert list(results.keys()) == ["company_name", "products"]


def test_extract_scalar_accepts_whitespace_normalized_quote():
    """Chunker joins paragraphs with '\\n\\n'; LLM often returns the quote
    with a single space. Substring match must still succeed."""
    hits = [_hit(1, "https://ex.com/a", "ACME Inc. was founded\n\nin 1998.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.",
            "source_url": "https://ex.com/a",
            "quote": "ACME Inc. was founded in 1998.",
            "reasoning": "from about page",
        }
    })
    result = extract_variable(client=fake, variable=_scalar_spec(),
                              hits=hits, model="fake")
    assert result["value"] == "ACME Inc."
    assert result["quote"] == "ACME Inc. was founded in 1998."


def test_quote_fabrication_still_blocked_with_whitespace_normalization():
    """A genuinely fabricated quote must still fail even after normalization."""
    hits = [_hit(1, "https://ex.com/a", "Completely unrelated content.")]
    fake = FakeLLMClient(responses_by_schema={
        "extract_company_name": {
            "value": "ACME Inc.",
            "source_url": "https://ex.com/a",
            "quote": "ACME Inc. was founded in 1998.",
            "reasoning": "hallucinated",
        }
    })
    result = extract_variable(client=fake, variable=_scalar_spec(),
                              hits=hits, model="fake")
    assert result["value"] is None
