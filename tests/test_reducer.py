from tarantula.config import VariableSpec
from tarantula.reducer import reduce_candidates, Candidate
from tarantula.llm import FakeLLMClient


def _vars():
    return [
        VariableSpec(name="company_name", type="string", description="Legal name.", required=True),
        VariableSpec(name="founded_year", type="integer", description="Year."),
        VariableSpec(name="products", type="array", items="string", description="Products."),
        VariableSpec(name="has_careers_page", type="boolean", description="Careers page?"),
    ]


def test_reduce_emits_null_when_no_candidates():
    fake = FakeLLMClient(responses=[{
        "company_name": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
        "founded_year": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
        "products": {"value": None, "sources": [], "reasoning": "n/a"},
        "has_careers_page": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
    }])
    results = reduce_candidates(
        client=fake, variables=_vars(), candidates_by_var={}, model="fake",
    )
    assert results["company_name"]["value"] is None
    assert results["company_name"]["required_missing"] is True
    assert results["has_careers_page"]["required_missing"] is False


def test_reduce_picks_winner_per_scalar():
    fake = FakeLLMClient(responses=[{
        "company_name": {
            "value": "ACME Inc.",
            "source_url": "https://a.com/about",
            "quote": "ACME Inc. is a private company.",
            "reasoning": "From /about.",
        },
        "founded_year": {"value": 1998, "source_url": "https://a.com/history",
                         "quote": "Founded in 1998.", "reasoning": "From history."},
        "products": {"value": None, "sources": [], "reasoning": "none"},
        "has_careers_page": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
    }])
    candidates = {
        "company_name": [
            Candidate(value="ACME Inc.", quote="ACME Inc. is a private company.",
                      source_url="https://a.com/about", page_title="About"),
            Candidate(value="ACME", quote="ACME blog post.",
                      source_url="https://a.com/blog/1", page_title="Blog"),
        ],
    }
    results = reduce_candidates(
        client=fake, variables=_vars(), candidates_by_var=candidates, model="fake",
    )
    assert results["company_name"]["value"] == "ACME Inc."
    assert results["company_name"]["source_url"] == "https://a.com/about"


def test_reduce_arrays_include_sources_pairs():
    fake = FakeLLMClient(responses=[{
        "company_name": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
        "founded_year": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
        "products": {
            "value": ["Widget Pro", "Widget Lite"],
            "sources": [
                {"value_item": "Widget Pro", "source_url": "https://a.com/p1", "quote": "Widget Pro is flagship."},
                {"value_item": "Widget Lite", "source_url": "https://a.com/p2", "quote": "Widget Lite for small teams."},
            ],
            "reasoning": "Union across product pages.",
        },
        "has_careers_page": {"value": None, "source_url": None, "quote": None, "reasoning": "n/a"},
    }])
    candidates = {
        "products": [
            Candidate(value=["Widget Pro"], quote="Widget Pro is flagship.",
                      source_url="https://a.com/p1", page_title="P1"),
            Candidate(value=["Widget Lite"], quote="Widget Lite for small teams.",
                      source_url="https://a.com/p2", page_title="P2"),
        ]
    }
    results = reduce_candidates(
        client=fake, variables=_vars(), candidates_by_var=candidates, model="fake",
    )
    assert results["products"]["value"] == ["Widget Pro", "Widget Lite"]
    assert len(results["products"]["sources"]) == 2
    assert results["products"]["sources"][0]["source_url"] == "https://a.com/p1"
