from tarantula.agent_tools import Corpus
from tarantula.agent_extractor import extract_variable_agent, extract_all_agent
from tarantula.config import VariableSpec
from tarantula.llm import FakeLLMClient


def _corpus():
    return Corpus.from_pages([
        ("https://a.example/", "Home", "Welcome.\nCNPJ: 12.345.678/0001-90"),
        ("https://a.example/about", "About", "Founded in 1998 in Rio."),
    ])


def _var(**kw):
    base = dict(name="cnpj", type="string", description="tax id",
                extraction_type="agent")
    base.update(kw)
    return VariableSpec(**base)


def _grep_then_answer(value, source_url, quote):
    return [
        {"thought": "search", "action": "grep",
         "pattern": "CNPJ", "ignore_case": True, "url": None, "answer": None},
        {"thought": "found", "action": "answer",
         "pattern": None, "ignore_case": None, "url": None,
         "answer": {"value": value, "source_url": source_url,
                    "quote": quote, "reasoning": "matched"}},
    ]


def test_agent_greps_then_answers_with_grounded_quote():
    fake = FakeLLMClient(responses=_grep_then_answer(
        "12.345.678/0001-90", "https://a.example/", "CNPJ: 12.345.678/0001-90"))
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=8)
    assert out["value"] == "12.345.678/0001-90"
    assert out["source_url"] == "https://a.example/"
    assert out["required_missing"] is False
    assert len(fake.calls) == 2


def test_agent_explicit_null_answer_returns_immediately():
    fake = FakeLLMClient(responses=[
        {"thought": "give up", "action": "answer",
         "pattern": None, "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None,
                    "quote": None, "reasoning": "not present"}},
    ])
    out = extract_variable_agent(
        client=fake, variable=_var(required=True), corpus=_corpus(),
        model="fake", max_steps=8)
    assert out["value"] is None
    assert out["required_missing"] is True
    assert len(fake.calls) == 1


def test_agent_retries_when_quote_not_grounded():
    bad = {"thought": "guess", "action": "answer",
           "pattern": None, "ignore_case": None, "url": None,
           "answer": {"value": "00.000.000/0000-00",
                      "source_url": "https://a.example/",
                      "quote": "CNPJ: 00.000.000/0000-00", "reasoning": "hallucinated"}}
    good = {"thought": "fixed", "action": "answer",
            "pattern": None, "ignore_case": None, "url": None,
            "answer": {"value": "12.345.678/0001-90",
                       "source_url": "https://a.example/",
                       "quote": "CNPJ: 12.345.678/0001-90", "reasoning": "real"}}
    fake = FakeLLMClient(responses=[bad, good])
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=8)
    assert out["value"] == "12.345.678/0001-90"
    assert len(fake.calls) == 2


def test_agent_returns_null_when_step_budget_exhausted():
    grep_forever = {"thought": "search", "action": "grep",
                    "pattern": "x", "ignore_case": True, "url": None, "answer": None}
    fake = FakeLLMClient(responses=[grep_forever, grep_forever, grep_forever])
    out = extract_variable_agent(
        client=fake, variable=_var(required=True), corpus=_corpus(),
        model="fake", max_steps=3)
    assert out["value"] is None
    assert out["required_missing"] is True
    assert "exhausted" in out["reasoning"]
    assert len(fake.calls) == 3


def test_agent_hint_appears_in_system_prompt():
    fake = FakeLLMClient(responses=[
        {"thought": "x", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None, "quote": None,
                    "reasoning": "n/a"}},
    ])
    extract_variable_agent(
        client=fake, variable=_var(hint="14 digits XX.XXX.XXX/XXXX-XX"),
        corpus=_corpus(), model="fake", max_steps=4)
    assert "14 digits XX.XXX.XXX/XXXX-XX" in fake.calls[0].system


def test_agent_unknown_url_observation_does_not_crash():
    fake = FakeLLMClient(responses=[
        {"thought": "read", "action": "read_page", "pattern": None,
         "ignore_case": None, "url": "https://nope", "answer": None},
        {"thought": "done", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {"value": None, "source_url": None, "quote": None,
                    "reasoning": "n/a"}},
    ])
    out = extract_variable_agent(
        client=fake, variable=_var(), corpus=_corpus(), model="fake", max_steps=5)
    assert out["value"] is None
    assert len(fake.calls) == 2


def _array_corpus():
    return Corpus.from_pages([
        ("https://a.example/products", "Products",
         "We sell Widget Pro and Widget Lite to everyone."),
    ])


def test_agent_array_collects_grounded_sources():
    fake = FakeLLMClient(responses=[
        {"thought": "answer", "action": "answer", "pattern": None,
         "ignore_case": None, "url": None,
         "answer": {
             "value": ["Widget Pro", "Widget Lite"],
             "sources": [
                 {"value_item": "Widget Pro",
                  "source_url": "https://a.example/products",
                  "quote": "Widget Pro"},
                 {"value_item": "Widget Lite",
                  "source_url": "https://a.example/products",
                  "quote": "Widget Lite"},
             ],
             "reasoning": "listed",
         }},
    ])
    v = VariableSpec(name="products", type="array", items="string",
                     description="products", extraction_type="agent")
    out = extract_variable_agent(
        client=fake, variable=v, corpus=_array_corpus(), model="fake", max_steps=5)
    assert out["value"] == ["Widget Pro", "Widget Lite"]
    assert len(out["sources"]) == 2


def test_agent_array_drops_ungrounded_items_and_retries():
    bad = {"thought": "guess", "action": "answer", "pattern": None,
           "ignore_case": None, "url": None,
           "answer": {"value": ["Phantom"],
                      "sources": [{"value_item": "Phantom",
                                   "source_url": "https://a.example/products",
                                   "quote": "Phantom Device"}],
                      "reasoning": "hallucinated"}}
    good = {"thought": "fix", "action": "answer", "pattern": None,
            "ignore_case": None, "url": None,
            "answer": {"value": ["Widget Pro"],
                       "sources": [{"value_item": "Widget Pro",
                                    "source_url": "https://a.example/products",
                                    "quote": "Widget Pro"}],
                       "reasoning": "real"}}
    fake = FakeLLMClient(responses=[bad, good])
    v = VariableSpec(name="products", type="array", items="string",
                     description="products", extraction_type="agent")
    out = extract_variable_agent(
        client=fake, variable=v, corpus=_array_corpus(), model="fake", max_steps=5)
    assert out["value"] == ["Widget Pro"]
    assert len(fake.calls) == 2


def test_extract_all_agent_preserves_order_and_resolves_max_steps():
    # Two agent variables, each answered in a single step. Using
    # responses_by_schema keyed by schema_name makes responses addressable
    # per-variable, which is concurrency-safe across the thread pool.
    fake = FakeLLMClient(responses_by_schema={
        "agent_nome": {
            "thought": "x", "action": "answer", "pattern": None,
            "ignore_case": None, "url": None,
            "answer": {"value": "ACME", "source_url": "https://a.example/",
                       "quote": "Welcome.", "reasoning": "home"}},
        "agent_cnpj": {
            "thought": "x", "action": "answer", "pattern": None,
            "ignore_case": None, "url": None,
            "answer": {"value": "12.345.678/0001-90",
                       "source_url": "https://a.example/",
                       "quote": "CNPJ: 12.345.678/0001-90", "reasoning": "home"}},
    })
    variables = [
        VariableSpec(name="nome", type="string", description="name",
                     extraction_type="agent"),
        VariableSpec(name="cnpj", type="string", description="id",
                     extraction_type="agent", max_steps=3),
    ]
    out = extract_all_agent(
        client=fake, variables=variables, corpus=_corpus(), model="fake")
    # Order matches input variable order.
    assert list(out.keys()) == ["nome", "cnpj"]
    assert out["nome"]["value"] == "ACME"
    assert out["cnpj"]["value"] == "12.345.678/0001-90"


def test_extract_all_agent_empty_returns_empty():
    fake = FakeLLMClient(responses=[])
    out = extract_all_agent(client=fake, variables=[], corpus=_corpus(),
                            model="fake")
    assert out == {}
