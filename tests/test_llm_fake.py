import pytest
from pydantic import BaseModel
from tarantula.llm import FakeLLMClient, LLMRequest


class _M(BaseModel):
    x: int = 0


def test_fake_returns_scripted_response():
    client = FakeLLMClient(responses=[{"x": 1}])
    resp = client.complete_json(
        LLMRequest(
            system="sys",
            user="usr",
            response_model=_M,
            model="fake",
            temperature=0,
        )
    )
    assert resp == {"x": 1}


def test_fake_raises_when_exhausted():
    client = FakeLLMClient(responses=[{"x": 1}])
    client.complete_json(
        LLMRequest(system="s", user="u", response_model=_M, model="fake", temperature=0)
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete_json(
            LLMRequest(system="s", user="u", response_model=_M, model="fake", temperature=0)
        )


def test_fake_records_requests():
    client = FakeLLMClient(responses=[{"ok": True}, {"ok": True}])
    client.complete_json(LLMRequest(system="sys1", user="u1", response_model=_M, model="m", temperature=0))
    client.complete_json(LLMRequest(system="sys2", user="u2", response_model=_M, model="m", temperature=0))
    assert len(client.calls) == 2
    assert client.calls[0].system == "sys1"
    assert client.calls[1].user == "u2"


def test_fake_embed_returns_scripted_vector():
    fake = FakeLLMClient(embeddings_by_text={"hello world": [0.1, 0.2, 0.3]})
    out = fake.embed(["hello world"], model="stub")
    assert out == [[0.1, 0.2, 0.3]]


def test_fake_embed_falls_back_to_deterministic_vector():
    fake = FakeLLMClient()
    out = fake.embed(["alpha", "beta"], model="stub")
    assert len(out) == 2
    assert len(out[0]) == 8  # default dim
    # Deterministic: same input → same output
    assert fake.embed(["alpha"], model="stub")[0] == out[0]
    # Different inputs → different outputs
    assert out[0] != out[1]


def test_fake_embed_records_calls():
    fake = FakeLLMClient()
    fake.embed(["x", "y"], model="m1")
    assert fake.embed_calls == [(["x", "y"], "m1")]
