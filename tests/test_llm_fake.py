import pytest
from tarantula.llm import FakeLLMClient, LLMRequest


def test_fake_returns_scripted_response():
    client = FakeLLMClient(responses=[{"x": 1}])
    resp = client.complete_json(
        LLMRequest(
            system="sys",
            user="usr",
            json_schema={"type": "object"},
            model="fake",
            temperature=0,
        )
    )
    assert resp == {"x": 1}


def test_fake_raises_when_exhausted():
    client = FakeLLMClient(responses=[{"x": 1}])
    client.complete_json(
        LLMRequest(system="s", user="u", json_schema={}, model="fake", temperature=0)
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        client.complete_json(
            LLMRequest(system="s", user="u", json_schema={}, model="fake", temperature=0)
        )


def test_fake_records_requests():
    client = FakeLLMClient(responses=[{"ok": True}, {"ok": True}])
    client.complete_json(LLMRequest(system="sys1", user="u1", json_schema={}, model="m", temperature=0))
    client.complete_json(LLMRequest(system="sys2", user="u2", json_schema={}, model="m", temperature=0))
    assert len(client.calls) == 2
    assert client.calls[0].system == "sys1"
    assert client.calls[1].user == "u2"
