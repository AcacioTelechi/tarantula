import json
import yaml
from typer.testing import CliRunner
from tarantula.cli import app
from tarantula.llm import FakeLLMClient


def test_retrieve_prints_ranked_hits(tmp_path, monkeypatch):
    # Build a DB with two chunks under one crawl.
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("u", "v")
    crawl_id = store.start_crawl(run_id, "https://ex.com")
    p = store.save_page(url="https://ex.com/a", raw_bytes=b"<a/>",
                        http_status=200, content_type="text/html",
                        fetcher="http", title="A")
    store.link_page(crawl_id, p, depth=0, parent_url=None)
    c1 = store.save_chunk(p, 0, "Company founded in 1998.", 6)
    c2 = store.save_chunk(p, 1, "We sell widgets.", 4)
    store.save_chunk_embedding(c1, [1.0, 0.0], model="stub")
    store.save_chunk_embedding(c2, [0.0, 1.0], model="stub")
    store.finish_crawl(crawl_id, "ok", 1)
    store.finish_run(run_id, "ok")
    store.conn.close()

    # Inject the fake LLM client (used for query embedding only).
    fake = FakeLLMClient(embeddings_by_text={
        "founded_year The year the organization was founded.": [1.0, 0.0],
    })
    monkeypatch.setattr("tarantula.cli.OpenAIClient", lambda: fake)

    vars_path = tmp_path / "vars.yaml"
    vars_path.write_text(yaml.safe_dump({"variables": [
        {"name": "founded_year", "type": "integer",
         "description": "The year the organization was founded."}
    ]}))

    runner = CliRunner()
    result = runner.invoke(app, [
        "retrieve",
        "--db", str(tmp_path / "t.db"),
        "--data-dir", str(tmp_path / "data"),
        "--variables", str(vars_path),
        "--seed-url", "https://ex.com",
        "--variable", "founded_year",
        "--k", "2",
        "--json",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["variable"] == "founded_year"
    assert [h["chunk_id"] for h in payload["hits"]][0] == c1


def test_retrieve_mode_bm25_does_not_instantiate_openai_client(tmp_path, monkeypatch):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("u", "v")
    crawl_id = store.start_crawl(run_id, "https://ex.com")
    p = store.save_page(url="https://ex.com/a", raw_bytes=b"<a/>",
                        http_status=200, content_type="text/html",
                        fetcher="http", title="A")
    store.link_page(crawl_id, p, depth=0, parent_url=None)
    store.save_chunk(p, 0, "Company founded in 1998.", 6)
    store.finish_crawl(crawl_id, "ok", 1)
    store.finish_run(run_id, "ok")
    store.conn.close()

    instantiations: list[str] = []

    class ExplodingClient:
        def __init__(self):
            instantiations.append("OpenAIClient()")
            raise RuntimeError("OpenAIClient should not be constructed in bm25 mode")

    monkeypatch.setattr("tarantula.cli.OpenAIClient", ExplodingClient)

    vars_path = tmp_path / "vars.yaml"
    vars_path.write_text(yaml.safe_dump({"variables": [
        {"name": "founded_year", "type": "integer",
         "description": "The year the organization was founded."}
    ]}))

    runner = CliRunner()
    result = runner.invoke(app, [
        "retrieve",
        "--db", str(tmp_path / "t.db"),
        "--data-dir", str(tmp_path / "data"),
        "--variables", str(vars_path),
        "--seed-url", "https://ex.com",
        "--variable", "founded_year",
        "--mode", "bm25",
        "--k", "2",
        "--json",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert instantiations == []


def test_retrieve_reports_friendly_error_when_no_crawl(tmp_path, monkeypatch):
    from tarantula.store import Store
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    store.conn.close()

    # Keep OpenAIClient import path valid but never used in this path.
    monkeypatch.setattr("tarantula.cli.OpenAIClient", lambda: None)

    vars_path = tmp_path / "vars.yaml"
    vars_path.write_text(yaml.safe_dump({"variables": [
        {"name": "foo", "type": "string", "description": "x"}
    ]}))

    runner = CliRunner()
    result = runner.invoke(app, [
        "retrieve",
        "--db", str(tmp_path / "t.db"),
        "--data-dir", str(tmp_path / "data"),
        "--variables", str(vars_path),
        "--seed-url", "https://nope.example.com",
    ], catch_exceptions=False)
    assert result.exit_code == 1
    # The stderr message should contain a human-friendly "Error:" prefix,
    # not Typer's "Invalid value:" wording.
    assert "Error: no crawl found" in (result.stderr or result.output)
    assert "Invalid value" not in (result.stderr or result.output)
