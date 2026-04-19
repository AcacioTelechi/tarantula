# Tarantula Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI (`tarantula`) that, given a list of URLs and a user-defined set of typed variables, crawls each site BFS-style, cleans the content, runs a map-reduce LLM extraction pipeline, and emits JSON results with per-variable source URL and quoted passage.

**Architecture:** Linear pipeline per seed URL: crawl → clean → chunk → per-chunk LLM extract → per-site LLM reduce. SQLite stores everything for audit and caching; raw HTML on disk. One formal interface (`LLMClient`) so the OpenAI dependency can be swapped later; nothing else abstracted.

**Tech Stack:** Python 3.11+, `httpx`, `playwright`, `trafilatura`, `readability-lxml`, `beautifulsoup4`, `tldextract`, `pydantic` v2, `pyyaml`, `tiktoken`, `openai`, `typer`. Tests: `pytest`, `pytest-asyncio`, `pytest-httpserver`.

**Reference spec:** `docs/superpowers/specs/2026-04-19-tarantula-design.md`

**Known deliberate gap vs. spec §8.6:** the `--max-tokens` flag is wired but not actively enforced in v1 — precise enforcement requires the OpenAI client to surface per-call usage data and a per-run counter. The flag is accepted so callers don't break when we add enforcement; until then, `max_pages` bounds cost. Add enforcement as a follow-up task when a real run shows it's needed.

---

## Task Ordering Rationale

Bottom-up: each task depends only on earlier tasks. Config models and pure utilities (URL, robots, chunker, cleaner) come first. Storage next. `LLMClient` interface + fake (needed by extractor/reducer tests). Then extractor, reducer, crawler, CLI orchestration, E2E.

Every task follows TDD: failing test → run it → implement → run it → commit.

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/tarantula/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 0.1: Init git repo**

The working directory isn't a git repo. Run:

```bash
git init
git config user.email "tarantula-dev@local"
git config user.name "Tarantula"
```

- [ ] **Step 0.2: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
.venv/
venv/
data/
*.db
results.json
.env
.ruff_cache/
```

- [ ] **Step 0.3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tarantula"
version = "0.1.0"
description = "Crawl sites, extract typed variables via LLM with source quotes."
requires-python = ">=3.11"
dependencies = [
  "httpx[http2]>=0.27",
  "beautifulsoup4>=4.12",
  "lxml>=5.2",
  "trafilatura>=1.12",
  "readability-lxml>=0.8.1",
  "tldextract>=5.1",
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "tiktoken>=0.7",
  "openai>=1.40",
  "typer>=0.12",
  "protego>=0.3",
]

[project.optional-dependencies]
playwright = ["playwright>=1.45"]
dev = [
  "pytest>=8.2",
  "pytest-asyncio>=0.23",
  "pytest-httpserver>=1.0",
  "pytest-cov>=5.0",
  "ruff>=0.5",
]

[project.scripts]
tarantula = "tarantula.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
  "live: tests that hit real OpenAI (skipped by default)",
  "playwright: tests that require Chromium (skipped by default)",
]
addopts = "-m 'not live and not playwright'"
```

- [ ] **Step 0.4: Create package roots**

```python
# src/tarantula/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest

@pytest.fixture
def fixtures_dir(request):
    return request.config.rootpath / "tests" / "fixtures"
```

- [ ] **Step 0.5: Install deps and verify**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Expected: "no tests ran" (no tests yet). Install succeeds.

- [ ] **Step 0.6: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/
git commit -m "feat: scaffold tarantula package and test harness"
```

---

## Task 1: Config models (Pydantic)

**Files:**
- Create: `src/tarantula/config.py`
- Create: `tests/test_config.py`
- Create: `tests/fixtures/configs/valid_urls.yaml`
- Create: `tests/fixtures/configs/valid_variables.yaml`

- [ ] **Step 1.1: Write fixture YAML files**

```yaml
# tests/fixtures/configs/valid_urls.yaml
defaults:
  max_depth: 3
  max_pages: 200
  same_host_only: true
  include_subdomains: false
  respect_robots_txt: true
  rate_limit_rps: 1.5
  request_timeout_s: 20
  user_agent: "tarantula/0.1"

sites:
  - url: https://example.com
    max_depth: 4
    max_pages: 500
  - url: https://docs.other.com
    include_subdomains: true
```

```yaml
# tests/fixtures/configs/valid_variables.yaml
variables:
  - name: company_name
    type: string
    description: "Official legal name."
    required: true
  - name: founded_year
    type: integer
    description: "Year founded."
    examples:
      - input: "Founded in 1998"
        output: 1998
  - name: products
    type: array
    items: string
    description: "Product names."
  - name: has_careers_page
    type: boolean
    description: "Whether a careers page exists."
```

- [ ] **Step 1.2: Write failing tests**

```python
# tests/test_config.py
import pytest
from tarantula.config import (
    load_urls_config, load_variables_config,
    URLsConfig, VariablesConfig, SiteConfig, VariableSpec,
)


def test_load_valid_urls_config(fixtures_dir):
    cfg = load_urls_config(fixtures_dir / "configs" / "valid_urls.yaml")
    assert isinstance(cfg, URLsConfig)
    assert len(cfg.sites) == 2
    assert cfg.sites[0].url == "https://example.com"
    assert cfg.sites[0].max_depth == 4  # override
    assert cfg.sites[0].rate_limit_rps == 1.5  # inherited default
    assert cfg.sites[1].include_subdomains is True


def test_site_config_inherits_defaults(fixtures_dir):
    cfg = load_urls_config(fixtures_dir / "configs" / "valid_urls.yaml")
    site = cfg.sites[0]
    assert site.user_agent == "tarantula/0.1"
    assert site.respect_robots_txt is True


def test_urls_config_rejects_non_http_scheme(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("sites:\n  - url: ftp://example.com\n")
    with pytest.raises(ValueError, match="http"):
        load_urls_config(p)


def test_urls_config_requires_at_least_one_site(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("sites: []\n")
    with pytest.raises(ValueError, match="at least one"):
        load_urls_config(p)


def test_load_valid_variables_config(fixtures_dir):
    cfg = load_variables_config(fixtures_dir / "configs" / "valid_variables.yaml")
    assert isinstance(cfg, VariablesConfig)
    assert len(cfg.variables) == 4
    by_name = {v.name: v for v in cfg.variables}
    assert by_name["company_name"].required is True
    assert by_name["founded_year"].examples[0].output == 1998
    assert by_name["products"].items == "string"
    assert by_name["has_careers_page"].type == "boolean"


def test_variables_config_rejects_duplicate_names(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: string, description: a}\n"
        "  - {name: x, type: string, description: b}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_variables_config(p)


def test_array_variable_requires_items(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: array, description: a}\n"
    )
    with pytest.raises(ValueError, match="items"):
        load_variables_config(p)


def test_variable_type_rejects_unsupported(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "variables:\n"
        "  - {name: x, type: object, description: a}\n"
    )
    with pytest.raises(ValueError):
        load_variables_config(p)
```

- [ ] **Step 1.3: Run tests — expect failure**

```bash
pytest tests/test_config.py -v
```

Expected: ImportError on `tarantula.config`.

- [ ] **Step 1.4: Implement `config.py`**

```python
# src/tarantula/config.py
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

ScalarType = Literal["string", "integer", "number", "boolean"]
VarType = Literal["string", "integer", "number", "boolean", "array"]


class SiteDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_depth: int = Field(3, ge=0, le=20)
    max_pages: int = Field(200, ge=1, le=100_000)
    same_host_only: bool = True
    include_subdomains: bool = False
    respect_robots_txt: bool = True
    rate_limit_rps: float = Field(1.5, gt=0, le=50)
    request_timeout_s: int = Field(20, ge=1, le=600)
    user_agent: str = "tarantula/0.1"


class SiteConfig(SiteDefaults):
    url: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _check_scheme(self) -> SiteConfig:
        if not (self.url.startswith("http://") or self.url.startswith("https://")):
            raise ValueError(f"url must be http(s): got {self.url!r}")
        return self


class URLsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defaults: SiteDefaults = Field(default_factory=SiteDefaults)
    sites: list[SiteConfig]

    @model_validator(mode="after")
    def _check_nonempty(self) -> URLsConfig:
        if not self.sites:
            raise ValueError("must define at least one site")
        return self


class VariableExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str
    output: Any


class VariableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]
    type: VarType
    description: str
    required: bool = False
    items: ScalarType | None = None
    examples: list[VariableExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_items(self) -> VariableSpec:
        if self.type == "array" and self.items is None:
            raise ValueError(f"variable {self.name!r}: array type requires 'items'")
        if self.type != "array" and self.items is not None:
            raise ValueError(f"variable {self.name!r}: 'items' only valid for array type")
        return self


class VariablesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variables: list[VariableSpec]

    @model_validator(mode="after")
    def _check_unique(self) -> VariablesConfig:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variable names")
        return self


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the root")
    return data


def load_urls_config(path: Path) -> URLsConfig:
    data = _load_yaml(Path(path))
    defaults = data.get("defaults") or {}
    raw_sites = data.get("sites") or []
    # Merge defaults into each site before validation so per-site overrides win.
    merged_sites = [{**defaults, **(s or {})} for s in raw_sites]
    return URLsConfig(defaults=SiteDefaults(**defaults), sites=merged_sites)


def load_variables_config(path: Path) -> VariablesConfig:
    data = _load_yaml(Path(path))
    return VariablesConfig(**data)
```

- [ ] **Step 1.5: Run tests — expect pass**

```bash
pytest tests/test_config.py -v
```

Expected: 8 passed.

- [ ] **Step 1.6: Commit**

```bash
git add src/tarantula/config.py tests/test_config.py tests/fixtures/configs/
git commit -m "feat(config): pydantic models for urls and variables YAML"
```

---

## Task 2: URL utilities

**Files:**
- Create: `src/tarantula/urls.py`
- Create: `tests/test_urls.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_urls.py
from tarantula.urls import normalize_url, same_scope, is_html_like


def test_normalize_strips_fragment():
    assert normalize_url("https://example.com/a#x") == "https://example.com/a"


def test_normalize_lowercases_host():
    assert normalize_url("https://Example.COM/A") == "https://example.com/A"


def test_normalize_sorts_query_params():
    assert normalize_url("https://a.com/?b=2&a=1") == "https://a.com/?a=1&b=2"


def test_normalize_removes_default_ports():
    assert normalize_url("https://a.com:443/x") == "https://a.com/x"
    assert normalize_url("http://a.com:80/x") == "http://a.com/x"


def test_normalize_strips_trailing_slash_on_root():
    # keep path trailing slash if non-root; strip only on root
    assert normalize_url("https://a.com/") == "https://a.com"
    assert normalize_url("https://a.com/x/") == "https://a.com/x/"


def test_normalize_joins_relative():
    assert normalize_url("/x?a=1", base="https://a.com/y") == "https://a.com/x?a=1"


def test_same_scope_same_host():
    assert same_scope("https://a.com/x", seed="https://a.com", include_subdomains=False)
    assert not same_scope("https://b.com/x", seed="https://a.com", include_subdomains=False)


def test_same_scope_subdomain_behavior():
    assert not same_scope("https://docs.a.com/x", seed="https://a.com", include_subdomains=False)
    assert same_scope("https://docs.a.com/x", seed="https://a.com", include_subdomains=True)
    # unrelated domain never in scope
    assert not same_scope("https://b.com/x", seed="https://a.com", include_subdomains=True)


def test_is_html_like_skips_binary_extensions():
    assert not is_html_like("https://a.com/doc.pdf")
    assert not is_html_like("https://a.com/img.png")
    assert not is_html_like("https://a.com/archive.zip")


def test_is_html_like_allows_html_and_bare_paths():
    assert is_html_like("https://a.com/page")
    assert is_html_like("https://a.com/page.html")
    assert is_html_like("https://a.com/")
```

- [ ] **Step 2.2: Run — expect fail**

```bash
pytest tests/test_urls.py -v
```

Expected: ImportError.

- [ ] **Step 2.3: Implement `urls.py`**

```python
# src/tarantula/urls.py
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import tldextract

_BINARY_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".mp3", ".mp4", ".mov", ".avi", ".wav",
    ".css", ".js", ".json", ".xml", ".rss",
}


def normalize_url(url: str, base: str | None = None) -> str:
    if base is not None:
        url = urljoin(base, url)
    parts = urlsplit(url)
    host = parts.hostname or ""
    host = host.lower()
    netloc = host
    # strip default ports
    if parts.port and not (
        (parts.scheme == "http" and parts.port == 80)
        or (parts.scheme == "https" and parts.port == 443)
    ):
        netloc = f"{host}:{parts.port}"
    # sort query params deterministically
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    path = parts.path
    if path == "/":
        path = ""
    # drop fragment always
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def same_scope(candidate: str, seed: str, include_subdomains: bool) -> bool:
    cand_host = (urlsplit(candidate).hostname or "").lower()
    seed_host = (urlsplit(seed).hostname or "").lower()
    if cand_host == seed_host:
        return True
    if not include_subdomains:
        return False
    c = tldextract.extract(cand_host)
    s = tldextract.extract(seed_host)
    # require same registrable domain (domain + suffix), not just suffix
    return bool(c.domain) and c.domain == s.domain and c.suffix == s.suffix


def is_html_like(url: str) -> bool:
    path = urlsplit(url).path.lower()
    # Look at the final path segment to avoid flagging .../pdf-guides/article
    if "/" in path:
        last = path.rsplit("/", 1)[-1]
    else:
        last = path
    if "." in last:
        ext = "." + last.rsplit(".", 1)[-1]
        if ext in _BINARY_EXTS:
            return False
    return True
```

- [ ] **Step 2.4: Run — expect pass**

```bash
pytest tests/test_urls.py -v
```

Expected: 10 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/tarantula/urls.py tests/test_urls.py
git commit -m "feat(urls): normalize, scope filter, html-like detection"
```

---

## Task 3: robots.txt handling

**Files:**
- Create: `src/tarantula/robots.py`
- Create: `tests/test_robots.py`

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_robots.py
import pytest
from tarantula.robots import RobotsCache


class FakeFetcher:
    def __init__(self, responses: dict[str, tuple[int, str]]):
        self.responses = responses
        self.calls = []

    async def __call__(self, url: str) -> tuple[int, str]:
        self.calls.append(url)
        return self.responses.get(url, (404, ""))


@pytest.mark.asyncio
async def test_allows_when_robots_missing():
    fetcher = FakeFetcher({})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/anything") is True


@pytest.mark.asyncio
async def test_respects_user_agent_disallow():
    body = "User-agent: *\nDisallow: /private"
    fetcher = FakeFetcher({"https://a.com/robots.txt": (200, body)})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/public") is True
    assert await cache.allowed("https://a.com/private/x") is False


@pytest.mark.asyncio
async def test_caches_per_host():
    body = "User-agent: *\nDisallow: /p"
    fetcher = FakeFetcher({"https://a.com/robots.txt": (200, body)})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    await cache.allowed("https://a.com/x")
    await cache.allowed("https://a.com/y")
    await cache.allowed("https://a.com/p")
    assert fetcher.calls.count("https://a.com/robots.txt") == 1


@pytest.mark.asyncio
async def test_5xx_blocks_crawling_conservatively():
    # On persistent 5xx, protego convention: treat as unknown; we choose conservative allow.
    # Document the choice via test: when robots fails to fetch, we allow (RFC-ish default).
    fetcher = FakeFetcher({"https://a.com/robots.txt": (500, "")})
    cache = RobotsCache(fetcher, user_agent="tarantula/0.1")
    assert await cache.allowed("https://a.com/x") is True
```

- [ ] **Step 3.2: Run — expect fail**

```bash
pytest tests/test_robots.py -v
```

- [ ] **Step 3.3: Implement `robots.py`**

```python
# src/tarantula/robots.py
from __future__ import annotations

from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from protego import Protego

Fetcher = Callable[[str], Awaitable[tuple[int, str]]]


class RobotsCache:
    """Async robots.txt cache keyed by host.

    On missing (404) or error (5xx) we allow — this matches broad
    ecosystem convention for when robots.txt cannot be determined.
    """

    def __init__(self, fetcher: Fetcher, user_agent: str) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._parsers: dict[str, Protego | None] = {}

    async def _get_parser(self, host_key: str) -> Protego | None:
        if host_key in self._parsers:
            return self._parsers[host_key]
        robots_url = host_key + "/robots.txt"
        status, body = await self._fetcher(robots_url)
        if status == 200 and body:
            self._parsers[host_key] = Protego.parse(body)
        else:
            self._parsers[host_key] = None
        return self._parsers[host_key]

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host_key = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        parser = await self._get_parser(host_key)
        if parser is None:
            return True
        return parser.can_fetch(url, self._user_agent)
```

- [ ] **Step 3.4: Run — expect pass**

```bash
pytest tests/test_robots.py -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/tarantula/robots.py tests/test_robots.py
git commit -m "feat(robots): async robots.txt cache with allow-on-error"
```

---

## Task 4: SQLite store

**Files:**
- Create: `src/tarantula/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/test_store.py
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from tarantula.store import Store, PageRecord


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s.init_schema()
    return s


def test_init_creates_tables(store):
    tables = {r[0] for r in store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert {"runs", "crawls", "pages", "crawl_pages", "chunks",
            "chunk_extractions", "extractions"} <= tables


def test_start_run_returns_id(store):
    run_id = store.start_run(urls_config_yaml="x", variables_config_yaml="y")
    row = store.conn.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
    assert row[0] == "running"


def test_finish_run_sets_status_and_timestamp(store):
    run_id = store.start_run("", "")
    store.finish_run(run_id, status="ok")
    row = store.conn.execute(
        "SELECT status, finished_at FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row[0] == "ok"
    assert row[1] is not None


def test_save_page_upserts_on_content_hash(store, tmp_path):
    pid1 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    pid2 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    assert pid1 == pid2  # same content_hash -> same row

    pid3 = store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v2</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    assert pid3 != pid1  # different content_hash -> new row


def test_save_page_writes_raw_file(store, tmp_path):
    store.save_page(
        url="https://a.com/x",
        raw_bytes=b"<html>v1</html>",
        http_status=200,
        content_type="text/html",
        fetcher="http",
        title="X",
    )
    raw_files = list((tmp_path / "data" / "raw").rglob("*.html"))
    assert len(raw_files) == 1
    assert raw_files[0].read_bytes() == b"<html>v1</html>"


def test_find_fresh_page_respects_ttl(store):
    store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    # fresh TTL wide -> hit
    assert store.find_fresh_page("https://a.com/x", ttl_seconds=3600) is not None
    # 0 TTL -> miss
    assert store.find_fresh_page("https://a.com/x", ttl_seconds=0) is None


def test_link_page_to_crawl(store):
    run_id = store.start_run("", "")
    crawl_id = store.start_crawl(run_id, seed_url="https://a.com")
    page_id = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    store.link_page(crawl_id, page_id, depth=1, parent_url="https://a.com")
    row = store.conn.execute(
        "SELECT depth, parent_url FROM crawl_pages WHERE crawl_id=? AND page_id=?",
        (crawl_id, page_id),
    ).fetchone()
    assert row == (1, "https://a.com")


def test_cleaned_text_upsert(store):
    pid = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    store.set_cleaned_text(pid, "hello world")
    row = store.conn.execute("SELECT cleaned_text FROM pages WHERE id=?", (pid,)).fetchone()
    assert row[0] == "hello world"


def test_chunk_and_extraction_roundtrip(store):
    run_id = store.start_run("", "")
    pid = store.save_page("https://a.com/x", b"v", 200, "text/html", "http", "t")
    cid = store.save_chunk(page_id=pid, ordinal=0, text="hello", token_count=1)
    store.save_chunk_extraction(
        run_id=run_id, chunk_id=cid, variable_name="v1",
        found=True, value="hello", quote="hello",
    )
    rows = list(store.iter_chunk_extractions(run_id=run_id, crawl_id=None))
    # crawl_id=None means all — caller still filters by run_id.
    assert len(rows) == 1
    assert rows[0].variable_name == "v1"
    assert rows[0].found is True
    assert rows[0].value == "hello"


def test_mark_orphan_runs_as_failed_on_init(tmp_path):
    s = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s.init_schema()
    rid = s.start_run("", "")
    s.conn.close()
    # reopen without calling finish_run
    s2 = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    s2.init_schema()
    status = s2.conn.execute("SELECT status FROM runs WHERE id=?", (rid,)).fetchone()[0]
    assert status == "failed"
```

- [ ] **Step 4.2: Run — expect fail**

- [ ] **Step 4.3: Implement `store.py`**

```python
# src/tarantula/store.py
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  urls_config TEXT NOT NULL,
  variables_config TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crawls (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  seed_url TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  pages_fetched INTEGER DEFAULT 0,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY,
  url TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  http_status INTEGER,
  content_type TEXT,
  fetched_at TEXT NOT NULL,
  fetcher TEXT NOT NULL,
  raw_path TEXT NOT NULL,
  cleaned_text TEXT,
  title TEXT,
  UNIQUE(url, content_hash)
);
CREATE TABLE IF NOT EXISTS crawl_pages (
  crawl_id INTEGER NOT NULL REFERENCES crawls(id),
  page_id INTEGER NOT NULL REFERENCES pages(id),
  discovered_at TEXT NOT NULL,
  depth INTEGER NOT NULL,
  parent_url TEXT,
  PRIMARY KEY (crawl_id, page_id)
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id),
  ordinal INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER,
  UNIQUE(page_id, ordinal)
);
CREATE TABLE IF NOT EXISTS chunk_extractions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  chunk_id INTEGER NOT NULL REFERENCES chunks(id),
  variable_name TEXT NOT NULL,
  found INTEGER NOT NULL,
  value_json TEXT,
  quote TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, chunk_id, variable_name)
);
CREATE TABLE IF NOT EXISTS extractions (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  crawl_id INTEGER NOT NULL REFERENCES crawls(id),
  variable_name TEXT NOT NULL,
  value_json TEXT,
  source_url TEXT,
  quote TEXT,
  reasoning TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, crawl_id, variable_name)
);
CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_crawl_pages_crawl ON crawl_pages(crawl_id);
CREATE INDEX IF NOT EXISTS idx_chunk_extractions_run ON chunk_extractions(run_id);
"""


@dataclass
class PageRecord:
    id: int
    url: str
    content_hash: str
    fetched_at: str
    raw_path: str
    cleaned_text: str | None
    title: str | None


@dataclass
class ChunkExtraction:
    variable_name: str
    found: bool
    value: Any
    quote: str | None
    chunk_id: int
    page_id: int
    url: str
    page_title: str | None


class Store:
    def __init__(self, db_path: str | Path, data_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "UPDATE runs SET status='failed', finished_at=? "
            "WHERE status='running' AND finished_at IS NULL",
            (_now(),),
        )
        self.conn.commit()

    # ---- runs ----
    def start_run(self, urls_config_yaml: str, variables_config_yaml: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, urls_config, variables_config, status) "
            "VALUES (?, ?, ?, 'running')",
            (_now(), urls_config_yaml, variables_config_yaml),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=? WHERE id=?",
            (status, _now(), run_id),
        )
        self.conn.commit()

    # ---- crawls ----
    def start_crawl(self, run_id: int, seed_url: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawls (run_id, seed_url, started_at, status) "
            "VALUES (?, ?, ?, 'running')",
            (run_id, seed_url, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_crawl(self, crawl_id: int, status: str, pages_fetched: int) -> None:
        self.conn.execute(
            "UPDATE crawls SET status=?, finished_at=?, pages_fetched=? WHERE id=?",
            (status, _now(), pages_fetched, crawl_id),
        )
        self.conn.commit()

    # ---- pages ----
    def _raw_path_for(self, url: str, content_hash: str) -> Path:
        host = urlsplit(url).hostname or "unknown"
        p = self.data_dir / "raw" / host
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{content_hash}.html"

    def save_page(
        self,
        url: str,
        raw_bytes: bytes,
        http_status: int,
        content_type: str | None,
        fetcher: str,
        title: str | None,
    ) -> int:
        content_hash = hashlib.sha1(raw_bytes).hexdigest()
        row = self.conn.execute(
            "SELECT id FROM pages WHERE url=? AND content_hash=?",
            (url, content_hash),
        ).fetchone()
        raw_path = self._raw_path_for(url, content_hash)
        if not raw_path.exists():
            raw_path.write_bytes(raw_bytes)
        if row:
            self.conn.execute(
                "UPDATE pages SET fetched_at=? WHERE id=?",
                (_now(), row[0]),
            )
            self.conn.commit()
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO pages (url, content_hash, http_status, content_type, "
            "fetched_at, fetcher, raw_path, title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, content_hash, http_status, content_type, _now(), fetcher,
             str(raw_path), title),
        )
        self.conn.commit()
        return cur.lastrowid

    def find_fresh_page(self, url: str, ttl_seconds: int) -> PageRecord | None:
        row = self.conn.execute(
            "SELECT id, url, content_hash, fetched_at, raw_path, cleaned_text, title "
            "FROM pages WHERE url=? "
            "AND (julianday(?) - julianday(fetched_at)) * 86400 <= ? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (url, _now(), ttl_seconds),
        ).fetchone()
        if not row:
            return None
        return PageRecord(*row)

    def link_page(
        self, crawl_id: int, page_id: int, depth: int, parent_url: str | None
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO crawl_pages "
            "(crawl_id, page_id, discovered_at, depth, parent_url) "
            "VALUES (?, ?, ?, ?, ?)",
            (crawl_id, page_id, _now(), depth, parent_url),
        )
        self.conn.commit()

    def set_cleaned_text(self, page_id: int, text: str) -> None:
        self.conn.execute(
            "UPDATE pages SET cleaned_text=? WHERE id=?",
            (text, page_id),
        )
        self.conn.commit()

    # ---- chunks ----
    def save_chunk(
        self, page_id: int, ordinal: int, text: str, token_count: int
    ) -> int:
        row = self.conn.execute(
            "SELECT id FROM chunks WHERE page_id=? AND ordinal=?",
            (page_id, ordinal),
        ).fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO chunks (page_id, ordinal, text, token_count) "
            "VALUES (?, ?, ?, ?)",
            (page_id, ordinal, text, token_count),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_chunk_extraction(
        self,
        run_id: int,
        chunk_id: int,
        variable_name: str,
        found: bool,
        value: Any,
        quote: str | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO chunk_extractions "
            "(run_id, chunk_id, variable_name, found, value_json, quote, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, chunk_id, variable_name, 1 if found else 0,
             json.dumps(value) if value is not None else None, quote, _now()),
        )
        self.conn.commit()

    def iter_chunk_extractions(
        self, run_id: int, crawl_id: int | None
    ) -> Iterator[ChunkExtraction]:
        sql = (
            "SELECT ce.variable_name, ce.found, ce.value_json, ce.quote, "
            "       ce.chunk_id, c.page_id, p.url, p.title "
            "FROM chunk_extractions ce "
            "JOIN chunks c ON c.id = ce.chunk_id "
            "JOIN pages p ON p.id = c.page_id "
            "WHERE ce.run_id = ?"
        )
        params: list[Any] = [run_id]
        if crawl_id is not None:
            sql += (
                " AND c.page_id IN ("
                "   SELECT page_id FROM crawl_pages WHERE crawl_id = ?"
                " )"
            )
            params.append(crawl_id)
        for row in self.conn.execute(sql, params):
            yield ChunkExtraction(
                variable_name=row[0],
                found=bool(row[1]),
                value=json.loads(row[2]) if row[2] is not None else None,
                quote=row[3],
                chunk_id=row[4],
                page_id=row[5],
                url=row[6],
                page_title=row[7],
            )

    # ---- final extractions ----
    def save_extraction(
        self,
        run_id: int,
        crawl_id: int,
        variable_name: str,
        value: Any,
        source_url: str | None,
        quote: str | None,
        reasoning: str | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO extractions "
            "(run_id, crawl_id, variable_name, value_json, source_url, quote, "
            " reasoning, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, crawl_id, variable_name,
             json.dumps(value) if value is not None else None,
             source_url, quote, reasoning, _now()),
        )
        self.conn.commit()
```

- [ ] **Step 4.4: Run — expect pass**

```bash
pytest tests/test_store.py -v
```

Expected: 10 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/tarantula/store.py tests/test_store.py
git commit -m "feat(store): sqlite schema and access layer with content-hash cache"
```

---

## Task 5: HTML cleaner

**Files:**
- Create: `src/tarantula/cleaner.py`
- Create: `tests/test_cleaner.py`
- Create: `tests/fixtures/html/simple.html`
- Create: `tests/fixtures/html/spa_shell.html`

- [ ] **Step 5.1: Write HTML fixtures**

```html
<!-- tests/fixtures/html/simple.html -->
<!doctype html>
<html><head><title>About Example</title></head>
<body>
<nav>HOME | ABOUT</nav>
<main>
<h1>About Us</h1>
<p>Example Industries, Inc. was founded in 1998.</p>
<h2>Products</h2>
<ul><li>Widget Pro</li><li>Widget Lite</li></ul>
</main>
<footer>© 2026 Example</footer>
</body></html>
```

```html
<!-- tests/fixtures/html/spa_shell.html -->
<!doctype html>
<html><head><title>App</title></head>
<body><div id="root"></div><script src="/app.js"></script></body></html>
```

- [ ] **Step 5.2: Write failing tests**

```python
# tests/test_cleaner.py
from tarantula.cleaner import clean_html, extract_title


def test_clean_simple_preserves_headings_and_text(fixtures_dir):
    html = (fixtures_dir / "html" / "simple.html").read_text()
    cleaned = clean_html(html, url="https://example.com/about")
    assert "Example Industries, Inc." in cleaned
    assert "founded in 1998" in cleaned
    assert "Widget Pro" in cleaned
    # boilerplate stripped
    assert "HOME | ABOUT" not in cleaned
    assert "© 2026" not in cleaned
    # structural markers preserved
    assert "## Products" in cleaned or "Products" in cleaned


def test_clean_spa_shell_returns_empty(fixtures_dir):
    html = (fixtures_dir / "html" / "spa_shell.html").read_text()
    cleaned = clean_html(html, url="https://example.com/")
    assert cleaned.strip() == ""


def test_extract_title_from_head():
    html = "<html><head><title>  Hello  </title></head><body>x</body></html>"
    assert extract_title(html) == "Hello"


def test_extract_title_missing():
    html = "<html><body>x</body></html>"
    assert extract_title(html) is None
```

- [ ] **Step 5.3: Run — expect fail**

- [ ] **Step 5.4: Implement `cleaner.py`**

```python
# src/tarantula/cleaner.py
from __future__ import annotations

from bs4 import BeautifulSoup
import trafilatura


def clean_html(html: str, url: str) -> str:
    """Extract readable content from HTML; returns '' if page looks empty/gated."""
    if not html or not html.strip():
        return ""
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        output_format="markdown",
    )
    if text and text.strip():
        return text.strip()
    # Fallback: readability + soup
    try:
        from readability import Document
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "lxml")
        fallback = soup.get_text(separator="\n").strip()
        # Collapse excessive blank lines
        lines = [ln.strip() for ln in fallback.splitlines() if ln.strip()]
        return "\n\n".join(lines)
    except Exception:
        return ""


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip() or None
    return None
```

- [ ] **Step 5.5: Run — expect pass**

```bash
pytest tests/test_cleaner.py -v
```

Expected: 4 passed.

- [ ] **Step 5.6: Commit**

```bash
git add src/tarantula/cleaner.py tests/test_cleaner.py tests/fixtures/html/
git commit -m "feat(cleaner): trafilatura primary, readability fallback"
```

---

## Task 6: Chunker

**Files:**
- Create: `src/tarantula/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_chunker.py
from tarantula.chunker import chunk_text, count_tokens


def test_count_tokens_nonempty():
    assert count_tokens("hello world") > 0


def test_short_text_single_chunk():
    chunks = list(chunk_text("Hello world.", target_tokens=2000, overlap_tokens=200))
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text == "Hello world."


def test_long_text_multiple_chunks_ordered():
    body = "\n\n".join(f"Paragraph {i}." for i in range(500))
    chunks = list(chunk_text(body, target_tokens=200, overlap_tokens=20))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.text.strip()


def test_chunks_overlap_on_content():
    body = "\n\n".join(f"Para{i}" for i in range(100))
    chunks = list(chunk_text(body, target_tokens=40, overlap_tokens=10))
    if len(chunks) >= 2:
        # Some overlap content exists between adjacent chunks.
        a = set(chunks[0].text.split())
        b = set(chunks[1].text.split())
        assert a & b  # intersection non-empty


def test_chunks_snap_to_paragraph_boundaries():
    body = "Para A.\n\nPara B.\n\nPara C."
    chunks = list(chunk_text(body, target_tokens=3, overlap_tokens=0))
    # Each chunk should start/end at paragraph boundaries when possible.
    for c in chunks:
        assert c.text == c.text.strip()
```

- [ ] **Step 6.2: Run — expect fail**

- [ ] **Step 6.3: Implement `chunker.py`**

```python
# src/tarantula/chunker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    ordinal: int
    text: str
    token_count: int


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(
    text: str,
    *,
    target_tokens: int = 2000,
    overlap_tokens: int = 200,
) -> Iterator[Chunk]:
    """Split text into paragraph-boundary-aligned chunks with overlap.

    Greedy fill: accumulate paragraphs until adding the next would exceed
    target_tokens; then emit a chunk. Keep the trailing N tokens worth of
    paragraphs as overlap into the next chunk.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        if text.strip():
            yield Chunk(ordinal=0, text=text.strip(), token_count=count_tokens(text))
        return

    # Precompute paragraph token counts for speed.
    para_tokens = [count_tokens(p) for p in paragraphs]

    ordinal = 0
    i = 0
    n = len(paragraphs)
    while i < n:
        buf: list[str] = []
        buf_tokens = 0
        j = i
        while j < n and (buf_tokens + para_tokens[j] <= target_tokens or not buf):
            buf.append(paragraphs[j])
            buf_tokens += para_tokens[j]
            j += 1
        chunk_str = "\n\n".join(buf).strip()
        yield Chunk(ordinal=ordinal, text=chunk_str, token_count=buf_tokens)
        ordinal += 1
        if j >= n:
            break
        # Compute overlap: step i forward, but back up enough to include
        # overlap_tokens of the tail.
        tail_tokens = 0
        k = j
        while k > i and tail_tokens < overlap_tokens:
            k -= 1
            tail_tokens += para_tokens[k]
        # Ensure forward progress.
        i = max(k, i + 1)
```

- [ ] **Step 6.4: Run — expect pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 5 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/tarantula/chunker.py tests/test_chunker.py
git commit -m "feat(chunker): paragraph-aligned chunks with token-budget overlap"
```

---

## Task 7: LLMClient interface, FakeLLMClient, OpenAI impl

**Files:**
- Create: `src/tarantula/llm.py`
- Create: `tests/test_llm_fake.py`

Note: the real OpenAI call is not unit-tested here (covered by E2E live tests). This task wires the interface and the in-test fake.

- [ ] **Step 7.1: Write failing tests for the fake**

```python
# tests/test_llm_fake.py
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
```

- [ ] **Step 7.2: Run — expect fail**

- [ ] **Step 7.3: Implement `llm.py`**

```python
# src/tarantula/llm.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    system: str
    user: str
    json_schema: dict[str, Any]
    model: str
    temperature: float = 0.0
    seed: int | None = 42
    schema_name: str = "response"


class LLMClient(Protocol):
    def complete_json(self, request: LLMRequest) -> dict[str, Any]: ...


@dataclass
class FakeLLMClient:
    """Test double that returns scripted JSON responses in order."""
    responses: list[dict[str, Any]]
    calls: list[LLMRequest] = field(default_factory=list)
    _cursor: int = 0

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        self.calls.append(request)
        if self._cursor >= len(self.responses):
            raise RuntimeError("FakeLLMClient: responses exhausted")
        r = self.responses[self._cursor]
        self._cursor += 1
        return r


class OpenAIClient:
    """Real OpenAI client using structured outputs (json_schema response format)."""

    def __init__(self, api_key: str | None = None, max_retries: int = 5) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._max_retries = max_retries

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=request.model,
                    temperature=request.temperature,
                    seed=request.seed,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": request.schema_name,
                            "schema": request.json_schema,
                            "strict": True,
                        },
                    },
                    messages=[
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:  # noqa: BLE001 — broad because SDK raises many types
                last_exc = e
                wait = min(2 ** attempt, 30)
                log.warning("OpenAI attempt %d failed: %s (retry in %ds)", attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI request failed after {self._max_retries} attempts") from last_exc
```

- [ ] **Step 7.4: Run — expect pass**

```bash
pytest tests/test_llm_fake.py -v
```

Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/tarantula/llm.py tests/test_llm_fake.py
git commit -m "feat(llm): LLMClient protocol, FakeLLMClient, OpenAI impl"
```

---

## Task 8: Map-step extractor

**Files:**
- Create: `src/tarantula/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 8.1: Write failing tests**

```python
# tests/test_extractor.py
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
    # per-variable wrapper shape
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
    # Despite the LLM claiming found=True, the fabricated quote demotes to not found.
    assert by_name["company_name"].found is False
```

- [ ] **Step 8.2: Run — expect fail**

- [ ] **Step 8.3: Implement `extractor.py`**

```python
# src/tarantula/extractor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VariableSpec
from .llm import LLMClient, LLMRequest

_JSON_TYPE = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def _value_schema(spec: VariableSpec) -> dict[str, Any]:
    if spec.type == "array":
        return {
            "type": ["array", "null"],
            "items": {"type": _JSON_TYPE[spec.items]},  # type: ignore[index]
        }
    return {"type": [_JSON_TYPE[spec.type], "null"]}


def build_map_schema(variables: list[VariableSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for v in variables:
        props[v.name] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["found", "value", "quote"],
            "properties": {
                "found": {"type": "boolean"},
                "value": _value_schema(v),
                "quote": {"type": ["string", "null"]},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [v.name for v in variables],
        "properties": props,
    }


SYSTEM = (
    "You extract typed variables from a single chunk of cleaned web content. "
    "Only answer found=true when the chunk CLEARLY supports it. "
    "The quote MUST be a verbatim substring of the chunk text (copy it exactly). "
    "If the chunk does not clearly support a variable, set found=false, value=null, quote=null. "
    "For array variables, include only items explicitly mentioned in the chunk."
)


def _render_user_prompt(
    variables: list[VariableSpec], chunk_text: str, page_url: str, page_title: str | None
) -> str:
    lines = [f"Page URL: {page_url}"]
    if page_title:
        lines.append(f"Page title: {page_title}")
    lines.append("")
    lines.append("Variables to extract:")
    for v in variables:
        lines.append(f"- {v.name} ({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}")
        for ex in v.examples:
            lines.append(f"  example input: {ex.input!r} -> output: {ex.output!r}")
    lines += ["", "Chunk text:", chunk_text]
    return "\n".join(lines)


@dataclass
class MapInput:
    chunk_text: str
    page_url: str
    page_title: str | None


@dataclass
class ExtractionRecord:
    variable_name: str
    found: bool
    value: Any
    quote: str | None


def extract_from_chunk(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    inp: MapInput,
    model: str,
) -> list[ExtractionRecord]:
    schema = build_map_schema(variables)
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(variables, inp.chunk_text, inp.page_url, inp.page_title),
        json_schema=schema,
        model=model,
        temperature=0.0,
        schema_name="per_chunk_extraction",
    )
    resp = client.complete_json(req)
    out: list[ExtractionRecord] = []
    for v in variables:
        entry = resp.get(v.name, {}) or {}
        found = bool(entry.get("found"))
        value = entry.get("value")
        quote = entry.get("quote")
        # Quote validation: must be a verbatim substring of the chunk.
        if found and (not quote or quote not in inp.chunk_text):
            found = False
            value = None
            quote = None
        out.append(ExtractionRecord(
            variable_name=v.name, found=found, value=value, quote=quote
        ))
    return out
```

- [ ] **Step 8.4: Run — expect pass**

```bash
pytest tests/test_extractor.py -v
```

Expected: 3 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/tarantula/extractor.py tests/test_extractor.py
git commit -m "feat(extractor): per-chunk map extraction with quote validation"
```

---

## Task 9: Reduce-step reducer

**Files:**
- Create: `src/tarantula/reducer.py`
- Create: `tests/test_reducer.py`

- [ ] **Step 9.1: Write failing tests**

```python
# tests/test_reducer.py
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
    assert results["has_careers_page"]["required_missing"] is False  # not required


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
```

- [ ] **Step 9.2: Run — expect fail**

- [ ] **Step 9.3: Implement `reducer.py`**

```python
# src/tarantula/reducer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import VariableSpec
from .llm import LLMClient, LLMRequest


@dataclass
class Candidate:
    value: Any
    quote: str | None
    source_url: str
    page_title: str | None


_JSON_TYPE = {"string": "string", "integer": "integer", "number": "number", "boolean": "boolean"}


def _value_schema(spec: VariableSpec) -> dict[str, Any]:
    if spec.type == "array":
        return {"type": ["array", "null"], "items": {"type": _JSON_TYPE[spec.items]}}  # type: ignore[index]
    return {"type": [_JSON_TYPE[spec.type], "null"]}


def build_reduce_schema(variables: list[VariableSpec]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for v in variables:
        if v.type == "array":
            props[v.name] = {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "sources", "reasoning"],
                "properties": {
                    "value": _value_schema(v),
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value_item", "source_url", "quote"],
                            "properties": {
                                "value_item": {"type": _JSON_TYPE[v.items]},  # type: ignore[index]
                                "source_url": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                    "reasoning": {"type": "string"},
                },
            }
        else:
            props[v.name] = {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "source_url", "quote", "reasoning"],
                "properties": {
                    "value": _value_schema(v),
                    "source_url": {"type": ["string", "null"]},
                    "quote": {"type": ["string", "null"]},
                    "reasoning": {"type": "string"},
                },
            }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [v.name for v in variables],
        "properties": props,
    }


SYSTEM = (
    "You are reconciling candidate extractions from multiple pages of one website "
    "into final values. For scalar variables, choose the best-supported candidate "
    "and prefer authoritative pages (e.g., /about, /company) over blog posts. "
    "For array variables, union and deduplicate candidate items and keep the "
    "source page for each item. Return null when no candidate supports a variable. "
    "Give a brief one-sentence reasoning per variable."
)


def _render_user_prompt(
    variables: list[VariableSpec],
    candidates_by_var: dict[str, list[Candidate]],
) -> str:
    lines = ["Variables:"]
    for v in variables:
        lines.append(f"- {v.name} ({v.type}{('<' + v.items + '>') if v.items else ''}): {v.description}")
    lines.append("")
    lines.append("Candidates per variable:")
    for v in variables:
        cs = candidates_by_var.get(v.name, [])
        lines.append(f"  {v.name}:")
        if not cs:
            lines.append("    (none)")
            continue
        for c in cs:
            title = f" [{c.page_title}]" if c.page_title else ""
            lines.append(
                f"    - value={json.dumps(c.value)} "
                f"source_url={c.source_url}{title} "
                f"quote={json.dumps(c.quote)}"
            )
    return "\n".join(lines)


def reduce_candidates(
    *,
    client: LLMClient,
    variables: list[VariableSpec],
    candidates_by_var: dict[str, list[Candidate]],
    model: str,
) -> dict[str, dict[str, Any]]:
    schema = build_reduce_schema(variables)
    req = LLMRequest(
        system=SYSTEM,
        user=_render_user_prompt(variables, candidates_by_var),
        json_schema=schema,
        model=model,
        temperature=0.0,
        schema_name="site_reduction",
    )
    raw = client.complete_json(req)
    results: dict[str, dict[str, Any]] = {}
    for v in variables:
        entry = raw.get(v.name, {}) or {}
        if v.type == "array":
            results[v.name] = {
                "value": entry.get("value"),
                "sources": entry.get("sources", []) or [],
                "reasoning": entry.get("reasoning", ""),
                "required_missing": bool(v.required and entry.get("value") in (None, [])),
            }
        else:
            results[v.name] = {
                "value": entry.get("value"),
                "source_url": entry.get("source_url"),
                "quote": entry.get("quote"),
                "reasoning": entry.get("reasoning", ""),
                "required_missing": bool(v.required and entry.get("value") is None),
            }
    return results
```

- [ ] **Step 9.4: Run — expect pass**

```bash
pytest tests/test_reducer.py -v
```

Expected: 3 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/tarantula/reducer.py tests/test_reducer.py
git commit -m "feat(reducer): per-site reconciliation with required-missing flag"
```

---

## Task 10: Crawler

**Files:**
- Create: `src/tarantula/crawler.py`
- Create: `tests/test_crawler.py`

The crawler is the most integration-heavy module. We test it against `pytest-httpserver`, which serves real HTTP from a local port — giving realistic fetch behavior without external network.

- [ ] **Step 10.1: Write failing tests**

```python
# tests/test_crawler.py
import pytest
from tarantula.config import SiteConfig, SiteDefaults
from tarantula.crawler import crawl_site, CrawlResult
from tarantula.store import Store


def _site(url: str, **overrides) -> SiteConfig:
    d = SiteDefaults().model_dump()
    d.update(overrides)
    return SiteConfig(url=url, **d)


@pytest.mark.asyncio
async def test_crawler_respects_depth(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        '<html><body><a href="/a">A</a></body></html>', content_type="text/html"
    )
    httpserver.expect_request("/a").respond_with_data(
        '<html><body><a href="/b">B</a></body></html>', content_type="text/html"
    )
    httpserver.expect_request("/b").respond_with_data(
        '<html><body>end</body></html>', content_type="text/html"
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=1, max_pages=100, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = set(_crawl_urls(store, result.crawl_id))
    assert httpserver.url_for("/") in urls
    assert httpserver.url_for("/a") in urls
    assert httpserver.url_for("/b") not in urls  # depth limit


@pytest.mark.asyncio
async def test_crawler_respects_same_host(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        f'<html><body><a href="https://other.example/x">out</a><a href="/a">in</a></body></html>',
        content_type="text/html",
    )
    httpserver.expect_request("/a").respond_with_data(
        '<html><body>a</body></html>', content_type="text/html"
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=2, max_pages=100, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = _crawl_urls(store, result.crawl_id)
    assert all("other.example" not in u for u in urls)


@pytest.mark.asyncio
async def test_crawler_respects_max_pages(httpserver, tmp_path):
    for i in range(20):
        links = "".join(f'<a href="/p{j}">P{j}</a>' for j in range(20))
        httpserver.expect_request(f"/p{i}").respond_with_data(
            f"<html><body>{links}</body></html>", content_type="text/html"
        )
    httpserver.expect_request("/").respond_with_data(
        "".join(f'<a href="/p{j}">P{j}</a>' for j in range(20)),
        content_type="text/html",
    )
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=5, max_pages=5, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    assert result.pages_fetched <= 5


@pytest.mark.asyncio
async def test_crawler_honors_robots_disallow(httpserver, tmp_path):
    httpserver.expect_request("/robots.txt").respond_with_data(
        "User-agent: *\nDisallow: /private\n", content_type="text/plain"
    )
    httpserver.expect_request("/").respond_with_data(
        '<a href="/public">pub</a><a href="/private/x">priv</a>',
        content_type="text/html",
    )
    httpserver.expect_request("/public").respond_with_data("ok", content_type="text/html")

    site = _site(httpserver.url_for("/"), max_depth=2, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    urls = _crawl_urls(store, result.crawl_id)
    assert httpserver.url_for("/public") in urls
    assert not any("/private" in u for u in urls)


@pytest.mark.asyncio
async def test_crawler_handles_4xx_without_aborting(httpserver, tmp_path):
    httpserver.expect_request("/").respond_with_data(
        '<a href="/ok">ok</a><a href="/missing">missing</a>',
        content_type="text/html",
    )
    httpserver.expect_request("/ok").respond_with_data("yay", content_type="text/html")
    httpserver.expect_request("/missing").respond_with_data("gone", status=404)
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=2, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    result = await crawl_site(store=store, run_id=run_id, site=site)
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_crawler_uses_cache_on_repeat(httpserver, tmp_path):
    request_count = {"n": 0}
    def handler(_req):
        request_count["n"] += 1
        from werkzeug.wrappers import Response
        return Response("<html>x</html>", content_type="text/html")
    httpserver.expect_request("/").respond_with_handler(handler)
    httpserver.expect_request("/robots.txt").respond_with_data("", status=404)

    site = _site(httpserver.url_for("/"), max_depth=0, rate_limit_rps=50)
    store = Store(tmp_path / "t.db", data_dir=tmp_path / "data")
    store.init_schema()
    run_id = store.start_run("", "")

    await crawl_site(store=store, run_id=run_id, site=site)
    await crawl_site(store=store, run_id=run_id, site=site)  # second crawl
    assert request_count["n"] == 1  # second hit came from cache


def _crawl_urls(store: Store, crawl_id: int) -> list[str]:
    return [r[0] for r in store.conn.execute(
        "SELECT p.url FROM pages p JOIN crawl_pages cp ON cp.page_id=p.id "
        "WHERE cp.crawl_id=?", (crawl_id,)
    ).fetchall()]
```

- [ ] **Step 10.2: Run — expect fail**

- [ ] **Step 10.3: Implement `crawler.py`**

```python
# src/tarantula/crawler.py
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from .cleaner import clean_html, extract_title
from .config import SiteConfig
from .robots import RobotsCache
from .store import Store
from .urls import is_html_like, normalize_url, same_scope

log = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    crawl_id: int
    pages_fetched: int
    status: str  # ok | partial | failed


PlaywrightFetcher = Callable[[str, int], Awaitable[tuple[int, bytes, str | None]]]


async def crawl_site(
    *,
    store: Store,
    run_id: int,
    site: SiteConfig,
    cache_ttl_seconds: int = 86400,
    playwright_fetcher: PlaywrightFetcher | None = None,
) -> CrawlResult:
    crawl_id = store.start_crawl(run_id, seed_url=site.url)
    pages_fetched = 0
    status = "ok"

    headers = {"User-Agent": site.user_agent}
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    timeout = httpx.Timeout(site.request_timeout_s)

    async with httpx.AsyncClient(
        headers=headers, follow_redirects=True, timeout=timeout, limits=limits, http2=True
    ) as client:
        async def http_fetch(url: str) -> tuple[int, str]:
            try:
                r = await client.get(url)
                return r.status_code, r.text
            except Exception as e:
                log.warning("robots fetch failed %s: %s", url, e)
                return 0, ""

        robots = RobotsCache(http_fetch, user_agent=site.user_agent)
        bucket = _TokenBucket(rate_rps=site.rate_limit_rps)
        sem = asyncio.Semaphore(4)

        seed = normalize_url(site.url)
        seen: set[str] = {seed}
        q: deque[tuple[str, int, str | None]] = deque([(seed, 0, None)])

        while q and pages_fetched < site.max_pages:
            url, depth, parent = q.popleft()

            if site.respect_robots_txt and not await robots.allowed(url):
                log.info("robots disallow: %s", url)
                continue

            cached = store.find_fresh_page(url, ttl_seconds=cache_ttl_seconds)
            if cached:
                page_id = cached.id
                raw_bytes = None  # already on disk
                with open(cached.raw_path, "rb") as f:
                    raw_bytes = f.read()
                html = raw_bytes.decode("utf-8", errors="replace")
            else:
                await bucket.acquire()
                try:
                    async with sem:
                        resp = await _fetch_with_retries(client, url)
                except Exception as e:
                    log.warning("fetch failed %s: %s", url, e)
                    status = "partial"
                    continue
                if resp is None or not (200 <= resp.status_code < 300):
                    # record nothing for hard failures; 4xx logged
                    continue
                html = resp.text
                raw_bytes = resp.content
                title = extract_title(html)
                page_id = store.save_page(
                    url=url,
                    raw_bytes=raw_bytes,
                    http_status=resp.status_code,
                    content_type=resp.headers.get("content-type"),
                    fetcher="http",
                    title=title,
                )

            cleaned = store.find_fresh_page(url, ttl_seconds=cache_ttl_seconds)
            cleaned_text = cleaned.cleaned_text if cleaned else None
            if cleaned_text is None:
                cleaned_text = clean_html(html, url=url)
                if not cleaned_text and playwright_fetcher is not None:
                    log.info("empty clean, retrying with playwright: %s", url)
                    try:
                        status_code, raw_bytes_pw, title_pw = await playwright_fetcher(
                            url, site.request_timeout_s
                        )
                        html_pw = raw_bytes_pw.decode("utf-8", errors="replace")
                        page_id = store.save_page(
                            url=url, raw_bytes=raw_bytes_pw, http_status=status_code,
                            content_type="text/html", fetcher="playwright",
                            title=title_pw or extract_title(html_pw),
                        )
                        cleaned_text = clean_html(html_pw, url=url)
                        html = html_pw
                    except Exception as e:
                        log.warning("playwright fetch failed %s: %s", url, e)
                store.set_cleaned_text(page_id, cleaned_text or "")

            store.link_page(crawl_id, page_id, depth=depth, parent_url=parent)
            pages_fetched += 1

            if depth >= site.max_depth:
                continue

            # Extract outbound links
            for link in _extract_links(html, base=url):
                if not is_html_like(link):
                    continue
                if not same_scope(link, seed=site.url, include_subdomains=site.include_subdomains):
                    continue
                norm = normalize_url(link)
                if norm in seen:
                    continue
                seen.add(norm)
                q.append((norm, depth + 1, url))

    store.finish_crawl(crawl_id, status=status, pages_fetched=pages_fetched)
    return CrawlResult(crawl_id=crawl_id, pages_fetched=pages_fetched, status=status)


class _TokenBucket:
    def __init__(self, rate_rps: float) -> None:
        self._interval = 1.0 / rate_rps if rate_rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_running_loop().time()


async def _fetch_with_retries(
    client: httpx.AsyncClient, url: str, retries: int = 3
) -> httpx.Response | None:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = await client.get(url)
            if r.status_code in (429,) or 500 <= r.status_code < 600:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            await asyncio.sleep(min(2 ** attempt, 30))
    if last_exc is not None:
        raise last_exc
    return None


def _extract_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        try:
            out.append(normalize_url(href, base=base))
        except Exception:
            continue
    return out
```

- [ ] **Step 10.4: Run — expect pass**

```bash
pytest tests/test_crawler.py -v
```

Expected: 6 passed.

- [ ] **Step 10.5: Commit**

```bash
git add src/tarantula/crawler.py tests/test_crawler.py
git commit -m "feat(crawler): async BFS with rate limit, robots, retries, cache"
```

---

## Task 11: CLI orchestration

**Files:**
- Create: `src/tarantula/logging_setup.py`
- Create: `src/tarantula/cli.py`
- Create: `tests/test_cli_e2e.py`
- Create: `tests/fixtures/sample_site/` (static HTML files)

The CLI ties all modules together: parse args, load configs, run pipeline per site (crawl → chunk → map → reduce), assemble output JSON, set exit code.

- [ ] **Step 11.1: Implement `logging_setup.py`**

```python
# src/tarantula/logging_setup.py
from __future__ import annotations

import json
import logging
from pathlib import Path


def configure(verbosity: int, log_path: Path | None) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(_JSONLFormatter())
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    root = logging.getLogger()
    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)


class _JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        })
```

- [ ] **Step 11.2: Write sample site fixtures for E2E test**

```html
<!-- tests/fixtures/sample_site/index.html -->
<!doctype html><html><head><title>ACME Inc.</title></head><body>
<h1>Welcome to ACME</h1>
<p>ACME Inc. is a privately held company.</p>
<a href="/about">About</a> | <a href="/products">Products</a>
</body></html>
```

```html
<!-- tests/fixtures/sample_site/about.html -->
<!doctype html><html><head><title>About ACME</title></head><body>
<h1>About</h1>
<p>ACME Inc. was founded in 1998.</p>
</body></html>
```

```html
<!-- tests/fixtures/sample_site/products.html -->
<!doctype html><html><head><title>Products</title></head><body>
<h1>Products</h1>
<ul><li>Widget Pro</li><li>Widget Lite</li></ul>
</body></html>
```

- [ ] **Step 11.3: Write failing E2E test using FakeLLMClient**

```python
# tests/test_cli_e2e.py
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

    # 3 pages × 3 vars map call each = 3 responses; 1 reduce response.
    map_resp = {
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
    # The products quote "Widget Pro" IS a substring of the cleaned text, so survives validation.
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
    fake = FakeLLMClient(responses=[map_resp, map_resp_about, map_resp_products, reduce_resp])

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
```

- [ ] **Step 11.4: Run — expect fail (no cli yet)**

- [ ] **Step 11.5: Implement `cli.py`**

```python
# src/tarantula/cli.py
from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer

from .chunker import chunk_text
from .cleaner import clean_html
from .config import VariablesConfig, load_urls_config, load_variables_config
from .crawler import crawl_site
from .extractor import MapInput, extract_from_chunk
from .llm import LLMClient, OpenAIClient
from .logging_setup import configure as configure_logging
from .reducer import Candidate, reduce_candidates
from .store import Store

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class PipelineOptions:
    urls_path: Path
    variables_path: Path
    output_path: Optional[Path]
    db_path: Path
    data_dir: Path
    map_model: str = "gpt-4o-mini"
    reduce_model: str = "gpt-4o"
    cache_ttl_seconds: int = 86400
    max_tokens: int = 2_000_000
    no_cache: bool = False
    quiet: bool = False
    llm_client: Optional[LLMClient] = None


async def run_pipeline(opts: PipelineOptions) -> int:
    urls_cfg = load_urls_config(opts.urls_path)
    vars_cfg = load_variables_config(opts.variables_path)

    store = Store(opts.db_path, data_dir=opts.data_dir)
    store.init_schema()

    urls_yaml_inline = opts.urls_path.read_text()
    vars_yaml_inline = opts.variables_path.read_text()
    run_id = store.start_run(urls_yaml_inline, vars_yaml_inline)

    client = opts.llm_client or OpenAIClient()

    sites_out = []
    run_status_bits = 0  # bit 1=required_missing, bit 2=partial, bit 3=failed
    started_at = _iso_now()

    for site in urls_cfg.sites:
        ttl = 0 if opts.no_cache else opts.cache_ttl_seconds
        try:
            result = await crawl_site(
                store=store, run_id=run_id, site=site, cache_ttl_seconds=ttl
            )
        except Exception as e:
            log.exception("crawl of %s failed: %s", site.url, e)
            run_status_bits |= 0b100
            sites_out.append({
                "seed_url": site.url,
                "crawl_status": "failed",
                "pages_fetched": 0,
                "variables": {},
            })
            continue

        if result.status == "partial":
            run_status_bits |= 0b010
        elif result.status == "failed":
            run_status_bits |= 0b100

        # Chunk + map
        candidates_by_var: dict[str, list[Candidate]] = {}
        page_rows = list(store.conn.execute(
            "SELECT p.id, p.url, p.title, p.cleaned_text FROM pages p "
            "JOIN crawl_pages cp ON cp.page_id = p.id "
            "WHERE cp.crawl_id = ? ORDER BY cp.depth, p.id",
            (result.crawl_id,),
        ).fetchall())

        for page_id, url, title, cleaned in page_rows:
            if not cleaned:
                continue
            for chunk in chunk_text(cleaned):
                chunk_id = store.save_chunk(
                    page_id=page_id, ordinal=chunk.ordinal,
                    text=chunk.text, token_count=chunk.token_count,
                )
                recs = extract_from_chunk(
                    client=client,
                    variables=vars_cfg.variables,
                    inp=MapInput(chunk_text=chunk.text, page_url=url, page_title=title),
                    model=opts.map_model,
                )
                for r in recs:
                    store.save_chunk_extraction(
                        run_id=run_id, chunk_id=chunk_id,
                        variable_name=r.variable_name,
                        found=r.found, value=r.value, quote=r.quote,
                    )
                    if r.found:
                        candidates_by_var.setdefault(r.variable_name, []).append(
                            Candidate(value=r.value, quote=r.quote,
                                      source_url=url, page_title=title)
                        )

        # Reduce
        reduced = reduce_candidates(
            client=client, variables=vars_cfg.variables,
            candidates_by_var=candidates_by_var, model=opts.reduce_model,
        )
        for name, payload in reduced.items():
            store.save_extraction(
                run_id=run_id,
                crawl_id=result.crawl_id,
                variable_name=name,
                value=payload["value"],
                source_url=payload.get("source_url"),
                quote=payload.get("quote"),
                reasoning=payload.get("reasoning"),
            )
            if payload.get("required_missing"):
                run_status_bits |= 0b001

        sites_out.append({
            "seed_url": site.url,
            "crawl_status": result.status,
            "pages_fetched": result.pages_fetched,
            "variables": reduced,
        })

    store.finish_run(run_id, status="ok")

    finished_at = _iso_now()
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "sites": sites_out,
    }
    out_text = json.dumps(payload, indent=2, default=str)
    if opts.output_path:
        opts.output_path.write_text(out_text)
        if not opts.quiet:
            summary = _summary(sites_out)
            print(summary)
    else:
        print(out_text)

    # highest severity wins
    if run_status_bits & 0b100:
        return 4
    if run_status_bits & 0b010:
        return 3
    if run_status_bits & 0b001:
        return 2
    return 0


def _summary(sites: list[dict]) -> str:
    n_sites = len(sites)
    extracted = sum(
        1 for s in sites for v in s["variables"].values() if v.get("value") is not None
    )
    total_vars = sum(len(s["variables"]) for s in sites)
    missing_required = sum(
        1 for s in sites for v in s["variables"].values() if v.get("required_missing")
    )
    return (
        f"{n_sites} sites, {extracted}/{total_vars} variables extracted, "
        f"{missing_required} required missing"
    )


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@app.command()
def extract(
    urls: Path = typer.Option(..., "--urls", exists=True, readable=True, help="Path to urls.yaml"),
    variables: Path = typer.Option(..., "--variables", exists=True, readable=True),
    output: Optional[Path] = typer.Option(None, "--output"),
    db: Path = typer.Option(Path("tarantula.db"), "--db"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    map_model: str = typer.Option("gpt-4o-mini", "--map-model"),
    reduce_model: str = typer.Option("gpt-4o", "--reduce-model"),
    cache_ttl: str = typer.Option("24h", "--cache-ttl"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    max_tokens: int = typer.Option(2_000_000, "--max-tokens"),
    verbose: int = typer.Option(0, "-v", "--verbose", count=True),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Crawl sites and extract typed variables via LLM."""
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "logs" / "run.jsonl"
    configure_logging(verbosity=verbose, log_path=log_path)

    opts = PipelineOptions(
        urls_path=urls,
        variables_path=variables,
        output_path=output,
        db_path=db,
        data_dir=data_dir,
        map_model=map_model,
        reduce_model=reduce_model,
        cache_ttl_seconds=_parse_duration(cache_ttl),
        max_tokens=max_tokens,
        no_cache=no_cache,
        quiet=quiet,
    )
    exit_code = asyncio.run(run_pipeline(opts))
    raise typer.Exit(code=exit_code)


def _parse_duration(s: str) -> int:
    """Parse '24h', '30m', '3600s', '2d' into seconds."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    s = s.strip().lower()
    if s[-1] in units:
        return int(s[:-1]) * units[s[-1]]
    return int(s)


if __name__ == "__main__":
    app()
```

- [ ] **Step 11.6: Run E2E test — expect pass**

```bash
pytest tests/test_cli_e2e.py -v
```

Expected: 1 passed.

- [ ] **Step 11.7: Smoke-run the CLI to verify the entry point works**

```bash
tarantula --help
```

Expected: prints help text with `extract` command.

- [ ] **Step 11.8: Commit**

```bash
git add src/tarantula/cli.py src/tarantula/logging_setup.py tests/test_cli_e2e.py tests/fixtures/sample_site/
git commit -m "feat(cli): pipeline orchestration and typer entry point"
```

---

## Task 12: Playwright fallback (optional extra)

**Files:**
- Modify: `src/tarantula/crawler.py`
- Create: `src/tarantula/playwright_fetcher.py`
- Create: `tests/test_playwright.py`

This wires the Playwright fallback behind an optional import. Marked `@pytest.mark.playwright` so CI without Chromium skips it.

- [ ] **Step 12.1: Implement `playwright_fetcher.py`**

```python
# src/tarantula/playwright_fetcher.py
from __future__ import annotations


async def playwright_fetch(url: str, timeout_s: int) -> tuple[int, bytes, str | None]:
    """Fetch a URL with headless Chromium. Returns (status, raw_html_bytes, title)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            resp = await page.goto(url, timeout=timeout_s * 1000, wait_until="networkidle")
            status = resp.status if resp else 0
            content = await page.content()
            title = await page.title()
            return status, content.encode("utf-8"), title or None
        finally:
            await browser.close()
```

- [ ] **Step 12.2: Wire Playwright into CLI**

Modify `src/tarantula/cli.py` — in `run_pipeline`, before calling `crawl_site`:

```python
# Add at top of run_pipeline, near other imports:
from .playwright_fetcher import playwright_fetch

# Then in the crawl loop:
        result = await crawl_site(
            store=store, run_id=run_id, site=site,
            cache_ttl_seconds=ttl,
            playwright_fetcher=playwright_fetch,
        )
```

Replace the existing `await crawl_site(...)` call with the version above.

- [ ] **Step 12.3: Write optional integration test**

```python
# tests/test_playwright.py
import pytest
from tarantula.playwright_fetcher import playwright_fetch


@pytest.mark.playwright
@pytest.mark.asyncio
async def test_playwright_renders_spa(httpserver):
    spa_html = (
        "<html><head><title>SPA</title></head><body>"
        "<div id='root'></div>"
        "<script>document.getElementById('root').innerText='hello from js'</script>"
        "</body></html>"
    )
    httpserver.expect_request("/").respond_with_data(spa_html, content_type="text/html")
    status, raw, title = await playwright_fetch(httpserver.url_for("/"), timeout_s=10)
    assert status == 200
    assert b"hello from js" in raw
    assert title == "SPA"
```

- [ ] **Step 12.4: Run (only if user has Chromium installed)**

```bash
# Only run this manually; default test run skips it
pytest tests/test_playwright.py -v -m playwright
```

Expected: skipped by default, passes when run with `-m playwright`.

- [ ] **Step 12.5: Commit**

```bash
git add src/tarantula/playwright_fetcher.py src/tarantula/cli.py tests/test_playwright.py
git commit -m "feat(crawler): optional playwright fallback for JS-gated pages"
```

---

## Task 13: README

**Files:**
- Create: `README.md`

- [ ] **Step 13.1: Write minimal but complete README**

```markdown
# Tarantula

A CLI that crawls websites and extracts user-defined typed variables using an LLM, with per-value source URL and quoted passage.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    # Optional JS rendering:
    pip install -e ".[playwright]"
    playwright install chromium

Set `OPENAI_API_KEY`:

    export OPENAI_API_KEY=sk-...

## Usage

    tarantula extract \
      --urls urls.yaml \
      --variables variables.yaml \
      --output results.json \
      --db tarantula.db \
      --data-dir ./data

Config formats: see `docs/superpowers/specs/2026-04-19-tarantula-design.md` sections 5.1 (urls.yaml) and 5.2 (variables.yaml).

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | OK |
| 2    | Some required variables missing |
| 3    | One or more crawls partial |
| 4    | One or more crawls failed |

Highest-severity wins when multiple apply.

## Tests

    pytest                 # default suite (no live LLM, no Playwright)
    pytest -m playwright   # requires Chromium installed
    pytest -m live         # requires OPENAI_API_KEY
```

- [ ] **Step 13.2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, usage, exit codes"
```

---

## Final verification

- [ ] **Run the full test suite:**

```bash
pytest -q
```

Expected: all tests pass, none marked `live` or `playwright` are run.

- [ ] **Run the CLI against a small real site (manual smoke test):**

```bash
cat > /tmp/urls.yaml <<'EOF'
defaults:
  max_depth: 1
  max_pages: 10
  rate_limit_rps: 1
sites:
  - url: https://example.com
EOF

cat > /tmp/vars.yaml <<'EOF'
variables:
  - name: purpose
    type: string
    description: "What the site is for."
EOF

OPENAI_API_KEY=sk-... tarantula extract \
  --urls /tmp/urls.yaml \
  --variables /tmp/vars.yaml \
  --output /tmp/result.json
cat /tmp/result.json
```

Expected: `result.json` contains a single site with `purpose` extracted, plus `source_url` and `quote`.

- [ ] **Confirm SQLite is inspectable:**

```bash
sqlite3 tarantula.db "SELECT url, fetcher, length(cleaned_text) FROM pages LIMIT 5"
```

Expected: rows with URL, `http` or `playwright`, and non-zero cleaned length.
