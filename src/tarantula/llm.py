from __future__ import annotations

import json
import logging
import threading
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
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


@dataclass
class FakeLLMClient:
    """Test double that returns scripted JSON responses and embeddings.

    complete_json modes (may be combined):
      - list: responses returned in order via `responses` (legacy).
      - keyed: if request.schema_name matches a key in `responses_by_schema`,
        that response is returned. Useful when calls run concurrently.

    embed modes:
      - keyed: if a text matches `embeddings_by_text`, that vector is returned.
      - fallback: deterministic 8-d vector derived from sha1(text).
    """
    responses: list[dict[str, Any]] = field(default_factory=list)
    responses_by_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    embeddings_by_text: dict[str, list[float]] = field(default_factory=dict)
    embed_dim: int = 8
    calls: list[LLMRequest] = field(default_factory=list)
    embed_calls: list[tuple[list[str], str]] = field(default_factory=list)
    _cursor: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        with self._lock:
            self.calls.append(request)
            if request.schema_name in self.responses_by_schema:
                return self.responses_by_schema[request.schema_name]
            if self._cursor >= len(self.responses):
                raise RuntimeError("FakeLLMClient: responses exhausted")
            r = self.responses[self._cursor]
            self._cursor += 1
            return r

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        import hashlib
        with self._lock:
            self.embed_calls.append((list(texts), model))
            out: list[list[float]] = []
            for t in texts:
                if t in self.embeddings_by_text:
                    out.append(list(self.embeddings_by_text[t]))
                    continue
                # Deterministic pseudo-embedding from sha1 bytes.
                h = hashlib.sha1(t.encode("utf-8")).digest()
                vec = [((h[i % len(h)] / 255.0) * 2.0) - 1.0 for i in range(self.embed_dim)]
                out.append(vec)
            return out


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
            except Exception as e:
                last_exc = e
                wait = min(2 ** attempt, 30)
                log.warning("OpenAI attempt %d failed: %s (retry in %ds)", attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI request failed after {self._max_retries} attempts") from last_exc

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.embeddings.create(model=model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_exc = e
                wait = min(2 ** attempt, 30)
                log.warning("OpenAI embed attempt %d failed: %s (retry in %ds)",
                            attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(
            f"OpenAI embed failed after {self._max_retries} attempts"
        ) from last_exc
