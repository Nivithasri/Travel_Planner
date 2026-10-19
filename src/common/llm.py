"""Ollama client. All LLM access in the project goes through this module.

- Base URL comes from OLLAMA_BASE_URL via config; models are always passed in by the caller.
- Every call goes through the disk cache. The cache key uses the model and the API path but NOT the
  host, so a rotating ngrok URL does not invalidate cached results. The seed and temperature are
  part of the options and therefore part of the key.
- Retries use exponential backoff on transport errors (any requests.RequestException except a
  malformed URL), 429, 5xx, and on HTTP 200 bodies that are not an Ollama reply (e.g. the ngrok
  browser-warning page) or are empty. Such bodies are never cached.
- Total time per call is bounded by `retry_budget_s` (default a few minutes), retries included.
- A 404 means the model or the endpoint is wrong; it is raised immediately with a clear message.
- TODO(Phase 7): streaming. Phase 0 only supports non-streamed calls.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ValidationError

from src.common.cache import DiskCache
from src.common.schemas import DEFAULT_LLM_SEED, DEFAULT_RETRY_BUDGET_S, load_settings

T = TypeVar("T", bound=BaseModel)

CHAT_PATH = "/api/chat"
TAGS_PATH = "/api/tags"
EMBED_PATH = "/api/embed"
DEFAULT_TIMEOUT = (10.0, 600.0)  # (connect, read); capped per attempt by the remaining budget


class LLMError(RuntimeError):
    pass


class ModelNotFoundError(LLMError):
    pass


class BadResponseError(LLMError):
    """HTTP 200 but not a usable Ollama reply (wrong structure, invalid JSON, empty)."""


class JSONValidationError(LLMError):
    pass


class _Retryable(Exception):
    pass


class _BadBody(_Retryable):
    pass


def _reply_text(body: Any) -> str:
    """Extract the reply from an Ollama body by structure, or raise _BadBody."""
    if not isinstance(body, dict):
        raise _BadBody("body is not a JSON object")
    message = body.get("message")
    if isinstance(message, dict) and "content" in message:
        content = message["content"]
    elif "response" in body:  # /api/generate shape
        content = body["response"]
    else:
        raise _BadBody("body has neither a 'message' nor a 'response' key")
    if not isinstance(content, str):
        raise _BadBody("reply text is not a string")
    if not content.strip():
        raise _BadBody("empty reply")
    return content


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        cache: DiskCache,
        *,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        retry_budget_s: float = DEFAULT_RETRY_BUDGET_S,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        max_json_attempts: int = 3,
        seed: int = DEFAULT_LLM_SEED,
        post: Callable[..., Any] | None = None,
        get: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout
        self.retry_budget_s = retry_budget_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_json_attempts = max_json_attempts
        self.seed = seed
        self._post = post or requests.post
        self._get = get or requests.get
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def from_settings(cls, **kwargs: Any) -> OllamaClient:
        s = load_settings()
        kwargs.setdefault("seed", s.llm_seed)
        kwargs.setdefault("retry_budget_s", s.llm_retry_budget_s)
        return cls(s.ollama_base_url, DiskCache(s.cache_dir), **kwargs)

    # ---- transport -------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if "ngrok" in self.base_url:
            headers["ngrok-skip-browser-warning"] = "1"
        return headers

    def _scrub(self, text: str) -> str:
        """Never let the endpoint (host or URL) reach a log line or an exception message."""
        host = urlsplit(self.base_url).netloc
        for secret in (self.base_url, host):
            if secret:
                text = text.replace(secret, "<OLLAMA_BASE_URL>")
        return text

    def _call(self, send: Callable[[], Any], what: str) -> Any:
        """Run one HTTP exchange and map failures to _Retryable / LLMError."""
        try:
            resp = send()
        except (
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema,
            requests.exceptions.InvalidURL,
        ) as exc:
            raise LLMError(
                f"OLLAMA_BASE_URL is not a valid URL ({type(exc).__name__}); check .env"
            ) from None  # the original message holds the URL
        except requests.RequestException as exc:  # connection, timeout, chunked encoding, ...
            raise _Retryable(
                f"transport error ({type(exc).__name__}): {self._scrub(str(exc))}"
            ) from None  # the original message holds the host

        status = resp.status_code
        if status == 404:
            raise ModelNotFoundError(
                f"model or endpoint not found: check that the model is pulled and the URL is "
                f"right ({what}; the URL is OLLAMA_BASE_URL in .env and is not printed)"
            )
        if status == 429 or status >= 500:
            raise _Retryable(f"HTTP {status}")
        if status != 200:
            raise LLMError(f"HTTP {status} from Ollama: {self._scrub(resp.text[:300])!r}")
        try:
            return resp.json()
        except ValueError as exc:
            snippet = self._scrub(resp.text[:120]).replace("\n", " ")
            raise _BadBody(f"HTTP 200 but body is not JSON ({snippet!r})") from exc

    def _with_budget(self, attempt_fn: Callable[[tuple[float, float]], Any]) -> Any:
        """Call attempt_fn(timeout) with retries, bounded by max_retries and the time budget."""
        start = self._clock()
        last: Exception | None = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            remaining = self.retry_budget_s - (self._clock() - start)
            if remaining <= 0:
                break
            connect, read = self.timeout
            attempts += 1
            try:
                return attempt_fn((min(connect, remaining), min(read, remaining)))
            except _Retryable as exc:
                last = exc
            delay = self.backoff_base * (2**attempt) * (0.5 + random.random() / 2)
            if attempt == self.max_retries or self._clock() - start + delay >= self.retry_budget_s:
                break
            self._sleep(delay)
        if isinstance(last, _BadBody):
            raise BadResponseError(f"{last} (after {attempts} attempts)") from last
        raise LLMError(
            f"Giving up after {attempts} attempts within {self.retry_budget_s:.0f}s: {last}"
        ) from last

    def _request(self, payload: dict) -> str:
        def attempt(timeout: tuple[float, float]) -> str:
            body = self._call(
                lambda: self._post(
                    self.base_url + CHAT_PATH,
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout,
                ),
                f"model {payload['model']!r}",
            )
            return _reply_text(body)

        return self._with_budget(attempt)

    def _options(self, options: dict | None) -> dict:
        return {"temperature": 0, "seed": self.seed, **(options or {})}

    def _payload(self, model: str, messages: list[dict], options: dict | None, fmt: Any) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": self._options(options),
        }
        if fmt is not None:
            payload["format"] = fmt
        return payload

    # ---- public API ------------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Names of the models the server has pulled (GET /api/tags). Not cached: it is a live
        health check."""

        def attempt(timeout: tuple[float, float]) -> list[str]:
            body = self._call(
                lambda: self._get(
                    self.base_url + TAGS_PATH, headers=self._headers(), timeout=timeout
                ),
                "listing models",
            )
            models = body.get("models") if isinstance(body, dict) else None
            if not isinstance(models, list):
                raise _BadBody("body has no 'models' list")
            return [m["name"] for m in models if isinstance(m, dict) and "name" in m]

        return self._with_budget(attempt)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embeddings from the server (POST /api/embed), one vector per text. Cached on
        (model, path, texts). Used only to check agreement with the sentence-transformers vectors
        (D51); corpus and query embeddings come from sentence-transformers."""
        inputs = {"input": list(texts), "path": EMBED_PATH}

        def fetch() -> list[list[float]]:
            def attempt(timeout: tuple[float, float]) -> list[list[float]]:
                body = self._call(
                    lambda: self._post(
                        self.base_url + EMBED_PATH,
                        json={"model": model, "input": list(texts)},
                        headers=self._headers(),
                        timeout=timeout,
                    ),
                    f"embedding model {model!r}",
                )
                vecs = body.get("embeddings") if isinstance(body, dict) else None
                ok = (
                    isinstance(vecs, list)
                    and len(vecs) == len(texts)
                    and all(
                        isinstance(v, list)
                        and v
                        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                        for v in vecs
                    )
                )
                if not ok:
                    raise _BadBody("body has no valid 'embeddings' list of the same length")
                return vecs

            return self._with_budget(attempt)

        return self.cache.get_or_compute("llm_embed", model, inputs, fetch)

    def chat(
        self,
        messages: list[dict],
        model: str,
        *,
        options: dict | None = None,
        format: Any = None,
    ) -> str:
        """Return the assistant text. Cached on (model, path, messages, options, format).
        Empty or malformed replies raise and are never cached."""
        payload = self._payload(model, messages, options, format)
        inputs = {
            "messages": messages,
            "options": payload["options"],
            "format": payload.get("format"),
            "path": CHAT_PATH,
        }
        return self.cache.get_or_compute("llm_chat", model, inputs, lambda: self._request(payload))

    def chat_json(
        self,
        messages: list[dict],
        model: str,
        schema: type[T],
        *,
        options: dict | None = None,
        on_attempt: Callable[[int, str, str | None], None] | None = None,
    ) -> T:
        """JSON mode with validation. Only validated output is cached; invalid output triggers a
        repair turn (up to max_json_attempts total) and is then discarded. `on_attempt(n, raw,
        error)` is called for every network attempt (n from 1; error None when valid) so a caller
        can keep the raw reply for debugging; a cache hit makes no attempt and no call."""
        payload = self._payload(model, messages, options, "json")
        inputs = {
            "messages": messages,
            "options": payload["options"],
            "schema": schema.model_json_schema(),
            "path": CHAT_PATH,
        }
        cached = self.cache.get("llm_json", model, inputs, default=None)
        if isinstance(cached, str):
            try:
                return schema.model_validate_json(cached)
            except ValidationError:
                pass  # stale/incompatible entry: treat as a miss, recompute and overwrite

        convo = list(messages)
        last_error = ""
        for attempt in range(1, self.max_json_attempts + 1):
            raw = self._request({**payload, "messages": convo})
            try:
                parsed = schema.model_validate_json(raw)
            except ValidationError as exc:
                last_error = str(exc)
                if on_attempt:
                    on_attempt(attempt, raw, last_error)
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "That reply was not valid for the required JSON schema. "
                        f"Errors: {last_error[:500]}. Reply with corrected JSON only.",
                    },
                ]
                continue
            if on_attempt:
                on_attempt(attempt, raw, None)
            self.cache.set("llm_json", model, inputs, parsed.model_dump_json())
            return parsed
        raise JSONValidationError(
            f"No valid JSON for {schema.__name__} after {self.max_json_attempts} attempts: "
            f"{last_error[:300]}"
        )
