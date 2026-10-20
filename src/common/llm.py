"""LLM clients. All LLM access in the project goes through this module: OllamaClient (local server,
legacy since D59) and OpenAICompatClient (hosted OpenAI-compatible API with a client-side rate
limiter, D59). The notes below describe the Ollama client; the hosted one is at the end of the file.


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

import hashlib
import json
import random
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


class AuthError(LLMError):
    """401/403: the key is wrong or lacks access. Never retried."""


class InvalidGeneration(LLMError):
    """The API itself rejected a JSON-mode reply as not valid JSON (HTTP 400, e.g. "Failed to
    generate JSON", with the text it produced). The reply is bad, the transport is fine: chat_json
    treats it like any invalid reply (repair turn, then give up), never as an outage."""

    def __init__(self, text: str):
        super().__init__("the API rejected the reply as invalid JSON")
        self.text = text


class QuotaExhausted(LLMError):
    """A daily budget is used up (or the server asked for a long wait). Not a transport failure,
    never a fallback: stop, and resume after the reset."""

    def __init__(self, reason: str, *, tokens_today: int = 0, requests_today: int = 0,
                 reset_in_s: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.tokens_today = tokens_today
        self.requests_today = requests_today
        self.reset_in_s = reset_in_s


class _Retryable(Exception):
    pass


class _RetryAfter(_Retryable):
    """429 with a Retry-After the client honours instead of its own backoff."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class _BadBody(_Retryable):
    pass


_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bgsk_[A-Za-z0-9]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:org|proj|key)_[A-Za-z0-9]{8,}"),  # account identifiers in API messages
    re.compile(r"(?i)(authorization|x-api-key|api-key)\s*[:=]\s*\S+"),
    re.compile(r"https?://[^\s'\"<>)\]]+"),
)


def scrub_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove keys, bearer tokens, auth headers, URLs and any given secret strings (the base URL,
    its host, the key) from text that may reach a log line, an exception or a debug record.
    Patterns run first, so a URL that contains a secret goes as a whole; then every secret, and
    the first and last 10 characters of a long one (error messages abbreviate long strings)."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("<redacted>", text)
    pieces = set()
    for secret in secrets:
        if secret:
            pieces.add(secret)
            if len(secret) >= 20:
                pieces.update({secret[:10], secret[-10:]})
    for piece in sorted(pieces, key=len, reverse=True):
        text = text.replace(piece, "<redacted>")
    return text


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
    chat_path = CHAT_PATH
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
            retry_after = getattr(last, "retry_after", None)
            if retry_after is not None:  # the server said how long to wait: honour it
                delay = float(retry_after)
            else:
                delay = self.backoff_base * (2**attempt) * (0.5 + random.random() / 2)
            server_wait = retry_after is not None  # D66: not part of the retry budget
            if attempt == self.max_retries or (
                    not server_wait and self._clock() - start + delay >= self.retry_budget_s):
                break
            self._sleep(delay)
            if server_wait:
                start += delay
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

    def _key_options(self, payload: dict) -> dict:
        """The part of the request that belongs in the cache key besides model and messages."""
        return payload["options"]

    def scrub(self, text: str) -> str:
        """Public name for _scrub: callers that write debug records use it."""
        return self._scrub(text)

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
            "options": self._key_options(payload),
            "format": payload.get("format"),
            "path": self.chat_path,
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
            "options": self._key_options(payload),
            "schema": schema.model_json_schema(),
            "path": self.chat_path,
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
            try:
                raw = self._request({**payload, "messages": convo})
            except InvalidGeneration as bad:
                raw = self._scrub(bad.text)[:2000] or "(the API returned no text)"
                last_error = "the API rejected the reply as invalid JSON (JSON mode)"
                if on_attempt:
                    on_attempt(attempt, raw, last_error)
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That reply was not valid JSON. Reply with "
                     "one complete, valid JSON object only."},
                ]
                continue
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


# ==== OpenAI-compatible hosted API (D59) ==========================================================
# Chat completions on a hosted free tier (Groq). Same cache, retry budget and JSON repair loop as
# the Ollama client above; adds a client-side rate limiter, usage accounting and scrubbing of
# keys, headers and URLs. Sequential only. Reasoning text is never read, cached or stored.

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
MAX_COMPLETION_TOKENS_CAP = 400  # D59 set 300; D64 raised it: 5% of replies were cut off at 300
REASONING_EFFORTS = ("low", "medium", "high")
MAX_RETRY_AFTER_S = 290.0  # a longer Retry-After is a quota stop, not a wait (D66: was 120)
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: Any) -> float | None:
    """Seconds from "30", "2m59.56s", "1h2m", "6ms" ... or None when it is not a duration."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        pass
    parts = _DURATION.findall(t)
    if not parts or "".join(a + b for a, b in parts) != t:
        return None
    return sum(float(a) * _UNIT_S[b] for a, b in parts)


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RateInfo:
    """Rate-limit headers of one response. As documented for the target API: the *-requests
    headers describe the DAILY request limit (RPD), the *-tokens headers the PER-MINUTE token
    limit (TPM). Retry-After is read separately and only on a 429."""

    rpd_limit: int | None = None
    rpd_remaining: int | None = None
    rpd_reset_s: float | None = None
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    tpm_reset_s: float | None = None


def _lower(headers: Any) -> dict[str, str]:
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
    except (TypeError, ValueError):
        return {}


def parse_rate_headers(headers: Any) -> RateInfo:
    h = _lower(headers)
    return RateInfo(
        rpd_limit=_to_int(h.get("x-ratelimit-limit-requests")),
        rpd_remaining=_to_int(h.get("x-ratelimit-remaining-requests")),
        rpd_reset_s=parse_duration(h.get("x-ratelimit-reset-requests")),
        tpm_limit=_to_int(h.get("x-ratelimit-limit-tokens")),
        tpm_remaining=_to_int(h.get("x-ratelimit-remaining-tokens")),
        tpm_reset_s=parse_duration(h.get("x-ratelimit-reset-tokens")),
    )


def retry_after_seconds(headers: Any) -> float | None:
    return parse_duration(_lower(headers).get("retry-after"))


class RateLimiter:
    """Client-side pacing for RPM, TPM, RPD and TPD. Never relies on 429s.

    Effective limits: RPM, TPM and RPD with a 20% safety margin; the daily token budget is 85% of
    TPD. Per-minute usage is a 60 s sliding window; the daily counters are persisted in
    `state_path` (UTC date, tokens, requests) so a run on the next day resumes with a fresh budget
    and a restart on the same day does not forget what was spent. A call is reserved with an
    ESTIMATE before it is sent and corrected with the `usage` the API reports afterwards."""

    def __init__(self, rpm: int, tpm: int, rpd: int, tpd: int, *, margin: float = 0.20,
                 tpd_share: float = 0.85, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 state_path: Path | str | None = None, key_id: str = "default",
                 trust_provider: bool = False):
        if min(rpm, tpm, rpd, tpd) <= 0:
            raise ValueError("RPM, TPM, RPD and TPD must be positive")
        self.limits = {"rpm": rpm, "tpm": tpm, "rpd": rpd, "tpd": tpd}
        self.rpm_eff = max(1, int(rpm * (1 - margin)))
        self.tpm_eff = tpm * (1 - margin)
        self.rpd_eff = max(1, int(rpd * (1 - margin)))
        self.tpd_eff = tpd * tpd_share
        self._clock, self._sleep, self._utcnow = clock, sleep, utcnow
        self.state_path = Path(state_path) if state_path else None
        self.trust_provider = trust_provider  # skip the local daily checks; the API's 429 decides
        self.key_id = key_id  # which API key's spend this limiter counts (never the key itself)
        self.window: deque[list] = deque()  # [time, tokens] per request in the last 60 s
        self.slept_s = 0.0
        self.n429 = 0
        self.rate_info: RateInfo | None = None
        self.day = self._today()
        self.day_tokens = 0
        self.day_requests = 0
        self._rpd_blocked_for_s: float | None = None
        self._tpm_hint: tuple[int, float] | None = None
        self._load()

    # ---- daily state -----------------------------------------------------------------------
    def _today(self) -> str:
        return self._utcnow().strftime("%Y-%m-%d")

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            st = self._read_state()
            if st.get("date") == self.day:
                mine = st["keys"].get(self.key_id)
                if mine:
                    self.day_tokens, self.day_requests = int(mine["tokens"]), int(mine["requests"])
        except (ValueError, KeyError, TypeError, OSError):
            pass  # unreadable state: start the day at zero rather than refuse to run

    def _read_state(self) -> dict:
        """{"date", "keys": {key_id: {"tokens", "requests"}}}. The first format had one flat
        count for whichever key was in use; it is kept under the id "legacy"."""
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        if "keys" not in st:
            st = {"date": st.get("date"), "keys": {"legacy": {
                "tokens": st.get("tokens", 0), "requests": st.get("requests", 0)}}}
        return st

    def _save(self) -> None:
        if self.state_path:
            from src.common.cache import atomic_write_text

            keys: dict = {}
            try:
                if self.state_path.exists():
                    old = self._read_state()
                    if old.get("date") == self.day:
                        keys = old["keys"]  # other keys' spend of today is kept
            except (ValueError, KeyError, TypeError, OSError):
                pass
            keys[self.key_id] = {"tokens": self.day_tokens, "requests": self.day_requests}
            atomic_write_text(self.state_path, json.dumps({"date": self.day, "keys": keys}))

    def _roll_day(self) -> None:
        today = self._today()
        if today != self.day:
            self.day, self.day_tokens, self.day_requests = today, 0, 0
            self._rpd_blocked_for_s = None
            self._save()

    def seconds_to_utc_midnight(self) -> float:
        now = self._utcnow()
        nxt = (now.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp() + 86400
        return max(0.0, nxt - now.timestamp())

    def _exhausted(self, reason: str, reset_in_s: float | None) -> QuotaExhausted:
        return QuotaExhausted(reason, tokens_today=self.day_tokens,
                              requests_today=self.day_requests, reset_in_s=reset_in_s)

    # ---- before / after a call -------------------------------------------------------------
    def before_call(self, est_tokens: int) -> list:
        """Block (sleep) until one more request of ~est_tokens fits every per-minute limit, or
        raise QuotaExhausted when a daily budget would be exceeded. Returns the reservation."""
        self._roll_day()
        if est_tokens > self.tpm_eff:
            raise LLMError("one request is larger than the per-minute token budget "
                           f"({est_tokens} > {int(self.tpm_eff)}); shorten the prompt")
        midnight = self.seconds_to_utc_midnight()
        if self._rpd_blocked_for_s is not None:
            raise self._exhausted("the API reports no daily requests left", self._rpd_blocked_for_s)
        if not self.trust_provider and self.day_requests + 1 > self.rpd_eff:
            raise self._exhausted(f"daily request budget used ({self.day_requests} of "
                                  f"{self.rpd_eff} allowed)", midnight)
        if not self.trust_provider and self.day_tokens + est_tokens > self.tpd_eff:
            raise self._exhausted(f"daily token budget used ({self.day_tokens} of "
                                  f"{int(self.tpd_eff)} allowed)", midnight)
        while True:
            now = self._clock()
            while self.window and self.window[0][0] <= now - 60:
                self.window.popleft()
            wait = 0.0
            if len(self.window) >= self.rpm_eff:
                wait = max(wait, self.window[len(self.window) - self.rpm_eff][0] + 60 - now)
            used = sum(e[1] for e in self.window)
            if used + est_tokens > self.tpm_eff:
                freed = 0
                for t, tokens in self.window:
                    freed += tokens
                    if used - freed + est_tokens <= self.tpm_eff:
                        wait = max(wait, t + 60 - now)
                        break
            if self._tpm_hint is not None:
                remaining, reset_s = self._tpm_hint
                self._tpm_hint = None
                if remaining < est_tokens:
                    wait = max(wait, min(reset_s, 60.0))
            if wait <= 0:
                break
            self._sleep(wait + 0.05)
            self.slept_s += wait + 0.05
        entry = [self._clock(), est_tokens]
        self.window.append(entry)
        return entry

    def after_call(self, entry: list, actual_tokens: int, info: RateInfo | None) -> None:
        entry[1] = actual_tokens
        self.day_tokens += actual_tokens
        self.day_requests += 1
        if info is not None:
            self.rate_info = info
            if info.rpd_remaining is not None and info.rpd_remaining <= 0:
                self._rpd_blocked_for_s = info.rpd_reset_s or self.seconds_to_utc_midnight()
            if info.tpm_remaining is not None and info.tpm_reset_s is not None:
                self._tpm_hint = (info.tpm_remaining, info.tpm_reset_s)
        self._save()

    def failed(self, entry: list, charged_tokens: int = 0) -> None:
        """A call that errored: the request counts and the estimate stays in the minute window. No
        tokens are added to the day (the API reported none) unless the caller knows the call was
        billed (`charged_tokens`, e.g. a reply the API rejected as invalid JSON)."""
        self.day_requests += 1
        self.day_tokens += charged_tokens
        self._save()

    def snapshot(self) -> dict:
        return {"date": self.day, "tokens_today": self.day_tokens,
                "requests_today": self.day_requests, "tpd_budget": int(self.tpd_eff),
                "rpd_budget": self.rpd_eff, "slept_s": round(self.slept_s, 1),
                "http_429": self.n429}


def _chat_completion_text(body: Any) -> str:
    """The final content of a chat completion. Only `message.content` is read: a reasoning field
    (gpt-oss) is ignored, never cached and never stored."""
    if not isinstance(body, dict):
        raise _BadBody("body is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _BadBody("body has no choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        why = choices[0].get("finish_reason")
        raise _BadBody(f"empty reply (finish_reason {why!r}); max_tokens may be too small for the "
                       "reasoning plus the JSON")
    return content


class OpenAICompatClient(OllamaClient):
    chat_path = CHAT_COMPLETIONS_PATH

    def __init__(self, base_url: str, api_key: str, cache: DiskCache, *,
                 limiter: RateLimiter | None = None, json_mode: str = "json_object",
                 max_tokens_cap: int = MAX_COMPLETION_TOKENS_CAP,
                 max_retry_after_s: float = MAX_RETRY_AFTER_S, **kwargs: Any):
        super().__init__(base_url, cache, **kwargs)
        if not api_key:
            raise ValueError("an API key is required")
        self._api_key = api_key
        self.limiter = limiter
        self.json_mode = json_mode
        self.max_tokens_cap = max_tokens_cap
        self.max_retry_after_s = max_retry_after_s  # a longer Retry-After is a quota stop
        self.usage: Counter = Counter()  # prompt, completion, reasoning, total, calls
        self.last_usage: dict[str, int] = {}
        self._last_headers: Any = {}

    @classmethod
    def from_settings(cls, **kwargs: Any) -> OpenAICompatClient:  # type: ignore[override]
        from src.common.schemas import load_llm_settings

        s = load_llm_settings()
        share = kwargs.pop("tpd_share", 0.85)
        trust = kwargs.pop("trust_provider", False)
        limiter = kwargs.pop("limiter", None) or RateLimiter(
            s.rpm, s.tpm, s.rpd, s.tpd, tpd_share=share, trust_provider=trust,
            state_path=s.quota_state_path,
            key_id=hashlib.sha256(s.api_key.encode("utf-8")).hexdigest()[:12])
        kwargs.setdefault("seed", s.seed)
        kwargs.setdefault("retry_budget_s", s.retry_budget_s)
        cache = kwargs.pop("cache", None) or DiskCache(s.cache_dir)
        return cls(s.base_url, s.api_key, cache, limiter=limiter, json_mode=s.json_mode,
                   **kwargs)

    # ---- request shape ---------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}

    def _scrub(self, text: str) -> str:
        return scrub_text(text, [self.base_url, urlsplit(self.base_url).netloc, self._api_key])

    def _options(self, options: dict | None) -> dict:
        return {"temperature": 0, "seed": self.seed, **(options or {})}

    def _payload(self, model: str, messages: list[dict], options: dict | None, fmt: Any) -> dict:
        opts = self._options(options)
        max_tokens = opts.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not (
                0 < max_tokens <= self.max_tokens_cap):
            raise ValueError(f"max_tokens must be an integer in 1..{self.max_tokens_cap}")
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "temperature": opts["temperature"], "seed": opts["seed"],
                                   "max_tokens": max_tokens, "stream": False}
        effort = opts.get("reasoning_effort")
        if effort is not None:
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
            if "gpt-oss" in model.lower():  # only reasoning models take it; others never get it
                payload["reasoning_effort"] = effort
        if fmt == "json" and self.json_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _key_options(self, payload: dict) -> dict:
        return {k: v for k, v in payload.items() if k not in ("model", "messages")}

    @staticmethod
    def _estimate_tokens(payload: dict) -> int:
        chars = sum(len(m["content"]) for m in payload["messages"])
        return chars // 3 + 1 + payload["max_tokens"]  # conservative: 3 characters per token

    # ---- transport -------------------------------------------------------------------------
    def _call(self, send: Callable[[], Any], what: str) -> Any:
        try:
            resp = send()
        except (requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema,
                requests.exceptions.InvalidURL) as exc:
            raise LLMError(f"LLM_API_BASE_URL is not a valid URL ({type(exc).__name__}); "
                           "check .env") from None
        except requests.RequestException as exc:
            raise _Retryable(f"transport error ({type(exc).__name__}): "
                             f"{self._scrub(str(exc))}") from None
        self._last_headers = getattr(resp, "headers", None) or {}
        status = resp.status_code
        if status in (401, 403):
            raise AuthError(f"authentication failed (HTTP {status}): check LLM_API_KEY in .env")
        if status == 404:
            raise ModelNotFoundError(f"model or endpoint not found ({what}); check the model id "
                                     "and LLM_API_BASE_URL in .env (the URL is not printed)")
        if status == 429:
            if self.limiter is not None:
                self.limiter.n429 += 1
            wait = retry_after_seconds(self._last_headers)
            detail = " ".join(self._scrub(resp.text[:300]).split())  # which limit, as the API says
            if wait is None:
                raise _Retryable(f"HTTP 429 without Retry-After: {detail}")
            if wait > self.max_retry_after_s:
                lim = self.limiter
                raise QuotaExhausted(
                    f"HTTP 429: the API asks to wait {int(wait)} s ({detail})", reset_in_s=wait,
                    tokens_today=lim.day_tokens if lim else 0,
                    requests_today=lim.day_requests if lim else 0)
            raise _RetryAfter(f"HTTP 429, retry after {wait:g} s: {detail}", wait)
        if status >= 500:
            raise _Retryable(f"HTTP {status}")
        if status == 400:
            try:
                err = (resp.json() or {}).get("error") or {}
            except (ValueError, AttributeError):
                err = {}
            if isinstance(err, dict) and "failed_generation" in err:
                raise InvalidGeneration(str(err.get("failed_generation") or ""))
        if status != 200:
            raise LLMError(f"HTTP {status} from the API: {self._scrub(resp.text[:300])!r}")
        try:
            return resp.json()
        except ValueError as exc:
            snippet = self._scrub(resp.text[:120]).replace("\n", " ")
            raise _BadBody(f"HTTP 200 but body is not JSON ({snippet!r})") from exc

    def _record_usage(self, body: Any, estimate: int) -> int:
        u = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(u, dict):
            self.usage["usage_missing"] += 1
            self.last_usage = {"prompt": 0, "completion": 0, "reasoning": 0, "total": estimate}
        else:
            prompt = _to_int(u.get("prompt_tokens")) or 0
            completion = _to_int(u.get("completion_tokens")) or 0
            details = u.get("completion_tokens_details")
            reasoning = _to_int(details.get("reasoning_tokens")) or 0 if isinstance(
                details, dict) else 0
            total = _to_int(u.get("total_tokens")) or (prompt + completion)
            self.last_usage = {"prompt": prompt, "completion": completion,
                               "reasoning": reasoning, "total": total}
        for k, v in self.last_usage.items():
            self.usage[k] += v
        self.usage["calls"] += 1
        return self.last_usage["total"]

    def _request(self, payload: dict) -> str:
        est = self._estimate_tokens(payload)

        def attempt(timeout: tuple[float, float]) -> str:
            entry = self.limiter.before_call(est) if self.limiter else None
            try:
                body = self._call(
                    lambda: self._post(self.base_url + self.chat_path, json=payload,
                                       headers=self._headers(), timeout=timeout),
                    f"model {payload['model']!r}")
            except BaseException as exc:
                if entry is not None:
                    self.limiter.failed(entry, entry[1] if isinstance(exc, InvalidGeneration)
                                        else 0)
                raise
            actual = self._record_usage(body, est)
            if entry is not None:
                self.limiter.after_call(entry, actual, parse_rate_headers(self._last_headers))
            return _chat_completion_text(body)

        return self._with_budget(attempt)

    def list_models(self) -> list[str]:
        """Model ids the key can use (GET /models). Not cached, not rate-limited: a health check."""

        def attempt(timeout: tuple[float, float]) -> list[str]:
            body = self._call(lambda: self._get(self.base_url + MODELS_PATH,
                                                headers=self._headers(), timeout=timeout),
                              "listing models")
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list):
                raise _BadBody("body has no 'data' list")
            return [m["id"] for m in data if isinstance(m, dict) and "id" in m]

        return self._with_budget(attempt)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise NotImplementedError("D59: embeddings are computed locally with sentence-transformers")
