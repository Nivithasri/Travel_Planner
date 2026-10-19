"""Polite, cached HTTP fetch. Every download in the project goes through this module (rule 4).

- Responses (200, 404, 410) are stored in the disk cache together with `retrieved_at` and the ETag.
  A fresh entry (age <= max_age, or max_age None) is returned without any network call. A stale
  entry is revalidated with If-None-Match / If-Modified-Since; a 304 refreshes `retrieved_at` only.
- robots.txt is fetched once per host (cached for ROBOTS_TTL_S) and checked before each page fetch.
  A missing robots.txt (404/410) allows; an unreachable or erroring one disallows (conservative).
- One request at a time per host with a minimum delay, shared by all threads.
- 429, 5xx and transport errors are retried with backoff (Retry-After honoured, capped), bounded by
  `max_retries` and the optional `deadline`. Other errors are not cached.
- Redirects are NOT followed by the transport. This client follows them itself, one hop at a time,
  and every hop is throttled, counted, logged and checked: a hop is only followed within the same
  site (host without a leading "www.") or to a host in `redirect_hosts`, never from https to http,
  and (when robots are checked for the request) only if robots.txt allows the target. A refused
  redirect is logged as "redirect-refused" and raises FetchError.
- TLS and DNS failures are recorded once per host and never retried; the host is not contacted
  again (`unreachable_hosts` can be preloaded from config). They are not blocks and are not counted
  against `max_requests`, the hard cap on network requests of one run (RequestBudgetExhausted).
- Circuit breaker: after `circuit_threshold` (2) consecutive connect or timeout failures to a host,
  every further request to it in this client's lifetime is skipped with a FetchError: no network, no
  throttle wait, not counted against the cap, logged once as "circuit open". Any completed HTTP
  response resets the count. A new client (the next run) starts closed. TLS/DNS is separate.
- `requests_made` counts real network calls (robots.txt included): the proof that a reuse is free.
  Every request, redirect hop and cache hit can be written to a RequestLog (no content).
- All exception text is redacted (URLs lose their query strings, key=/token= values are removed).
- Transport, clock and sleep are injectable so tests never touch the network or wait.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from src.common.cache import DiskCache
from src.common.redact import redact
from src.common.requestlog import RequestLog

ROBOTS_TTL_S = 24 * 3600
DEFAULT_MIN_DELAY_S = 1.0
DEFAULT_TIMEOUT = (10.0, 30.0)
MAX_RETRY_AFTER_S = 30.0
MAX_REDIRECTS = 5
CIRCUIT_THRESHOLD = 2  # consecutive connect/timeout failures that open a host's circuit
REDIRECT_STATUS = (301, 302, 303, 307, 308)
CACHEABLE_STATUS = (200, 404, 410)
_DNS_MARKERS = ("name or service not known", "getaddrinfo failed", "nameresolutionerror",
                "no address associated", "temporary failure in name resolution")


class FetchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(redact(message))
        self.status = status


class RobotsDisallowed(FetchError):
    pass


class DeadlineExceeded(FetchError):
    pass


class RequestBudgetExhausted(RuntimeError):
    """The hard cap on network requests for this run was reached. Deliberately not a FetchError,
    so no collector can swallow it: it ends the run."""


class TransportError(RuntimeError):
    def __init__(self, message: str = "", kind: str = "other"):
        super().__init__(redact(message))
        self.kind = kind  # tls | dns | connect | timeout | other


@dataclass
class TransportResponse:
    status: int
    headers: dict[str, str]
    text: str


def requests_transport(
    method: str, url: str, headers: Mapping[str, str], timeout: tuple[float, float]
) -> TransportResponse:
    try:
        r = requests.request(method, url, headers=dict(headers), timeout=timeout,
                             allow_redirects=False)  # redirects are followed by HttpClient
    except requests.exceptions.SSLError as exc:
        raise TransportError(str(exc), "tls") from exc
    except requests.exceptions.ConnectTimeout as exc:
        raise TransportError(str(exc), "connect") from exc
    except requests.exceptions.ReadTimeout as exc:
        raise TransportError(str(exc), "timeout") from exc
    except requests.exceptions.ConnectionError as exc:
        text = str(exc).lower()
        raise TransportError(str(exc), "dns" if any(m in text for m in _DNS_MARKERS)
                             else "connect") from exc
    except requests.RequestException as exc:
        raise TransportError(str(exc)) from exc
    return TransportResponse(r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.text)


Transport = Callable[[str, str, Mapping[str, str], tuple[float, float]], TransportResponse]


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    headers: dict[str, str]
    retrieved_at: float  # epoch seconds
    from_cache: bool
    etag: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200

    def json(self) -> Any:
        import json

        return json.loads(self.text)


@dataclass
class _HostState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_request: float = 0.0


def full_url(url: str, params: Mapping[str, Any] | None) -> str:
    if not params:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + urlencode(sorted((k, str(v)) for k, v in params.items()))


def site_of(host: str) -> str:
    h = host.lower()
    return h[4:] if h.startswith("www.") else h


class HttpClient:
    def __init__(
        self,
        cache: DiskCache,
        user_agent: str,
        *,
        transport: Transport = requests_transport,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        min_delay_s: float = DEFAULT_MIN_DELAY_S,
        host_delays: Mapping[str, float] | None = None,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        request_log: RequestLog | None = None,
        max_requests: int | None = None,
        redirect_hosts: Mapping[str, Any] | set[str] | None = None,
        max_redirects: int = MAX_REDIRECTS,
        unreachable_hosts: Mapping[str, str] | None = None,
        circuit_threshold: int = CIRCUIT_THRESHOLD,
        allowlist: Any = None,
        host_caps: Mapping[str, int] | None = None,
    ):
        if not user_agent.strip():
            raise ValueError("user_agent is required (include a contact address)")
        self.cache = cache
        self.user_agent = user_agent
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self.min_delay_s = min_delay_s
        self.host_delays = {k.lower(): v for k, v in (host_delays or {}).items()}
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.timeout = timeout
        self.request_log = request_log
        self.max_requests = max_requests
        self.redirect_hosts = {h.lower() for h in (redirect_hosts or ())}
        self.max_redirects = max_redirects
        self._unreachable: dict[str, str] = {
            k.lower(): v for k, v in (unreachable_hosts or {}).items()}
        self.unreachable_attempts = 0
        self.circuit_threshold = circuit_threshold
        self.allowlist = allowlist  # D43: object with .reason(url) -> str | None
        self.host_caps = {k.lower(): int(v) for k, v in (host_caps or {}).items()}
        self.refusals = 0
        self._failures: Counter[str] = Counter()  # consecutive connect/timeout failures per host
        self._circuit_open: dict[str, str] = {}
        self.circuit_skips = 0
        self.circuit_skips_by_host: dict[str, int] = {}
        self._hosts: dict[str, _HostState] = {}
        self._hosts_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._requests: Counter[str] = Counter()
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()

    # ---- counters -------------------------------------------------------------------------
    @property
    def requests_made(self) -> int:
        with self._counter_lock:
            return sum(self._requests.values())

    def requests_by_host(self) -> dict[str, int]:
        with self._counter_lock:
            return {k: v for k, v in self._requests.items() if v}

    @property
    def unreachable_hosts(self) -> dict[str, str]:
        with self._counter_lock:
            return dict(self._unreachable)

    @property
    def circuit_open_hosts(self) -> dict[str, str]:
        with self._counter_lock:
            return dict(self._circuit_open)

    def _log(self, url: str, status: int | None, from_cache: bool, attempt: int,
             note: str | None = None) -> None:
        if self.request_log is not None:
            self.request_log.write(url, status, from_cache, attempt, note)

    # ---- network --------------------------------------------------------------------------
    def _host_state(self, host: str) -> _HostState:
        with self._hosts_lock:
            return self._hosts.setdefault(host, _HostState())

    def _once(self, url: str, headers: Mapping[str, str], attempt: int) -> TransportResponse:
        """One throttled, counted, logged network request (one hop, one attempt)."""
        host = urlsplit(url).netloc.lower()
        if self.allowlist is not None:
            why = self.allowlist.reason(url)
            if why:
                self.refusals += 1
                raise FetchError(f"not allowlisted: {why}")
        with self._counter_lock:
            known = self._unreachable.get(host)
        if known is not None:
            raise FetchError(f"host unreachable ({known}); not contacted again")
        with self._counter_lock:
            opened = self._circuit_open.get(host)
            if opened is not None:
                self.circuit_skips += 1
                first_skip = self.circuit_skips_by_host.get(host, 0) == 0
                self.circuit_skips_by_host[host] = self.circuit_skips_by_host.get(host, 0) + 1
        if opened is not None:
            if first_skip:
                self._log(url, None, False, attempt, "circuit open")
            raise FetchError(f"circuit open ({opened}); host skipped for the rest of the run")
        state = self._host_state(host)
        with state.lock:
            delay = self.host_delays.get(host, self.min_delay_s)
            wait = state.last_request + delay - self._clock()
            if wait > 0:
                self._sleep(wait)
            state.last_request = self._clock()
            with self._counter_lock:
                used = sum(self._requests.values())
                if self.max_requests is not None and used >= self.max_requests:
                    raise RequestBudgetExhausted(
                        f"request budget exhausted: {self.max_requests} network requests")
                cap = self.host_caps.get(host)
                if cap is not None and self._requests[host] >= cap:
                    raise RequestBudgetExhausted(
                        f"per-host cap reached: {cap} network requests to {host}")
                self._requests[host] += 1
            try:
                resp = self._transport("GET", url, headers, self.timeout)
            except TransportError as exc:
                if exc.kind in ("tls", "dns"):
                    with self._counter_lock:  # never completed: not counted against the cap
                        self._requests[host] -= 1
                        self.unreachable_attempts += 1
                        first = host not in self._unreachable
                        self._unreachable[host] = f"{exc.kind}: {redact(exc)[:120]}"
                    if first:
                        self._log(url, None, False, attempt, f"unreachable:{exc.kind}")
                    raise FetchError(f"host unreachable ({exc.kind}): {host}") from exc
                self._log(url, None, False, attempt, f"transport:{exc.kind}")
                if exc.kind in ("connect", "timeout"):
                    with self._counter_lock:
                        self._failures[host] += 1
                        if self._failures[host] >= self.circuit_threshold:
                            self._circuit_open.setdefault(
                                host, f"{self._failures[host]} consecutive {exc.kind} failures")
                raise
        with self._counter_lock:
            self._failures[host] = 0  # any completed response resets the count
        self._log(url, resp.status, False, attempt,
                  "redirect" if resp.status in REDIRECT_STATUS else None)
        return resp

    def _redirect_refusal(self, cur: str, nxt: str, check_robots: bool,
                          deadline: float | None) -> str | None:
        a, b = urlsplit(cur), urlsplit(nxt)
        if b.scheme not in ("http", "https"):
            return f"scheme {b.scheme!r}"
        if a.scheme == "https" and b.scheme == "http":
            return "https to http downgrade"
        if site_of(a.netloc) != site_of(b.netloc) and b.netloc.lower() not in self.redirect_hosts:
            return f"host {b.netloc.lower()} is outside the allowlist"
        if check_robots and not self.allowed(nxt, deadline):
            return "robots.txt disallows the target"
        return None

    def _follow(self, url: str, headers: Mapping[str, str], attempt: int, check_robots: bool,
                deadline: float | None) -> TransportResponse:
        cur, hops = url, 0
        while True:
            resp = self._once(cur, headers, attempt)
            loc = resp.headers.get("location")
            if resp.status not in REDIRECT_STATUS or not loc:
                return resp
            hops += 1
            nxt = urljoin(cur, loc)
            if hops > self.max_redirects:
                raise FetchError(f"too many redirects from {url}", resp.status)
            reason = self._redirect_refusal(cur, nxt, check_robots, deadline)
            if reason:
                self._log(nxt, resp.status, False, attempt, f"redirect-refused: {reason}")
                raise FetchError(f"redirect from {cur} to {nxt} not followed: {reason}",
                                 resp.status)
            cur = nxt

    def _send(self, url: str, headers: Mapping[str, str], deadline: float | None,
              check_robots: bool = True) -> TransportResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if deadline is not None and self._clock() >= deadline:
                raise DeadlineExceeded(f"deadline reached before fetching {url}")
            retry_after = 0.0
            resp: TransportResponse | None
            try:
                resp = self._follow(url, headers, attempt + 1, check_robots, deadline)
            except TransportError as exc:
                last_exc, resp = exc, None
            if resp is not None:
                if resp.status != 429 and resp.status < 500:
                    return resp
                last_exc = FetchError(f"HTTP {resp.status} for {url}", resp.status)
                retry_after = _retry_after(resp.headers.get("retry-after"), self._clock())
            if attempt < self.max_retries:
                self._sleep(max(retry_after, self.backoff_s * (2**attempt)))
        if isinstance(last_exc, FetchError):
            raise last_exc
        raise FetchError(f"transport error for {url}: {last_exc}")

    # ---- robots ---------------------------------------------------------------------------
    def _robots_for(self, url: str, deadline: float | None) -> RobotFileParser | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._robots_lock:
            if origin in self._robots:
                return self._robots[origin]
        res = self._fetch(
            origin + "/robots.txt", None, ROBOTS_TTL_S, deadline, check_robots=False,
            raise_on_error=False,
        )
        rp: RobotFileParser
        if res is None:
            return None  # unreachable/erroring robots.txt: disallow now, retry on the next call
        elif res.status in (404, 410):
            rp = RobotFileParser()
            rp.parse([])  # no robots.txt: allowed
        else:
            rp = RobotFileParser()
            rp.parse(res.text.splitlines())
        with self._robots_lock:
            self._robots[origin] = rp
        return rp

    def allowed(self, url: str, deadline: float | None = None) -> bool:
        rp = self._robots_for(url, deadline)
        return rp is not None and rp.can_fetch(self.user_agent, url)

    # ---- public fetch ---------------------------------------------------------------------
    def fetch(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_age: float | None = None,
        headers: Mapping[str, str] | None = None,
        check_robots: bool = True,
        deadline: float | None = None,
        validate: Callable[[str], bool] | None = None,
        secret_headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """Return the response, from cache when fresh. Raises FetchError / RobotsDisallowed.

        `secret_headers` (API keys, bearer tokens) are sent but never part of the cache key, so a
        rotated token still hits the cache, and they are never stored or put in a message.

        `validate(text)` must be True for a 200 body to be cached or served from cache. Some APIs
        report errors with HTTP 200; those bodies raise FetchError and are never cached."""
        res = self._fetch(
            full_url(url, params), headers, max_age, deadline, check_robots=check_robots,
            raise_on_error=True, validate=validate, secret_headers=secret_headers,
        )
        assert res is not None
        return res

    def _fetch(
        self,
        url: str,
        extra_headers: Mapping[str, str] | None,
        max_age: float | None,
        deadline: float | None,
        *,
        check_robots: bool,
        raise_on_error: bool,
        validate: Callable[[str], bool] | None = None,
        secret_headers: Mapping[str, str] | None = None,
    ) -> FetchResult | None:
        host = urlsplit(url).netloc.lower()
        inputs = {"url": url, "headers": dict(sorted((extra_headers or {}).items()))}
        entry = self.cache.get("http", host, inputs)
        now = self._clock()
        if _valid_entry(entry) and not _passes(entry, validate):
            entry = None  # a cached error body is a miss; it is overwritten by a good one
        if _valid_entry(entry) and (max_age is None or now - entry["retrieved_at"] <= max_age):
            self._log(url, entry["status"], True, 0)
            return _result(url, entry, from_cache=True)
        if check_robots and not self.allowed(url, deadline):
            raise RobotsDisallowed(f"robots.txt disallows {url} for {self.user_agent}")
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        headers.update(extra_headers or {})
        headers.update(secret_headers or {})
        if _valid_entry(entry):
            if entry.get("etag"):
                headers["If-None-Match"] = entry["etag"]
            if entry.get("last_modified"):
                headers["If-Modified-Since"] = entry["last_modified"]
        try:
            resp = self._send(url, headers, deadline, check_robots)
        except DeadlineExceeded:
            raise
        except FetchError:
            if raise_on_error:
                raise
            return None
        retrieved_at = self._clock()
        if resp.status == 304 and _valid_entry(entry):
            entry = {**entry, "retrieved_at": retrieved_at}
            self.cache.set("http", host, inputs, entry)
            return _result(url, entry, from_cache=False)
        if resp.status == 200 and validate is not None and not validate(resp.text):
            if raise_on_error:
                raise FetchError(f"invalid response body for {url}: {resp.text[:200]!r}", 200)
            return None
        if resp.status in CACHEABLE_STATUS:
            entry = {
                "status": resp.status,
                "headers": resp.headers,
                "text": resp.text,
                "retrieved_at": retrieved_at,
                "etag": resp.headers.get("etag"),
                "last_modified": resp.headers.get("last-modified"),
            }
            self.cache.set("http", host, inputs, entry)
            return _result(url, entry, from_cache=False)
        if raise_on_error:
            raise FetchError(f"HTTP {resp.status} for {url}", resp.status)
        return None


def _passes(entry: dict, validate: Callable[[str], bool] | None) -> bool:
    return validate is None or entry.get("status") != 200 or validate(entry["text"])


def _valid_entry(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("retrieved_at"), (int, float))
        and isinstance(entry.get("text"), str)
        and isinstance(entry.get("status"), int)
    )


def _result(url: str, entry: dict, *, from_cache: bool) -> FetchResult:
    return FetchResult(
        url=url,
        status=entry["status"],
        text=entry["text"],
        headers=entry.get("headers") or {},
        retrieved_at=entry["retrieved_at"],
        from_cache=from_cache,
        etag=entry.get("etag"),
    )


def _retry_after(value: str | None, now: float) -> float:
    if not value:
        return 0.0
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER_S)
    except ValueError:
        pass
    try:
        return min(max(parsedate_to_datetime(value).timestamp() - now, 0.0), MAX_RETRY_AFTER_S)
    except (TypeError, ValueError):
        return 0.0
