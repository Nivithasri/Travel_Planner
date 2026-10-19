from __future__ import annotations

import json

import pytest
import requests

from src.common.cache import DiskCache
from src.common.http import (
    DeadlineExceeded,
    FetchError,
    HttpClient,
    RequestBudgetExhausted,
    RobotsDisallowed,
    TransportError,
    TransportResponse,
    requests_transport,
)
from src.common.redact import redact
from src.common.requestlog import RequestLog

UA = "TravelPlannerTest/0.1 (test@example.com)"


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeServer:
    """Maps url -> response (or list of responses consumed in order)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        r = self.routes.get(url)
        if r is None:
            return TransportResponse(404, {}, "")
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r


def make(tmp_path, routes, **kw):
    clock = FakeClock()
    server = FakeServer(routes)
    client = HttpClient(
        DiskCache(tmp_path / "c"), UA, transport=server, clock=clock, sleep=clock.sleep, **kw
    )
    return client, server, clock


def ok(text="hi", **headers):
    return TransportResponse(200, headers, text)


ROBOTS_OPEN = TransportResponse(200, {}, "User-agent: *\nAllow: /\n")


def test_cache_hit_makes_no_request(tmp_path):
    client, server, _ = make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                        "https://a.org/p": ok("body")})
    first = client.fetch("https://a.org/p")
    n = client.requests_made
    second = client.fetch("https://a.org/p")
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.text == "body"
    assert client.requests_made == n == 2  # robots.txt + page


def test_stale_entry_is_revalidated_with_etag_and_304_refreshes(tmp_path):
    routes = {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/p": [ok("v1", etag='"e1"'), TransportResponse(304, {}, "")],
    }
    client, server, clock = make(tmp_path, routes)
    client.fetch("https://a.org/p", max_age=60)
    clock.t += 120
    res = client.fetch("https://a.org/p", max_age=60)
    assert res.text == "v1" and not res.from_cache
    assert server.calls[-1][1]["If-None-Match"] == '"e1"'
    assert res.retrieved_at == clock.t
    n = client.requests_made
    client.fetch("https://a.org/p", max_age=60)  # fresh again
    assert client.requests_made == n


def test_robots_disallow_blocks_without_fetching_page(tmp_path):
    robots = TransportResponse(200, {}, "User-agent: *\nDisallow: /private\n")
    client, server, _ = make(tmp_path, {"https://a.org/robots.txt": robots,
                                        "https://a.org/private/x": ok()})
    with pytest.raises(RobotsDisallowed):
        client.fetch("https://a.org/private/x")
    assert all("private" not in u for u, _ in server.calls)
    assert client.fetch("https://a.org/robots.txt", check_robots=False).ok


def test_missing_robots_allows_but_unreachable_robots_disallows(tmp_path):
    client, _, _ = make(tmp_path, {"https://a.org/p": ok()})  # robots.txt -> 404
    assert client.fetch("https://a.org/p").ok
    bad_robots = TransportResponse(503, {}, "")
    client2, _, _ = make(
        tmp_path / "x", {"https://b.org/robots.txt": bad_robots, "https://b.org/p": ok()},
        max_retries=1,
    )
    with pytest.raises(RobotsDisallowed):
        client2.fetch("https://b.org/p")


def test_min_delay_between_requests_to_one_host(tmp_path):
    client, _, clock = make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                       "https://a.org/1": ok(), "https://a.org/2": ok()},
                            min_delay_s=2.0)
    client.fetch("https://a.org/1")
    t = clock.t
    client.fetch("https://a.org/2")
    assert clock.t - t >= 2.0


def test_retries_5xx_and_transport_errors_then_succeeds(tmp_path):
    routes = {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/p": [TransportResponse(503, {}, ""), TransportError("boom"), ok("fine")],
    }
    client, _, _ = make(tmp_path, routes)
    assert client.fetch("https://a.org/p").text == "fine"


def test_gives_up_and_does_not_cache_errors(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/p": [TransportResponse(500, {}, "")]}
    client, _, _ = make(tmp_path, routes, max_retries=2)
    with pytest.raises(FetchError) as ei:
        client.fetch("https://a.org/p")
    assert ei.value.status == 500
    assert not client.cache.has("http", "a.org", {"url": "https://a.org/p", "headers": {}})


def test_404_is_cached_as_negative_result(tmp_path):
    client, _, _ = make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN})
    assert client.fetch("https://a.org/missing").status == 404
    n = client.requests_made
    assert client.fetch("https://a.org/missing").status == 404
    assert client.requests_made == n


def test_deadline_stops_fetching(tmp_path):
    client, _, clock = make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                       "https://a.org/p": ok()})
    with pytest.raises(DeadlineExceeded):
        client.fetch("https://a.org/p", deadline=clock.t - 1)


def test_params_are_part_of_the_cache_key(tmp_path):
    client, _, _ = make(tmp_path, {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/api?q=1": ok("one"), "https://a.org/api?q=2": ok("two")})
    assert client.fetch("https://a.org/api", {"q": 1}).text == "one"
    assert client.fetch("https://a.org/api", {"q": 2}).text == "two"


def test_user_agent_required(tmp_path):
    with pytest.raises(ValueError):
        HttpClient(DiskCache(tmp_path), " ")






def test_failed_robots_lookup_is_not_memoized(tmp_path):
    routes = {
        "https://a.org/robots.txt": [TransportResponse(503, {}, ""), ROBOTS_OPEN],
        "https://a.org/p": ok("late"),
    }
    client, _, _ = make(tmp_path, routes, max_retries=0)
    with pytest.raises(RobotsDisallowed):
        client.fetch("https://a.org/p")
    assert client.fetch("https://a.org/p").text == "late"  # robots recovered, same client


def _no_error(text: str) -> bool:
    return '"error"' not in text


def test_validate_rejects_error_bodies_and_never_caches_them(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/api": [ok('{"error": {"code": "x"}}'), ok('{"fine": 1}')]}
    client, _, _ = make(tmp_path, routes)
    with pytest.raises(FetchError):
        client.fetch("https://a.org/api", validate=_no_error)
    assert client.fetch("https://a.org/api", validate=_no_error).text == '{"fine": 1}'


def test_cached_error_body_is_treated_as_a_miss(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/api": [ok('{"error": 1}'), ok('{"good": 1}')]}
    client, _, _ = make(tmp_path, routes)
    assert client.fetch("https://a.org/api").text == '{"error": 1}'  # cached without validation
    assert client.fetch("https://a.org/api", validate=_no_error).text == '{"good": 1}'


def test_secret_headers_are_sent_but_not_part_of_the_cache_key_or_the_cache_file(tmp_path):
    client, server, _ = make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                        "https://a.org/p": ok("body")})
    client.fetch("https://a.org/p", secret_headers={"Authorization": "Bearer TOKEN-1"})
    assert server.calls[-1][1]["Authorization"] == "Bearer TOKEN-1"
    n = client.requests_made
    hit = client.fetch("https://a.org/p", secret_headers={"Authorization": "Bearer TOKEN-2"})
    assert hit.from_cache and client.requests_made == n  # a rotated token still hits the cache
    stored = "".join(p.read_text("utf-8") for p in (tmp_path / "c").rglob("*.json"))
    assert "TOKEN" not in stored


# ---- merged from tests/test_http_policy.py: 

POLICY_UA = "Test/0.1 (x@example.org)"
OPEN = TransportResponse(404, {}, "")  # no robots.txt: everything allowed


def redirect(to, status=301):
    return TransportResponse(status, {"location": to}, "")


def pok(text="body"):
    return TransportResponse(200, {}, text)


class Server:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, headers, timeout):
        self.calls.append(url)
        r = self.routes.get(url, OPEN)
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r


def pmake(tmp_path, routes, log=False, **kw):
    server = Server(routes)
    rlog = RequestLog(tmp_path / "log.jsonl", run_id="t1") if log else None
    http = HttpClient(DiskCache(tmp_path / "c"), POLICY_UA, transport=server, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0, request_log=rlog, **kw)
    return http, server, rlog


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "log.jsonl").read_text("utf-8").splitlines()]


# ---- redirects ----------------------------------------------------------------------------
def test_transport_does_not_follow_redirects_itself(monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen.update(kw)
        raise requests.ConnectionError("stop")

    monkeypatch.setattr(requests, "request", fake_request)
    with pytest.raises(TransportError):
        requests_transport("GET", "https://a.org/", {}, (1, 1))
    assert seen["allow_redirects"] is False


def test_same_site_redirect_is_followed_and_every_hop_is_counted_and_logged(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://www.a.org/y"),
                                      "https://www.a.org/y": pok("final")}, log=True)
    r = http.fetch("https://a.org/x", check_robots=False)
    assert r.text == "final" and http.requests_made == 2
    assert [(x["host"], x["status"], x["note"]) for x in lines(tmp_path)] == [
        ("a.org", 301, "redirect"), ("www.a.org", 200, None)]


def test_redirect_to_another_site_is_refused_and_logged(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://evil.example/y")},
                           log=True)
    with pytest.raises(FetchError, match="not followed"):
        http.fetch("https://a.org/x", check_robots=False)
    assert server.calls == ["https://a.org/x"]  # the other host was never contacted
    refused = [x for x in lines(tmp_path) if (x["note"] or "").startswith("redirect-refused")]
    assert len(refused) == 1 and refused[0]["host"] == "evil.example"
    assert "outside the allowlist" in refused[0]["note"]


def test_redirect_host_on_the_allowlist_is_followed(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://b.org/y"),
                                 "https://b.org/y": pok("b")}, redirect_hosts={"b.org"})
    assert http.fetch("https://a.org/x", check_robots=False).text == "b"


def test_https_to_http_downgrade_is_refused(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("http://a.org/y")})
    with pytest.raises(FetchError, match="downgrade"):
        http.fetch("https://a.org/x", check_robots=False)
    assert server.calls == ["https://a.org/x"]


def test_redirect_target_disallowed_by_robots_is_not_followed(tmp_path):
    http, server, _ = pmake(tmp_path, {
        "https://a.org/robots.txt": pok("User-agent: *\nDisallow: /private\n"),
        "https://a.org/x": redirect("https://a.org/private/y")})
    with pytest.raises(FetchError, match="robots.txt disallows the target"):
        http.fetch("https://a.org/x")
    assert "https://a.org/private/y" not in server.calls


def test_redirect_loop_is_cut_off(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://a.org/x")})
    with pytest.raises(FetchError, match="too many redirects"):
        http.fetch("https://a.org/x", check_robots=False)
    assert http.requests_made == 6  # the first request plus 5 followed hops


# ---- redaction ----------------------------------------------------------------------------
FAKE_KEY = "ABCDEFGH12345678"


def test_redact_removes_query_strings_and_key_values():
    text = (f"403 Client Error: Forbidden for url: https://www.youtube.com/youtubei/v1/player"
            f"?key={FAKE_KEY}&prettyPrint=false and token={FAKE_KEY} and api_key: {FAKE_KEY}")
    out = redact(text)
    assert FAKE_KEY not in out and "prettyPrint" not in out
    assert "https://www.youtube.com/youtubei/v1/player" in out


def test_http_error_with_a_key_in_its_url_is_redacted_in_fetch_errors(tmp_path):
    http_error = requests.HTTPError(f"403 Client Error: Forbidden for url: "
                                    f"https://a.org/p?key={FAKE_KEY}")
    boom = TransportError(str(http_error))
    http, _, _ = pmake(tmp_path, {"https://a.org/p?key=" + FAKE_KEY: boom})
    with pytest.raises(FetchError) as ei:
        http.fetch("https://a.org/p", {"key": FAKE_KEY}, check_robots=False)
    assert FAKE_KEY not in str(ei.value)
    assert FAKE_KEY not in redact(http_error)


def test_request_log_has_no_query_string_content_or_key(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/p?key=" + FAKE_KEY: pok("SECRET-PAGE-TEXT")},
                      log=True)
    http.fetch("https://a.org/p", {"key": FAKE_KEY}, check_robots=False)
    raw = (tmp_path / "log.jsonl").read_text("utf-8")
    assert FAKE_KEY not in raw and "SECRET-PAGE-TEXT" not in raw and "?" not in raw


# ---- request log --------------------------------------------------------------------------
def test_request_log_records_fields_cache_hits_and_retry_attempts(tmp_path):
    routes = {"https://a.org/robots.txt": OPEN,
              "https://a.org/sitemap.xml": [TransportResponse(503, {}, ""), pok("<urlset/>")],
              "https://a.org/w/api.php?x=1": pok("{}")}
    http, _, rlog = pmake(tmp_path, routes, log=True)
    http.max_retries = 1
    rlog.source = "web"
    http.fetch("https://a.org/sitemap.xml")
    http.fetch("https://a.org/sitemap.xml")  # cache hit
    http.fetch("https://a.org/w/api.php", {"x": 1}, check_robots=False)
    got = [(x["kind"], x["status"], x["from_cache"], x["attempt"]) for x in lines(tmp_path)]
    assert got == [("robots", 404, False, 1), ("sitemap", 503, False, 1),
                   ("sitemap", 200, False, 2), ("sitemap", 200, True, 0),
                   ("api", 200, False, 1)]
    first = lines(tmp_path)[0]
    assert set(first) == {"ts", "run_id", "host", "path", "kind", "status", "from_cache",
                          "attempt", "source", "note"}
    assert first["source"] == "web" and first["run_id"] == "t1"


# ---- hard cap -----------------------------------------------------------------------------
def test_request_cap_raises_at_the_next_network_request_and_cache_hits_are_free(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/1": pok(), "https://a.org/2": pok(),
                                      "https://a.org/3": pok()}, max_requests=2)
    http.fetch("https://a.org/1", check_robots=False)
    http.fetch("https://a.org/1", check_robots=False)  # cache hit: free
    http.fetch("https://a.org/2", check_robots=False)
    with pytest.raises(RequestBudgetExhausted):
        http.fetch("https://a.org/3", check_robots=False)
    assert http.requests_made == 2 and "https://a.org/3" not in server.calls


def test_budget_exhaustion_is_not_swallowed_by_a_collector(tmp_path):
    from src.collect.types import Budget
    from src.collect.wiki import WikiCollector
    from src.common.schemas import PlaceRegistryEntry

    http, _, _ = pmake(tmp_path, {}, max_requests=0)
    e = PlaceRegistryEntry(name="X", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=1.0, lon=1.0, wikipedia_title="X")
    with pytest.raises(RequestBudgetExhausted):
        WikiCollector("wikipedia", http).collect(e, Budget(), lambda: 0, tmp_path / "raw")


# ---- TLS / DNS ----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["tls", "dns"])
def test_tls_and_dns_failures_are_recorded_once_never_retried_and_not_counted(tmp_path, kind):
    http, server, _ = pmake(tmp_path, {"https://bad.org/a": TransportError("handshake", kind)},
                           log=True, max_requests=1)
    http.max_retries = 3
    for _ in range(3):
        with pytest.raises(FetchError, match="unreachable") as ei:
            http.fetch("https://bad.org/a", check_robots=False)
        assert ei.value.status is None  # not a 403/429, so never a block
    assert server.calls == ["https://bad.org/a"]  # one attempt in total, no retry
    assert http.requests_made == 0 and http.unreachable_attempts == 1
    assert list(http.unreachable_hosts) == ["bad.org"]
    notes = [x["note"] for x in lines(tmp_path)]
    assert notes == [f"unreachable:{kind}"]  # logged once
    http._transport = Server({"https://good.org/": pok()})  # another host works under the same cap
    assert http.fetch("https://good.org/", check_robots=False).ok


def test_preloaded_unreachable_hosts_are_never_contacted(tmp_path):
    http, server, _ = pmake(tmp_path, {}, unreachable_hosts={"tourism.telangana.gov.in": "tls"})
    with pytest.raises(FetchError, match="unreachable"):
        http.fetch("https://tourism.telangana.gov.in/", check_robots=False)
    assert server.calls == [] and http.requests_made == 0


def test_timeout_is_not_treated_as_unreachable(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://slow.org/": TransportError("slow", "timeout")})
    with pytest.raises(FetchError):
        http.fetch("https://slow.org/", check_robots=False)
    assert http.unreachable_hosts == {} and http.requests_made == 1


