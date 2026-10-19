from __future__ import annotations

import json

import pytest

from src.common.cache import DiskCache
from src.common.http import (
    FetchError,
    HttpClient,
    TransportError,
    TransportResponse,
)
from src.common.requestlog import RequestLog

UA = "Test/0.1 (x@example.org)"
OPEN = TransportResponse(404, {}, "")  # no robots.txt: everything allowed
DEAD = "https://dead.example.org"
LIVE = "https://live.example.org"


class FakeWorld:
    """A fake transport plus a fake clock: sleeping advances the clock, nothing really waits."""

    def __init__(self, behaviour):
        self.behaviour = behaviour  # url -> list of outcomes (exceptions raise, responses return)
        self.calls: list[str] = []
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self, method, url, headers, timeout):
        self.calls.append(url)
        outcome = self.behaviour.get(url.split("/")[2], OPEN)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if len(outcome) > 1 else outcome[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(tmp_path, behaviour, **kw):
    world = FakeWorld(behaviour)
    log = RequestLog(tmp_path / "log.jsonl", run_id="t1")
    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=world, clock=world.clock,
                      sleep=world.sleep, min_delay_s=2.0, max_retries=0, request_log=log, **kw)
    return http, world


def notes(tmp_path):
    rows = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text("utf-8").splitlines()]
    return [r["note"] for r in rows]


def fetch_fails(http, url):
    with pytest.raises(FetchError) as exc:
        http.fetch(url, check_robots=False)
    return str(exc.value)


@pytest.mark.parametrize("kind", ["connect", "timeout"])
def test_second_consecutive_failure_opens_the_circuit_and_later_calls_skip_the_network(
        tmp_path, kind):
    http, world = make(tmp_path, {"dead.example.org": TransportError("down", kind)})
    for i in (1, 2):
        fetch_fails(http, f"{DEAD}/p{i}")
    assert len(world.calls) == 2 and http.requests_made == 2
    assert "dead.example.org" in http.circuit_open_hosts
    sleeps, now = list(world.sleeps), world.now
    msg = fetch_fails(http, f"{DEAD}/p3")
    assert "circuit open" in msg
    assert len(world.calls) == 2  # no network call
    assert http.requests_made == 2  # not counted against the cap
    assert world.sleeps == sleeps and world.now == now  # no throttle wait either (fake clock)
    assert http.circuit_skips == 1


def test_one_failure_does_not_open_and_a_success_resets_the_count(tmp_path):
    down = TransportError("down", "connect")
    http, world = make(tmp_path, {"dead.example.org": [down, TransportResponse(200, {}, "ok"),
                                                       down, TransportResponse(200, {}, "ok")]})
    fetch_fails(http, f"{DEAD}/a")  # failure 1
    assert http.fetch(f"{DEAD}/b", check_robots=False).ok  # success resets
    fetch_fails(http, f"{DEAD}/c")  # failure 1 again, not 2
    assert http.circuit_open_hosts == {}
    assert http.fetch(f"{DEAD}/d", check_robots=False).ok
    assert len(world.calls) == 4


def test_other_hosts_and_cache_hits_are_unaffected(tmp_path):
    http, world = make(tmp_path, {"dead.example.org": TransportError("down", "timeout"),
                                  "live.example.org": TransportResponse(200, {}, "fine")})
    assert http.fetch(f"{LIVE}/cached", check_robots=False).text == "fine"
    for i in (1, 2):
        fetch_fails(http, f"{DEAD}/p{i}")
    assert http.fetch(f"{LIVE}/other", check_robots=False).ok  # another host still works
    before = len(world.calls)
    again = http.fetch(f"{LIVE}/cached", check_robots=False)  # served from the disk cache
    assert again.from_cache and len(world.calls) == before


def test_circuit_open_is_logged_once_per_host_not_per_skipped_call(tmp_path):
    http, _ = make(tmp_path, {"dead.example.org": TransportError("down", "connect")})
    for i in range(6):
        fetch_fails(http, f"{DEAD}/p{i}")
    assert notes(tmp_path).count("circuit open") == 1
    assert http.circuit_skips == 4


def test_retries_stop_at_the_open_circuit(tmp_path):
    http, world = make(tmp_path, {"dead.example.org": TransportError("down", "connect")})
    http.max_retries = 5  # without the breaker this would make 6 network attempts
    fetch_fails(http, f"{DEAD}/p")
    assert len(world.calls) == 2


def test_a_new_client_starts_closed(tmp_path):
    http, _ = make(tmp_path, {"dead.example.org": TransportError("down", "connect")})
    for i in (1, 2):
        fetch_fails(http, f"{DEAD}/p{i}")
    assert http.circuit_open_hosts
    fresh, world = make(tmp_path / "again", {"dead.example.org": TransportResponse(200, {}, "up")})
    assert fresh.circuit_open_hosts == {}
    assert fresh.fetch(f"{DEAD}/p", check_robots=False).ok and len(world.calls) == 1


def test_tls_and_dns_handling_is_unchanged(tmp_path):
    http, world = make(tmp_path, {"dead.example.org": TransportError("bad cert", "tls")})
    fetch_fails(http, f"{DEAD}/a")
    assert http.requests_made == 0 and http.unreachable_attempts == 1  # not counted
    assert "dead.example.org" in http.unreachable_hosts
    assert http.circuit_open_hosts == {}  # TLS marks "unreachable", not "circuit open"
    assert "unreachable" in fetch_fails(http, f"{DEAD}/b")
    assert len(world.calls) == 1


def test_other_transport_errors_do_not_count_toward_the_circuit(tmp_path):
    http, _ = make(tmp_path, {"dead.example.org": TransportError("odd", "other")})
    for i in range(4):
        fetch_fails(http, f"{DEAD}/p{i}")
    assert http.circuit_open_hosts == {}
