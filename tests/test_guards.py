"""Guards the mutation checks rely on: allowlist, redirect refusals, api_hosts golden list,
request-log guard, contact privacy, circuit breaker, hold-out filter (D47)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from src.collect import run as orchestrator
from src.collect.run import make_http
from src.collect.sources import host_key as web_host_key
from src.common.cache import DiskCache
from src.common.http import (
    Allowlist,
    FetchError,
    HttpClient,
    RequestBudgetExhausted,
    RequestLog,
    TransportError,
    TransportResponse,
    host_key,
)
from src.common.schemas import REPO_ROOT, CollectSettings, ConfigError, load_collect_settings
from src.facts.facts import HoldoutFilter

# ==== from test_allowlist.py


ALLOW_UA = "Test/0.1 (x@example.org)"


@pytest.fixture(scope="module")
def shipped():
    return Allowlist.from_config()


def test_inditales_is_accepted_only_through_the_explicit_www_twin_rule(shipped):
    # the configured domain is www.inditales.com; its bare twin is admitted by host_key()
    assert host_key("www.inditales.com") == host_key("inditales.com") == "inditales.com"
    assert web_host_key("INDITALES.com") == "inditales.com"
    assert shipped.reason("https://inditales.com/post-sitemap.xml") is None
    assert shipped.reason("https://www.inditales.com/robots.txt") is None
    for look_alike in ("https://blog.inditales.com/x",  # subdomain: no suffix match
                       "https://evilinditales.com/x",
                       "https://inditales.com.evil.org/x",
                       "https://www.www.inditales.com/x",
                       "https://inditales.com:8443/x",
                       "ftp://inditales.com/x"):
        assert shipped.reason(look_alike), look_alike


def test_a_redirect_style_host_is_not_admitted_by_the_web_rule(shipped):
    # the redirect rule lives in HttpClient (site_of); the allowlist still checks the target
    assert shipped.reason("https://tracker.example.net/redirect?to=inditales.com")


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/w/api.php?action=query",
    "https://en.wikivoyage.org/w/api.php?action=query",
    "https://www.wikidata.org/w/api.php?action=wbsearchentities",
    "https://www.wikidata.org/wiki/Special:EntityData/Q1361.json",
    "https://query.wikidata.org/sparql?query=x",
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/X/monthly/1/2",
    "https://overpass-api.de/api/interpreter?data=x",
    "https://archive-api.open-meteo.com/v1/archive?latitude=1",
    "https://en.wikipedia.org/robots.txt",
])
def test_listed_hosts_and_paths_are_allowed(shipped, url):
    assert shipped.reason(url) is None


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/w/index.php?title=X",  # a non-listed Wikimedia path
    "https://en.wikipedia.org/wiki/Munnar",
    "https://en.wikivoyage.org/w/rest.php/v1/page/X",
    "https://www.wikidata.org/w/index.php",
    "https://www.wikidata.org/wiki/Q42",
    "https://query.wikidata.org/bigdata/namespace/wdq/sparql-extra",
    "https://wikimedia.org/api/rest_v1/metrics/edits/aggregate",
    "https://de.wikipedia.org/w/api.php",  # another Wikimedia host
    "https://commons.wikimedia.org/w/api.php",
    "https://overpass-api.de/api/status",
    "https://archive-api.open-meteo.com/v1/forecast",
    "https://api.open-meteo.com/v1/archive",
    "https://example.org/",
])
def test_everything_else_is_refused(shipped, url):
    assert shipped.reason(url), url


def test_a_path_prefix_without_a_trailing_slash_does_not_match_a_longer_name(shipped):
    assert shipped.reason("https://en.wikipedia.org/w/api.php2") is not None
    assert shipped.reason("https://en.wikipedia.org/w/api.php/extra") is None


def allow_client(tmp_path, allow, **kw):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        return TransportResponse(200, {}, "{}")

    http = HttpClient(DiskCache(tmp_path / "c"), ALLOW_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0, allowlist=allow, **kw)
    return http, calls


def test_a_refused_url_makes_no_request_and_is_not_counted(tmp_path, shipped):
    http, calls = allow_client(tmp_path, shipped)
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://en.wikipedia.org/w/index.php", check_robots=False)
    assert calls == [] and http.requests_made == 0 and http.refusals == 1
    http.fetch("https://en.wikipedia.org/w/api.php", {"action": "query"}, check_robots=False)
    assert len(calls) == 1


def test_a_sitemap_page_on_another_host_is_refused(tmp_path):
    allow = Allowlist(web_domains=["www.example.org"])
    http, calls = allow_client(tmp_path, allow)
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://elsewhere.net/munnar", check_robots=False)
    assert http.fetch("https://example.org/munnar", check_robots=False).ok
    assert calls == ["https://example.org/munnar"]


def test_per_host_cap_is_a_hard_stop(tmp_path):
    allow = Allowlist(api_prefixes={"api.example.org": ["/v1/"]}, api_caps={"api.example.org": 2})
    http, calls = allow_client(tmp_path, allow, host_caps=allow.caps)
    http.fetch("https://api.example.org/v1/a", check_robots=False)
    http.fetch("https://api.example.org/v1/b", check_robots=False)
    with pytest.raises(RequestBudgetExhausted):
        http.fetch("https://api.example.org/v1/c", check_robots=False)
    assert len(calls) == 2


def test_make_http_clients_carry_the_shipped_allowlist(tmp_path):
    cfg = CollectSettings("x@example.org", ALLOW_UA, Path(tmp_path) / "cache")
    http = make_http(cfg, request_log=None)
    assert http.allowlist is not None and http.host_caps["en.wikipedia.org"] == 30
    assert http.host_delays["en.wikipedia.org"] == 1.0
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://en.wikipedia.org/w/index.php", check_robots=False)


def test_wiki_collector_and_resolver_paths_are_the_listed_ones(shipped):
    from src.collect.sources import WIKIS
    from src.places import places as wd

    for cfg in WIKIS.values():
        assert shipped.reason(cfg["api"]) is None
    for url in (wd.WIKIDATA_API, wd.SPARQL_URL,
                wd.PAGEVIEWS_URL.format(project="en.wikipedia", title="X", start="1", end="2")):
        assert shipped.reason(url) is None, url


def test_every_api_host_entry_has_a_plain_hostname_and_a_decision():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "sources.json").read_text("utf-8"))
    for h in cfg["api_hosts"]["hosts"]:
        assert h["host"] == h["host"].lower() and "/" not in h["host"] and "[" not in h["host"]
        assert h["decision"] in ("D37", "D43") and h["robots_exempt"] is True
        if h["decision"] == "D43":
            assert h["min_delay_s"] == 1.0 and h["max_requests_per_run"] == 30 and h["terms_status"]


# ==== from test_redirect_allowlist.py


REDIRECT_UA = "Test/0.1 (x@example.org)"


AT = "@"  # built here so no email-like literal appears in the source (privacy test)


PROD_REDIRECT_HOSTS = {"sahapedia.org", "www.sahapedia.org", "inditales.com", "www.inditales.com",
                       "en.wikipedia.org", "en.wikivoyage.org"}


def redirecting(first_url, location, status=302):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        if url == first_url:
            return TransportResponse(status, {"location": location}, "")
        return TransportResponse(200, {}, "SHOULD NEVER BE FETCHED")

    return transport, calls


def run(tmp_path, first_url, location, extra_redirect_hosts=()):
    transport, calls = redirecting(first_url, location)
    http = HttpClient(DiskCache(tmp_path / "c"), REDIRECT_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0,
                      allowlist=Allowlist.from_config(),
                      redirect_hosts=PROD_REDIRECT_HOSTS | set(extra_redirect_hosts))
    with pytest.raises(FetchError):
        http.fetch(first_url, check_robots=False)
    return calls


def host_of(url):
    return url.split("//", 1)[1].split("/", 1)[0].lower()


@pytest.mark.parametrize("permissive", [False, True])
def test_allowed_web_host_redirecting_to_an_unlisted_host_is_refused_before_following(
        tmp_path, permissive):
    first, target = "https://sahapedia.org/a", "https://unlisted.example.net/landing"
    # permissive=True also puts the target in the redirect rule's host list, so only the
    # allowlist is left to stop it (defence in depth)
    calls = run(tmp_path, first, target, [host_of(target)] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("permissive", [False, True])
def test_api_host_redirecting_outside_its_path_prefix_is_refused_before_following(
        tmp_path, permissive):
    first = "https://overpass-api.de/api/interpreter?data=x"
    target = "https://overpass-api.de/api/status"
    calls = run(tmp_path, first, target, ["overpass-api.de"] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("target", [
    "https://inditales.com" + AT + "evil.example/",  # real host is evil.example
    "https://evil.example" + AT + "inditales.com/",  # allowed host behind user information
    "https://user:pass" + AT + "www.sahapedia.org/x",
])
@pytest.mark.parametrize("permissive", [False, True])
def test_redirect_to_a_userinfo_url_is_refused_before_following(tmp_path, target, permissive):
    first = "https://inditales.com/post"
    netloc = target.split("//", 1)[1].split("/", 1)[0].lower()
    calls = run(tmp_path, first, target, [netloc] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("target", ["ftp://inditales.com/x", "file:///etc/passwd",
                                    "javascript:alert(1)", "data:text/plain,hi",
                                    "gopher://inditales.com/"])
def test_redirect_to_a_non_http_scheme_is_refused_before_following(tmp_path, target):
    first = "https://inditales.com/post"
    calls = run(tmp_path, first, target)
    assert calls == [first]


def test_allowlist_itself_rejects_userinfo_and_other_schemes():
    allow = Allowlist.from_config()
    assert "user information" in allow.reason("https://inditales.com" + AT + "evil.example/")
    assert "user information" in allow.reason("https://evil.example" + AT + "inditales.com/")
    assert "scheme" in allow.reason("ftp://inditales.com/")


# ==== from test_api_hosts_golden.py


CONFIG = Path(__file__).resolve().parents[1] / "config" / "sources.json"


PLAIN_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")


# (host, path_prefix, min_delay_s, max_requests_per_run, robots_exempt): the three original
# entries (D37) plus the five D43 entries. Any change to this list is a reviewed config change.
GOLDEN = [
    ("www.wikidata.org", "/wiki/Special:EntityData/", 1.0, 30, True),
    ("overpass-api.de", "/api/interpreter", 30.0, 30, True),
    ("archive-api.open-meteo.com", "/v1/archive", 2.0, 30, True),
    ("en.wikipedia.org", "/w/api.php", 1.0, 30, True),
    ("en.wikivoyage.org", "/w/api.php", 1.0, 30, True),
    ("www.wikidata.org", "/w/api.php", 1.0, 30, True),
    ("query.wikidata.org", "/sparql", 1.0, 30, True),
    ("wikimedia.org", "/api/rest_v1/metrics/pageviews/per-article/", 1.0, 30, True),
]


def entries():
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    return doc["api_hosts"]["hosts"]


def test_api_hosts_match_the_golden_list_exactly():
    got = [(h["host"], h["path_prefix"], h["min_delay_s"], h["max_requests_per_run"],
            h["robots_exempt"]) for h in entries()]
    assert got == GOLDEN  # fails on any entry added, removed, reordered or changed
    assert len(got) == 8


def test_every_api_host_value_is_a_plain_hostname():
    for h in entries():
        host = h["host"]
        assert isinstance(host, str) and PLAIN_HOST.fullmatch(host), host
        for bad in ("/", "[", "]", "(", ")", ":", "@", " ", "http"):
            assert bad not in host, (host, bad)


# ==== from test_request_log_guard.py: Every network client must log. This guard fails if any code
# in src/ or scripts/ builds an


FIELDS = {"ts", "run_id", "host", "path", "kind", "status", "from_cache", "attempt", "source",
          "note"}


def unlogged_constructions(roots=("src", "scripts")) -> list[str]:
    bad = []
    for root in roots:
        for f in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == "HttpClient" and "request_log" not in {k.arg for k in node.keywords}:
                    bad.append(f"{f.relative_to(REPO_ROOT)}:{node.lineno}")
    return bad


def test_every_httpclient_construction_in_src_and_scripts_passes_a_request_log():
    assert unlogged_constructions() == []


def test_make_http_with_no_log_arguments_logs_a_fetch_with_all_fields(tmp_path, monkeypatch):
    log_path = tmp_path / "request_log.jsonl"
    monkeypatch.setattr(orchestrator, "REQUEST_LOG", log_path)
    cfg = CollectSettings("someone@example.org", "UA/0.1 (someone@example.org)", tmp_path / "c")

    def fake(method, url, headers, timeout):
        return TransportResponse(200, {}, "hello")

    http = orchestrator.make_http(cfg, transport=fake)
    http.fetch("https://sahapedia.org/some/page", {"q": "secret-looking-value"}, check_robots=False)
    http.fetch("https://sahapedia.org/some/page", {"q": "secret-looking-value"}, check_robots=False)
    rows = [json.loads(x) for x in log_path.read_text("utf-8").splitlines()]
    assert len(rows) == 2 and all(set(r) == FIELDS for r in rows)
    first, second = rows
    assert (first["host"], first["path"], first["kind"]) == ("sahapedia.org", "/some/page", "page")
    assert (first["status"], first["from_cache"], first["attempt"]) == (200, False, 1)
    assert (second["from_cache"], second["attempt"]) == (True, 0)
    raw = log_path.read_text("utf-8")
    assert "secret-looking-value" not in raw and "?" not in raw and "hello" not in raw
    assert first["run_id"] == second["run_id"] and first["ts"].endswith("Z")


# ==== from test_contact_privacy.py: CLAUDE.md rule 11: only values the user put in .env may be sent
# anywhere.


PRIVACY_ROOT = Path(__file__).resolve().parents[1]


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


AMBIENT = ("CONTACT_EMAIL", "USER_EMAIL", "EMAIL", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL")


def _clear(monkeypatch):
    for k in AMBIENT:
        monkeypatch.delenv(k, raising=False)


def test_missing_contact_email_is_an_error_not_a_guess(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USER_EMAIL", "someone@example.org")  # ambient identifiers are never used
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.org")
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    with pytest.raises(ConfigError, match="CONTACT_EMAIL"):
        load_collect_settings(env)


def test_user_agent_contains_only_the_env_file_address(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USER_EMAIL", "ambient@example.org")
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=owner@example.com\n")
    cfg = load_collect_settings(env)
    assert EMAIL.findall(cfg.user_agent) == ["owner@example.com"]


def test_http_client_sends_exactly_the_settings_user_agent(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "ambient@example.org")
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=owner@example.com\n")
    cfg = load_collect_settings(env)
    seen = []

    def transport(method, url, headers, timeout):
        seen.append(dict(headers))
        return TransportResponse(200, {}, "ok")

    client = HttpClient(DiskCache(tmp_path / "c"), cfg.user_agent, transport=transport,
                        sleep=lambda s: None, min_delay_s=0)
    client.fetch("https://example.org/x", check_robots=False)
    assert seen[0]["User-Agent"] == cfg.user_agent
    assert all(EMAIL.findall(v) in ([], ["owner@example.com"]) for v in seen[0].values())


def test_http_module_does_not_read_the_environment_or_contact_settings():
    src = (PRIVACY_ROOT / "src" / "common" / "http.py").read_text(encoding="utf-8")
    for forbidden in ("environ", "getenv", "dotenv", "CONTACT_EMAIL", "git config", "subprocess"):
        assert forbidden not in src, forbidden


def test_only_config_reads_contact_email():
    hits = [
        str(p.relative_to(PRIVACY_ROOT))
        for p in (PRIVACY_ROOT / "src").rglob("*.py")
        if "CONTACT_EMAIL" in p.read_text(encoding="utf-8")
    ]
    assert hits == [str(Path("src") / "common" / "schemas.py")]  # settings live there (D47)


def test_no_real_email_address_is_committed_in_source_or_docs():
    allowed = {"example.com", "example.org"}
    offenders = []
    for base in ("src", "scripts", "tests", "docs", "config"):
        for p in (PRIVACY_ROOT / base).rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".md", ".json", ".txt"}:
                continue
            for addr in EMAIL.findall(p.read_text(encoding="utf-8", errors="ignore")):
                if addr.split("@")[1] not in allowed:
                    offenders.append((str(p.relative_to(PRIVACY_ROOT)), addr))
    assert offenders == []


def test_env_example_and_readme_hold_only_placeholder_addresses():
    """.env.example is committed: its CONTACT_EMAIL must be a placeholder, never a real address."""
    allowed = {"example.com", "example.org"}
    for name in (".env.example", "README.md"):
        text = (PRIVACY_ROOT / name).read_text(encoding="utf-8")
        found = EMAIL.findall(text)
        assert all(a.split("@")[1] in allowed for a in found), name
    env_example = (PRIVACY_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "CONTACT_EMAIL= you@example.com" in env_example


# ==== from test_http_circuit.py


CIRCUIT_UA = "Test/0.1 (x@example.org)"


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
    http = HttpClient(DiskCache(tmp_path / "c"), CIRCUIT_UA, transport=world, clock=world.clock,
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


# ==== from test_facts.py


def flt():
    return HoldoutFilter(centres=[(9.6, 77.17)], radius_km=10.0)  # a fake hold-out centre


@pytest.mark.parametrize("name", [
    "Thekkady Bus Stand", "Kumily Bus Station", "Araku Railway Station", "Araku Valley Halt",
    "Brihadisvara Temple Stop", "Brihadeeswarar Nagar", "Thanjavur Junction", "Tanjore Airport",
    "Chikkamagaluru Bus Station", "Periyar Tiger Reserve Gate", "Periyar National Park Entry"])
def test_holdout_names_are_dropped(name):
    el = {"tags": {"name": name}, "lat": 12.0, "lon": 78.0}
    assert flt().reason(el) is not None


@pytest.mark.parametrize("name", ["Periyar University Bus Stop", "Munnar Bus Station",
                                  "Karaku Road", "Hampi Station"])
def test_other_names_are_kept(name):
    assert flt().reason({"tags": {"name": name}, "lat": 12.0, "lon": 78.0}) is None


def test_features_within_10_km_of_a_holdout_centre_are_dropped_and_others_kept():
    near = {"tags": {"name": "Plain Station"}, "lat": 9.65, "lon": 77.17}  # about 5.5 km away
    far = {"tags": {"name": "Plain Station"}, "lat": 9.8, "lon": 77.17}  # about 22 km away
    kept, dropped = flt().filter([near, far])
    assert kept == [far] and "10 km" in dropped[0][1]


def test_filter_uses_center_for_ways_and_alternate_name_tags():
    way = {"center": {"lat": 12.0, "lon": 78.0}, "tags": {"name": "Okay", "alt_name": "Kumily"}}
    assert flt().reason(way) == "name (alt_name)"
