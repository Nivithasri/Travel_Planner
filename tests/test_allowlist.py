from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collect.orchestrator import make_http
from src.collect.web import host_key as web_host_key
from src.common.allowlist import Allowlist, host_key
from src.common.cache import DiskCache
from src.common.config import CollectSettings
from src.common.http import FetchError, HttpClient, RequestBudgetExhausted, TransportResponse

UA = "Test/0.1 (x@example.org)"


@pytest.fixture(scope="module")
def shipped():
    return Allowlist.from_config()


# ---- the matching rule for configured web domains ------------------------------------------
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


# ---- api_hosts: host plus path prefix ------------------------------------------------------
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


# ---- enforcement inside HttpClient ---------------------------------------------------------
def client(tmp_path, allow, **kw):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        return TransportResponse(200, {}, "{}")

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0, allowlist=allow, **kw)
    return http, calls


def test_a_refused_url_makes_no_request_and_is_not_counted(tmp_path, shipped):
    http, calls = client(tmp_path, shipped)
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://en.wikipedia.org/w/index.php", check_robots=False)
    assert calls == [] and http.requests_made == 0 and http.refusals == 1
    http.fetch("https://en.wikipedia.org/w/api.php", {"action": "query"}, check_robots=False)
    assert len(calls) == 1


def test_a_sitemap_page_on_another_host_is_refused(tmp_path):
    allow = Allowlist(web_domains=["www.example.org"])
    http, calls = client(tmp_path, allow)
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://elsewhere.net/munnar", check_robots=False)
    assert http.fetch("https://example.org/munnar", check_robots=False).ok
    assert calls == ["https://example.org/munnar"]


def test_per_host_cap_is_a_hard_stop(tmp_path):
    allow = Allowlist(api_prefixes={"api.example.org": ["/v1/"]}, api_caps={"api.example.org": 2})
    http, calls = client(tmp_path, allow, host_caps=allow.caps)
    http.fetch("https://api.example.org/v1/a", check_robots=False)
    http.fetch("https://api.example.org/v1/b", check_robots=False)
    with pytest.raises(RequestBudgetExhausted):
        http.fetch("https://api.example.org/v1/c", check_robots=False)
    assert len(calls) == 2


def test_make_http_clients_carry_the_shipped_allowlist(tmp_path):
    cfg = CollectSettings("x@example.org", UA, Path(tmp_path) / "cache")
    http = make_http(cfg, request_log=None)
    assert http.allowlist is not None and http.host_caps["en.wikipedia.org"] == 30
    assert http.host_delays["en.wikipedia.org"] == 1.0
    with pytest.raises(FetchError, match="not allowlisted"):
        http.fetch("https://en.wikipedia.org/w/index.php", check_robots=False)


def test_wiki_collector_and_resolver_paths_are_the_listed_ones(shipped):
    from src.collect.wiki import WIKIS
    from src.places import wikidata as wd
    from src.places.wv_lists import WIKIVOYAGE_API

    for cfg in WIKIS.values():
        assert shipped.reason(cfg["api"]) is None
    for url in (wd.WIKIDATA_API, wd.SPARQL_URL, WIKIVOYAGE_API,
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
