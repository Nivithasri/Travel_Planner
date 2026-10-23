"""Tests for src/facts: RFC 9309 matcher and the facts fetchers (D47)."""
from __future__ import annotations

import json

import pytest

from src.collect.run import BlockDetected
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry
from src.facts import facts
from src.facts.facts import ApiHost, load_api_hosts, require_host

# ==== from test_robots_rfc9309.py


ROBOTS_UA = "TravelPlannerResearch/0.1 (academic IR project; x@example.org)"


BASE = "https://www.example.org"


def verdict(text, path):
    return facts.check(text, ROBOTS_UA, BASE + path)


def test_no_rules_or_no_match_means_allowed():
    assert verdict("", "/a").allowed
    assert verdict("User-agent: *\nDisallow: /private/\n", "/public").allowed
    assert verdict("User-agent: *\nDisallow:\n", "/anything").allowed  # empty Disallow


def test_longest_match_wins_over_a_shorter_opposite_rule():
    text = "User-agent: *\nDisallow: /a/\nAllow: /a/b/\n"
    assert not verdict(text, "/a/x").allowed
    v = verdict(text, "/a/b/x")
    assert v.allowed and v.rule.pattern == "/a/b/"


def test_allow_wins_a_tie_in_length():
    text = "User-agent: *\nDisallow: /same\nAllow: /same\n"
    v = verdict(text, "/same/page")
    assert v.allowed and v.rule.directive == "allow"


def test_star_wildcard_and_end_anchor():
    text = "User-agent: *\nDisallow: /*.pdf$\nAllow: /files/*.pdf$\n"
    assert not verdict(text, "/doc/report.pdf").allowed
    assert verdict(text, "/files/report.pdf").allowed
    assert verdict(text, "/doc/report.pdf?x=1").allowed  # `$` anchors at the end of path+query
    assert verdict("User-agent: *\nDisallow: /a*b\n", "/a-anything-b").allowed is False


def test_the_wikidata_style_pair_allows_a_json_file_but_not_other_special_pages():
    text = ("User-agent: *\nDisallow: /wiki/Special:\n"
            "Disallow: /wiki/Special:EntityData/\nAllow: /wiki/Special:EntityData/*.\n")
    v = verdict(text, "/wiki/Special:EntityData/Q1361.json")
    assert v.allowed and v.rule.pattern == "/wiki/Special:EntityData/*."
    assert len(v.matches) == 3
    assert not verdict(text, "/wiki/Special:EntityData/Q1361").allowed  # no dot: Allow can't match
    assert not verdict(text, "/wiki/Special:Search").allowed


def test_a_group_naming_our_product_token_beats_the_star_group():
    text = ("User-agent: *\nDisallow: /\n\n"
            "User-agent: TravelPlannerResearch\nAllow: /\nDisallow: /secret/\n")
    v = verdict(text, "/page")
    assert v.allowed and v.group == "travelplannerresearch"
    assert not verdict(text, "/secret/x").allowed


def test_consecutive_user_agent_lines_share_one_group_and_comments_are_ignored():
    text = "User-agent: Foo\nUser-agent: *  # all\nDisallow: /x # no\n"
    assert not verdict(text, "/x/y").allowed and verdict(text, "/z").allowed


def test_query_string_is_part_of_the_matched_target():
    text = "User-agent: *\nDisallow: /search?q=\n"
    assert not verdict(text, "/search?q=abc").allowed and verdict(text, "/search").allowed


# ==== from test_facts.py


FACTS_UA = "Test/0.1 (x@example.org)"


def entry(name="Munnar", qid="Q1", lat=10.09, lon=77.06):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=lat, lon=lon, wikidata_id=qid)


def stmt(value, rank="normal", p585=None):
    s = {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}
    if p585:
        s["qualifiers"] = {"P585": [{"datavalue": {"value": {"time": f"+{p585}T00:00:00Z"}}}]}
    return s


def client(tmp_path, outcome):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        return outcome(url) if callable(outcome) else outcome

    http = HttpClient(DiskCache(tmp_path / "c"), FACTS_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0)
    return http, calls


def test_p585_rule_preferred_beats_normal_then_latest_p585():
    old, new = stmt(1, p585="2011-00-00"), stmt(2, p585="2021-00-00")
    pref = stmt(3, rank="preferred", p585="2001-00-00")
    assert facts.select_statement([old, new, pref])[0] is pref  # preferred rank wins
    assert facts.select_statement([old, new])[0] is new  # latest P585
    assert facts.point_in_time(new) == "2021-00-00"


def test_p585_rule_ignores_deprecated_and_marks_undated_choices_ambiguous():
    dep = stmt(1, rank="deprecated", p585="2030-00-00")
    a, b = stmt(2), stmt(3)
    assert facts.select_statement([dep]) == (None, False)
    chosen, ambiguous = facts.select_statement([dep, a, b])
    assert chosen is a and ambiguous is True
    assert facts.select_statement([a])[1] is False
    assert facts.point_in_time(a) is None  # no P585: as_of is null


def test_extract_basic_reads_values_and_as_of():
    ent = {"lastrevid": 9, "modified": "2026-01-01T00:00:00Z", "claims": {
        "P625": [stmt({"latitude": 10.1, "longitude": 77.0})],
        "P2044": [stmt({"amount": "+1532", "unit": "http://www.wikidata.org/entity/Q11573"})],
        "P1082": [stmt({"amount": "+100", "unit": "1"}, p585="2011-00-00"),
                  stmt({"amount": "+120", "unit": "1"}, p585="2021-00-00")]}}
    out = facts.extract_basic(ent)
    assert out["coordinates"]["value"] == {"lat": 10.1, "lon": 77.0}
    assert out["elevation"]["value"]["amount"] == 1532.0 and out["elevation"]["as_of"] is None
    assert out["population"]["value"]["amount"] == 120.0
    assert out["population"]["as_of"] == "2021-00-00"
    assert facts.source_revision(ent) == {"lastrevid": 9, "modified": "2026-01-01T00:00:00Z"}


def test_wikidata_fetch_writes_an_envelope_and_a_cache_hit_makes_no_request(tmp_path):
    ent = json.dumps({"entities": {"Q1": {"lastrevid": 5, "modified": "2026-01-01T00:00:00Z",
                                          "claims": {}}}})
    http, calls = client(tmp_path, TransportResponse(200, {}, ent))
    host = load_api_hosts()["www.wikidata.org"]
    rep = facts.wikidata_fetch_place(http, host, entry(), tmp_path / "raw")
    env = json.loads((tmp_path / "raw/wikidata/munnar/entity.json").read_text("utf-8"))
    assert rep["wrote"] and len(calls) == 1
    for key in ("request_url", "retrieved_at", "status", "licence", "source_revision", "response"):
        assert key in env
    assert env["request_url"].endswith("/Special:EntityData/Q1.json") and env["status"] == 200
    again = facts.wikidata_fetch_place(http, host, entry(), tmp_path / "raw")
    assert len(calls) == 1 and again["wrote"] is False and again["from_cache"] is True


@pytest.mark.parametrize("fetcher", ["wikidata", "overpass"])
def test_429_stops_the_run_with_block_detected(tmp_path, fetcher):
    http, _ = client(tmp_path, TransportResponse(429, {}, ""))
    hosts = load_api_hosts()
    with pytest.raises(BlockDetected):
        if fetcher == "wikidata":
            facts.wikidata_fetch_place(http, hosts["www.wikidata.org"], entry(), tmp_path / "raw")
        else:
            facts.fetch_airports(http, hosts["overpass-api.de"], tmp_path / "raw")


def test_overpass_airport_timeout_is_reported_not_raised(tmp_path):
    from src.common.http import TransportError

    def boom(method, url, headers, timeout):
        raise TransportError("read timed out", "timeout")

    http = HttpClient(DiskCache(tmp_path / "c"), FACTS_UA, transport=boom, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    rep = facts.fetch_airports(http, load_api_hosts()["overpass-api.de"], tmp_path / "raw")
    assert "error" in rep and not (tmp_path / "raw").exists()


def test_overpass_406_is_also_a_block(tmp_path):
    http, _ = client(tmp_path, TransportResponse(406, {}, ""))
    with pytest.raises(BlockDetected):
        facts.fetch_airports(http, load_api_hosts()["overpass-api.de"], tmp_path / "raw")


def test_new_envelopes_add_manifest_lines_without_content_and_backfill_is_idempotent(tmp_path):
    from src.facts.facts import backfill_manifest
    ent = json.dumps({"entities": {"Q1": {"lastrevid": 5, "modified": "x", "claims": {}}}})
    http, _ = client(tmp_path, TransportResponse(200, {}, ent))
    host = load_api_hosts()["www.wikidata.org"]
    raw, manifest = tmp_path / "raw", tmp_path / "m.jsonl"
    facts.wikidata_fetch_place(http, host, entry(), raw, manifest=manifest)
    line = json.loads(manifest.read_text("utf-8").splitlines()[0])
    assert line["source"] == "wikidata" and line["place"] == "Munnar"
    assert line["url"].endswith("/Special:EntityData/Q1.json") and len(line["sha256"]) == 64
    assert "response" not in line and "body" not in line and line["bytes"] > 0
    assert backfill_manifest(raw, "wikidata", manifest) == 0  # already listed
    manifest.unlink()
    assert backfill_manifest(raw, "wikidata", manifest) == 1
    assert backfill_manifest(raw, "wikidata", manifest) == 0


@pytest.mark.parametrize("bad", ["[www.wikidata.org](https://www.wikidata.org)", "https://x.org",
                                 "x.org/path", "x.org:443", "x.org ", "-x.org", ""])
def test_api_host_values_must_be_plain_hostnames(tmp_path, bad):
    cfg = tmp_path / "sources.json"
    cfg.write_text(json.dumps({"api_hosts": {"hosts": [{
        "host": bad, "path_prefix": "/", "min_delay_s": 1, "max_requests_per_run": 1,
        "licence": "x", "terms_url": "x", "robots_exempt": True}]}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_api_hosts(cfg)


def test_shipped_api_hosts_are_plain_and_read_timeout_is_optional():
    hosts = load_api_hosts()
    assert all("/" not in h and "[" not in h and ":" not in h for h in hosts)
    assert hosts["www.wikidata.org"].read_timeout_s is None
    assert hosts["archive-api.open-meteo.com"].read_timeout_s == 60.0


def test_unknown_hosts_and_paths_are_refused():
    hosts = load_api_hosts()
    assert set(hosts) == {"www.wikidata.org", "overpass-api.de", "archive-api.open-meteo.com",
                          "en.wikipedia.org", "en.wikivoyage.org", "query.wikidata.org",
                          "wikimedia.org", "api.tavily.com"}  # D100 added
    assert hosts["www.wikidata.org"].path_prefixes == ("/wiki/Special:EntityData/", "/w/api.php")
    with pytest.raises(ValueError):
        require_host(hosts, "https://example.org/wiki/Special:EntityData/Q1.json")
    with pytest.raises(ValueError):
        require_host(hosts, "https://www.wikidata.org/w/index.php")  # not a listed path
    assert isinstance(require_host(hosts, "https://overpass-api.de/api/interpreter?data=x"),
                      ApiHost)
    assert hosts["overpass-api.de"].min_delay_s == 30.0
    assert hosts["archive-api.open-meteo.com"].min_delay_s == 2.0


def test_only_the_single_airport_bbox_query_exists():
    a = facts.airports_query()
    assert '"aeroway"="aerodrome"' in a and '["iata"]' in a and "(8.0,74.0,20.0,85.0)" in a
    assert a.count("aeroway") == 1  # one query, one bounding box
    for removed in ("stations_query", "fetch_stations", "classify", "RAIL_KM", "BUS_KM"):
        assert not hasattr(facts, removed)  # D40: per-place station queries removed


def test_validate_response_rejects_runtime_errors():
    assert facts.overpass_validate_response('{"elements": []}')
    timed_out = '{"elements": [], "remark": "runtime error: timed out"}'
    assert not facts.overpass_validate_response(timed_out)
    assert not facts.overpass_validate_response("<html>")


def test_request_params_use_archive_window_three_variables_and_elevation():
    p = facts.request_params(10.0885, 77.0595, 1532.0)
    assert p["start_date"] == "2015-01-01" and p["end_date"] == "2024-12-31"
    assert p["daily"] == "temperature_2m_max,temperature_2m_min,precipitation_sum"
    assert p["elevation"] == 1532.0
    assert "elevation" not in facts.request_params(10.0, 77.0)


def test_open_meteo_fetch_passes_params_and_writes_the_envelope(tmp_path):
    body = json.dumps({"daily": {"time": ["2015-01-01"], "precipitation_sum": [0.0]}})
    http, calls = client(tmp_path, TransportResponse(200, {}, body))
    host = load_api_hosts()["archive-api.open-meteo.com"]
    rep = facts.open_meteo_fetch_place(http, host, entry(), tmp_path / "raw", elevation_m=1532.0)
    assert "elevation=1532.0" in calls[0] and "start_date=2015-01-01" in calls[0]
    env = json.loads((tmp_path / "raw/open_meteo/munnar/archive_2015_2024.json").read_text("utf-8"))
    assert rep["days"] == 1 and env["params"]["elevation"] == 1532.0 and env["status"] == 200
    assert rep["check_ok"] is False and rep["status"] == 200  # 1 day is not 3653


def test_open_meteo_can_save_a_second_envelope_without_the_elevation_parameter(tmp_path):
    body = json.dumps({"daily": {"time": ["2015-01-01"], "precipitation_sum": [0.0]}})
    http, calls = client(tmp_path, TransportResponse(200, {}, body))
    host = load_api_hosts()["archive-api.open-meteo.com"]
    facts.open_meteo_fetch_place(http, host, entry(), tmp_path / "raw", None, None, None,
                          "archive_2015_2024_noelev.json")
    assert "elevation=" not in calls[0]
    assert (tmp_path / "raw/open_meteo/munnar/archive_2015_2024_noelev.json").exists()
    assert not (tmp_path / "raw/open_meteo/munnar/archive_2015_2024.json").exists()


def test_open_meteo_checks_3653_days_and_zero_nulls(tmp_path):
    days = [f"d{i}" for i in range(facts.EXPECTED_DAYS)]
    good = {"daily": {"time": days, "temperature_2m_max": [1.0] * len(days),
                      "temperature_2m_min": [0.0] * len(days),
                      "precipitation_sum": [0.0] * len(days)}}
    host = load_api_hosts()["archive-api.open-meteo.com"]
    http, _ = client(tmp_path, TransportResponse(200, {}, json.dumps(good)))
    assert facts.open_meteo_fetch_place(http, host, entry(), tmp_path / "raw")["check_ok"] is True
    good["daily"]["precipitation_sum"][5] = None
    http2, _ = client(tmp_path / "x", TransportResponse(200, {}, json.dumps(good)))
    rep = facts.open_meteo_fetch_place(http2, host, entry("Ooty", "Q2"), tmp_path / "raw")
    assert rep["nulls"] == 1 and rep["check_ok"] is False


def test_open_meteo_error_key_is_rejected_even_when_false(tmp_path):
    assert not facts.open_meteo_validate_response('{"error": false, "daily": {"time": []}}')
    assert not facts.open_meteo_validate_response('{"error": true, "reason": "x"}')
    assert facts.open_meteo_validate_response('{"daily": {"time": []}}')
