from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.collect.types import BlockDetected
from src.common.cache import DiskCache
from src.common.config import CollectSettings
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry
from src.facts import openmeteo, overpass, wikidata
from src.facts.holdout import HoldoutFilter
from src.facts.hosts import ApiHost, load_api_hosts, require_host

UA = "Test/0.1 (x@example.org)"
ROOT = Path(__file__).resolve().parents[1]


def entry(name="Munnar", qid="Q1", lat=10.09, lon=77.06):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=lat, lon=lon, wikidata_id=qid)


def flt():
    return HoldoutFilter(centres=[(9.6, 77.17)], radius_km=10.0)  # a fake hold-out centre


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

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    return http, calls


# ---- hold-out filter -------------------------------------------------------------------------
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


# ---- Wikidata --------------------------------------------------------------------------------
def test_p585_rule_preferred_beats_normal_then_latest_p585():
    old, new = stmt(1, p585="2011-00-00"), stmt(2, p585="2021-00-00")
    pref = stmt(3, rank="preferred", p585="2001-00-00")
    assert wikidata.select_statement([old, new, pref])[0] is pref  # preferred rank wins
    assert wikidata.select_statement([old, new])[0] is new  # latest P585
    assert wikidata.point_in_time(new) == "2021-00-00"


def test_p585_rule_ignores_deprecated_and_marks_undated_choices_ambiguous():
    dep = stmt(1, rank="deprecated", p585="2030-00-00")
    a, b = stmt(2), stmt(3)
    assert wikidata.select_statement([dep]) == (None, False)
    chosen, ambiguous = wikidata.select_statement([dep, a, b])
    assert chosen is a and ambiguous is True
    assert wikidata.select_statement([a])[1] is False
    assert wikidata.point_in_time(a) is None  # no P585: as_of is null


def test_extract_basic_reads_values_and_as_of():
    ent = {"lastrevid": 9, "modified": "2026-01-01T00:00:00Z", "claims": {
        "P625": [stmt({"latitude": 10.1, "longitude": 77.0})],
        "P2044": [stmt({"amount": "+1532", "unit": "http://www.wikidata.org/entity/Q11573"})],
        "P1082": [stmt({"amount": "+100", "unit": "1"}, p585="2011-00-00"),
                  stmt({"amount": "+120", "unit": "1"}, p585="2021-00-00")]}}
    out = wikidata.extract_basic(ent)
    assert out["coordinates"]["value"] == {"lat": 10.1, "lon": 77.0}
    assert out["elevation"]["value"]["amount"] == 1532.0 and out["elevation"]["as_of"] is None
    assert out["population"]["value"]["amount"] == 120.0
    assert out["population"]["as_of"] == "2021-00-00"
    assert wikidata.source_revision(ent) == {"lastrevid": 9, "modified": "2026-01-01T00:00:00Z"}


def test_wikidata_fetch_writes_an_envelope_and_a_cache_hit_makes_no_request(tmp_path):
    ent = json.dumps({"entities": {"Q1": {"lastrevid": 5, "modified": "2026-01-01T00:00:00Z",
                                          "claims": {}}}})
    http, calls = client(tmp_path, TransportResponse(200, {}, ent))
    host = load_api_hosts()["www.wikidata.org"]
    rep = wikidata.fetch_place(http, host, entry(), tmp_path / "raw")
    env = json.loads((tmp_path / "raw/wikidata/munnar/entity.json").read_text("utf-8"))
    assert rep["wrote"] and len(calls) == 1
    for key in ("request_url", "retrieved_at", "status", "licence", "source_revision", "response"):
        assert key in env
    assert env["request_url"].endswith("/Special:EntityData/Q1.json") and env["status"] == 200
    again = wikidata.fetch_place(http, host, entry(), tmp_path / "raw")
    assert len(calls) == 1 and again["wrote"] is False and again["from_cache"] is True


@pytest.mark.parametrize("fetcher", ["wikidata", "overpass"])
def test_429_stops_the_run_with_block_detected(tmp_path, fetcher):
    http, _ = client(tmp_path, TransportResponse(429, {}, ""))
    hosts = load_api_hosts()
    with pytest.raises(BlockDetected):
        if fetcher == "wikidata":
            wikidata.fetch_place(http, hosts["www.wikidata.org"], entry(), tmp_path / "raw")
        else:
            overpass.fetch_airports(http, hosts["overpass-api.de"], tmp_path / "raw")


def test_overpass_airport_timeout_is_reported_not_raised(tmp_path):
    from src.common.http import TransportError

    def boom(method, url, headers, timeout):
        raise TransportError("read timed out", "timeout")

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=boom, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    rep = overpass.fetch_airports(http, load_api_hosts()["overpass-api.de"], tmp_path / "raw")
    assert "error" in rep and not (tmp_path / "raw").exists()


def test_overpass_406_is_also_a_block(tmp_path):
    http, _ = client(tmp_path, TransportResponse(406, {}, ""))
    with pytest.raises(BlockDetected):
        overpass.fetch_airports(http, load_api_hosts()["overpass-api.de"], tmp_path / "raw")


# ---- raw manifest (no content) -----------------------------------------------------------------
def test_new_envelopes_add_manifest_lines_without_content_and_backfill_is_idempotent(tmp_path):
    from src.facts.envelope import backfill_manifest
    ent = json.dumps({"entities": {"Q1": {"lastrevid": 5, "modified": "x", "claims": {}}}})
    http, _ = client(tmp_path, TransportResponse(200, {}, ent))
    host = load_api_hosts()["www.wikidata.org"]
    raw, manifest = tmp_path / "raw", tmp_path / "m.jsonl"
    wikidata.fetch_place(http, host, entry(), raw, manifest=manifest)
    line = json.loads(manifest.read_text("utf-8").splitlines()[0])
    assert line["source"] == "wikidata" and line["place"] == "Munnar"
    assert line["url"].endswith("/Special:EntityData/Q1.json") and len(line["sha256"]) == 64
    assert "response" not in line and "body" not in line and line["bytes"] > 0
    assert backfill_manifest(raw, "wikidata", manifest) == 0  # already listed
    manifest.unlink()
    assert backfill_manifest(raw, "wikidata", manifest) == 1
    assert backfill_manifest(raw, "wikidata", manifest) == 0


# ---- host allowlist
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


# ---- host allowlist (continued)
def test_unknown_hosts_and_paths_are_refused():
    hosts = load_api_hosts()
    assert set(hosts) == {"www.wikidata.org", "overpass-api.de", "archive-api.open-meteo.com",
                          "en.wikipedia.org", "en.wikivoyage.org", "query.wikidata.org",
                          "wikimedia.org"}
    assert hosts["www.wikidata.org"].path_prefixes == ("/wiki/Special:EntityData/", "/w/api.php")
    with pytest.raises(ValueError):
        require_host(hosts, "https://example.org/wiki/Special:EntityData/Q1.json")
    with pytest.raises(ValueError):
        require_host(hosts, "https://www.wikidata.org/w/index.php")  # not a listed path
    assert isinstance(require_host(hosts, "https://overpass-api.de/api/interpreter?data=x"),
                      ApiHost)
    assert hosts["overpass-api.de"].min_delay_s == 30.0
    assert hosts["archive-api.open-meteo.com"].min_delay_s == 2.0


# ---- Overpass --------------------------------------------------------------------------------
def test_only_the_single_airport_bbox_query_exists():
    a = overpass.airports_query()
    assert '"aeroway"="aerodrome"' in a and '["iata"]' in a and "(8.0,74.0,20.0,85.0)" in a
    assert a.count("aeroway") == 1  # one query, one bounding box
    for removed in ("stations_query", "fetch_stations", "classify", "RAIL_KM", "BUS_KM"):
        assert not hasattr(overpass, removed)  # D40: per-place station queries removed


def test_validate_response_rejects_runtime_errors():
    assert overpass.validate_response('{"elements": []}')
    assert not overpass.validate_response('{"elements": [], "remark": "runtime error: timed out"}')
    assert not overpass.validate_response("<html>")


def test_nearest_airport_is_computed_in_code_and_holdout_airports_are_dropped():
    airports = [
        {"lat": 10.15, "lon": 76.40, "tags": {"name": "Cochin International", "iata": "COK"}},
        {"lat": 9.65, "lon": 77.17, "tags": {"name": "Plain Field", "iata": "XXX"}},  # near centre
        {"lat": 11.0, "lon": 77.0, "tags": {"name": "Tanjore Air Station", "iata": "TJV"}},
    ]
    best = overpass.nearest_airport(airports, 10.09, 77.06, flt())
    assert best[0]["tags"]["iata"] == "COK" and 60 < best[1] < 80
    assert overpass.nearest(airports, 10.09, 77.06, max_km=5) is None


# ---- Open-Meteo ------------------------------------------------------------------------------
def test_rainy_day_rule_boundary():
    assert overpass is not None and openmeteo.RAINY_DAY_MM == 2.5
    assert openmeteo.is_rainy(2.5) and not openmeteo.is_rainy(2.4) and not openmeteo.is_rainy(None)


def test_request_params_use_archive_window_three_variables_and_elevation():
    p = openmeteo.request_params(10.0885, 77.0595, 1532.0)
    assert p["start_date"] == "2015-01-01" and p["end_date"] == "2024-12-31"
    assert p["daily"] == "temperature_2m_max,temperature_2m_min,precipitation_sum"
    assert p["elevation"] == 1532.0
    assert "elevation" not in openmeteo.request_params(10.0, 77.0)


def test_monthly_normals_are_derived_from_daily_values_without_filling_gaps():
    daily = {"time": ["2023-06-01", "2023-06-02", "2023-06-03", "2024-06-01"],
             "temperature_2m_max": [20.0, 22.0, None, 24.0],
             "temperature_2m_min": [10.0, 12.0, None, 14.0],
             "precipitation_sum": [2.5, 1.0, None, 10.0]}
    june = openmeteo.monthly_normals(daily)[6]
    assert june["rainfall_mm"] == pytest.approx((3.5 + 10.0) / 2)  # mean of yearly monthly sums
    assert june["rainy_days"] == pytest.approx((1 + 1) / 2)
    assert june["temp_c"] == pytest.approx((15.0 + 17.0 + 19.0) / 3)
    assert june["derived"] is True and 7 not in openmeteo.monthly_normals(daily)


def test_open_meteo_fetch_passes_params_and_writes_the_envelope(tmp_path):
    body = json.dumps({"daily": {"time": ["2015-01-01"], "precipitation_sum": [0.0]}})
    http, calls = client(tmp_path, TransportResponse(200, {}, body))
    host = load_api_hosts()["archive-api.open-meteo.com"]
    rep = openmeteo.fetch_place(http, host, entry(), tmp_path / "raw", elevation_m=1532.0)
    assert "elevation=1532.0" in calls[0] and "start_date=2015-01-01" in calls[0]
    env = json.loads((tmp_path / "raw/open_meteo/munnar/archive_2015_2024.json").read_text("utf-8"))
    assert rep["days"] == 1 and env["params"]["elevation"] == 1532.0 and env["status"] == 200
    assert rep["check_ok"] is False and rep["status"] == 200  # 1 day is not 3653


def test_open_meteo_can_save_a_second_envelope_without_the_elevation_parameter(tmp_path):
    body = json.dumps({"daily": {"time": ["2015-01-01"], "precipitation_sum": [0.0]}})
    http, calls = client(tmp_path, TransportResponse(200, {}, body))
    host = load_api_hosts()["archive-api.open-meteo.com"]
    openmeteo.fetch_place(http, host, entry(), tmp_path / "raw", None, None, None,
                          "archive_2015_2024_noelev.json")
    assert "elevation=" not in calls[0]
    assert (tmp_path / "raw/open_meteo/munnar/archive_2015_2024_noelev.json").exists()
    assert not (tmp_path / "raw/open_meteo/munnar/archive_2015_2024.json").exists()


def test_open_meteo_checks_3653_days_and_zero_nulls(tmp_path):
    days = [f"d{i}" for i in range(openmeteo.EXPECTED_DAYS)]
    good = {"daily": {"time": days, "temperature_2m_max": [1.0] * len(days),
                      "temperature_2m_min": [0.0] * len(days),
                      "precipitation_sum": [0.0] * len(days)}}
    host = load_api_hosts()["archive-api.open-meteo.com"]
    http, _ = client(tmp_path, TransportResponse(200, {}, json.dumps(good)))
    assert openmeteo.fetch_place(http, host, entry(), tmp_path / "raw")["check_ok"] is True
    good["daily"]["precipitation_sum"][5] = None
    http2, _ = client(tmp_path / "x", TransportResponse(200, {}, json.dumps(good)))
    rep = openmeteo.fetch_place(http2, host, entry("Ooty", "Q2"), tmp_path / "raw")
    assert rep["nulls"] == 1 and rep["check_ok"] is False


def test_open_meteo_error_key_is_rejected_even_when_false(tmp_path):
    assert not openmeteo.validate_response('{"error": false, "daily": {"time": []}}')
    assert not openmeteo.validate_response('{"error": true, "reason": "x"}')
    assert openmeteo.validate_response('{"daily": {"time": []}}')


# ---- runner ----------------------------------------------------------------------------------
def load_script():
    path = ROOT / "scripts" / "collect_facts.py"
    spec = importlib.util.spec_from_file_location("collect_facts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_refuses_a_holdout_with_exit_code_2(monkeypatch, capsys):
    mod = load_script()
    cfg = CollectSettings("x@example.org", UA, Path("."))
    monkeypatch.setattr(mod, "load_collect_settings", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["collect_facts.py", "--source", "wikidata",
                                      "--places", "Thekkady", "--execute"])
    assert mod.main() == 2
    assert "hold-out" in capsys.readouterr().out
