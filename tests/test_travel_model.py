"""Nodes, hill flag, hub rule, fit and matrix (Phase 5; D76 to D83)."""
from __future__ import annotations

import json
import math

import pytest

from src.facts.facts import haversine_km
from src.places.places import Registry
from src.tools import travel_model as tm
from src.tools.routing import AIRPORTS_RAW_PATH, load_airports

PARAMS = {"hill_elevation_m": 1000, "hill_overrides": ["Kodagu", "Wayanad", "Srisailam"]}
needs_raw = pytest.mark.skipif(
    not (AIRPORTS_RAW_PATH.exists() and tm.WIKIDATA_RAW.exists()),
    reason="data/raw (Overpass, Wikidata) is gitignored and absent on a fresh clone")

FITTED = {"band_km": 100.0, "road_factor": 1.2, "speed_short_kmh": 50.0, "speed_long_kmh": 100.0,
          "ghat_surcharge_h": 1.0}


def real_nodes():
    registry = Registry.load(tm.REGISTRY_PATH)
    nodes, report = tm.build_nodes(registry, load_airports(), tm.load_params())
    return registry, nodes, report


def test_haversine_known_pair():
    # Chennai to Bengaluru is about 290 km in a straight line
    assert 280 < haversine_km(13.0827, 80.2707, 12.9716, 77.5946) < 300


def test_hill_flag_by_elevation_and_override():
    assert tm.hill_flag("Ooty", 2240, PARAMS)[0]
    assert not tm.hill_flag("Bengaluru", 920, PARAMS)[0]
    assert not tm.hill_flag("Mysuru", 770, PARAMS)[0]
    assert tm.hill_flag("Wayanad", 700, PARAMS)[0]
    assert tm.hill_flag("Srisailam", 409, PARAMS)[0]
    assert tm.hill_flag("Kodagu", None, PARAMS)[0]
    assert not tm.hill_flag("Auroville", None, PARAMS)[0]
    assert tm.hill_flag("X", 1000, PARAMS)[0] and not tm.hill_flag("X", 999.9, PARAMS)[0]


@needs_raw
def test_real_hill_nodes_are_exactly_the_six_and_not_plateau_cities():
    _, nodes, report = real_nodes()
    hill = {n["name"] for n in nodes if n["hill"]}
    assert hill == {"Ooty", "Kodaikanal", "Munnar", "Kodagu", "Wayanad", "Srisailam"}
    assert {n for n, _ in report["hill_nodes"]} == hill
    for name in ("Bengaluru", "Mysuru", "Hampi"):
        assert not next(n for n in nodes if n["name"] == name)["hill"]
    assert not any(n["hill"] for n in nodes if n["kind"] == "hub")


@needs_raw
def test_nodes_exclude_holdouts_and_have_22_hubs():
    registry, nodes, report = real_nodes()
    held = {e.name for e in registry.entries if e.holdout}
    assert held == {"Thanjavur", "Thekkady", "Chikmagalur", "Araku Valley"}
    assert not held & {n["name"] for n in nodes}
    assert report["destinations"] == 26 and len(report["hubs"]) == 22 and len(nodes) == 48
    assert len({n["id"] for n in nodes}) == 48
    assert {"IXX", "BEP"} <= set(report["excluded_military"])
    assert not {"IXX", "BEP"} & set(report["hubs"])


def test_hub_rule_boundary_and_military_exclusion():
    air = {"AAA": {"lat": 10.0, "lon": 77.0, "military": False},
           "BBB": {"lat": 10.0, "lon": 77.0, "military": True}}
    # a destination exactly 24.9 km away blocks the hub, 25.1 km does not
    near = (10.0 + 24.9 / 111.19, 77.0)
    far = (10.0 + 25.1 / 111.19, 77.0)
    assert tm.select_hubs(air, [near], [], 25) == []
    assert tm.select_hubs(air, [far], [], 25) == ["AAA"]  # BBB is military
    assert tm.select_hubs(air, [far], [near], 25) == []  # a hold-out blocks too


def test_hub_alias_strips_airport_words_and_commas():
    assert tm.hub_alias("Coimbatore International Airport") == "Coimbatore"
    assert tm.hub_alias("Salem Airport") == "Salem"
    assert tm.hub_alias("Basant Nagar Airstrip") == "Basant Nagar"
    kjb = "Uyyalawada Narasimha Reddy Airport ,   Kurnool"
    assert tm.hub_alias(kjb) == "Uyyalawada Narasimha Reddy"
    assert tm.hub_aliases(kjb, "KJB") == ["KJB", "Kurnool", "Uyyalawada Narasimha Reddy"]
    assert tm.clean_name(kjb) == "Uyyalawada Narasimha Reddy Airport, Kurnool"


def test_clean_aliases_drops_sentences_but_keeps_real_aliases():
    junk = "Munnar was a town that was formed before the formation of Kerala. It was created."
    got = tm.clean_aliases(["Kashmir of South India", junk, "Cochin", "  Cochin ",
                            "Hampi group of monuments (except those under ASI)"])
    assert got == ["Cochin", "Kashmir of South India"]
    assert tm.clean_aliases(["a b c d e", "a b c d e f"]) == ["a b c d e"]


def rows_for(refs, n_hill=0, split="fit"):
    return [{"ref_km": km, "ref_h": h, "gc_km": gc, "n_hill": n_hill, "hill": n_hill > 0,
             "split": split, "a": "x", "b": "y"} for gc, km, h in refs]


def synthetic_rows():
    # factor 1.2; short routes at 50 km/h; long routes: 100 km at 50, rest at 100 km/h
    short = rows_for([(50, 60, 1.2), (25, 30, 0.6), (75, 90, 1.8)])
    long_ = rows_for([(250, 300, 2.0 + 2.0), (500, 600, 2.0 + 5.0), (200, 240, 2.0 + 1.4)])
    hill = rows_for([(50, 60, 1.2 + 1.0), (75, 90, 1.8 + 1.0)], n_hill=1)
    return short + long_ + hill


def test_fit_recovers_known_parameters():
    rows = synthetic_rows()
    f = tm.fit(rows, band_km=100.0)
    assert f["road_factor"] == 1.2 and f["speed_short_kmh"] == 50.0
    assert f["speed_long_kmh"] == 100.0 and f["ghat_surcharge_h"] == 1.0
    assert (f["n_fit_short"], f["n_fit_long"], f["n_fit_hill"]) == (3, 3, 2)


def test_fit_uses_the_median_and_ignores_validation_routes():
    rows = synthetic_rows()
    rows[0] = {**rows[0], "ref_km": 600}  # one wrong reference among the plain routes
    rows += rows_for([(100, 9999, 1.0)], split="validate")
    f = tm.fit(rows, band_km=100.0)
    assert f["road_factor"] == 1.2


def test_fit_clamps_a_negative_surcharge_to_zero():
    rows = [r for r in synthetic_rows() if r["n_hill"] == 0]
    rows += rows_for([(50, 60, 0.5), (75, 90, 0.8)], n_hill=1)  # faster than the plain model
    assert tm.fit(rows, band_km=100.0)["ghat_surcharge_h"] == 0.0


def test_fit_needs_every_group():
    rows = synthetic_rows()
    for drop in (lambda r: r["n_hill"] > 0, lambda r: r["ref_km"] > 100 and r["n_hill"] == 0,
                 lambda r: r["ref_km"] <= 100 and r["n_hill"] == 0):
        with pytest.raises(ValueError):
            tm.fit([r for r in rows if not drop(r)], band_km=100.0)


def test_plain_hours_is_continuous_and_monotone_at_the_band_edge():
    below = tm.plain_hours(99.999, FITTED)
    above = tm.plain_hours(100.001, FITTED)
    assert above > below and above - below < 0.001
    assert tm.plain_hours(200, FITTED) == pytest.approx(100 / 50 + 100 / 100)
    ks = [k / 3 for k in range(1, 900)]
    hs = [tm.plain_hours(k, FITTED) for k in ks]
    assert hs == sorted(hs)


def test_estimate_surcharge_is_fixed_per_hill_endpoint_not_scaled_by_distance():
    h0 = tm.estimate(10.0, 77.0, 11.0, 77.0, 0, FITTED)[1]
    h1 = tm.estimate(10.0, 77.0, 11.0, 77.0, 1, FITTED)[1]
    h2 = tm.estimate(10.0, 77.0, 11.0, 77.0, 2, FITTED)[1]
    assert math.isclose(h1 - h0, 1.0) and math.isclose(h2 - h0, 2.0)
    far0 = tm.estimate(10.0, 77.0, 16.0, 77.0, 0, FITTED)[1]
    far1 = tm.estimate(10.0, 77.0, 16.0, 77.0, 1, FITTED)[1]
    assert math.isclose(far1 - far0, 1.0)  # the same surcharge on a much longer trip
    assert tm.estimate(10.0, 77.0, 10.001, 77.0, 0, FITTED)[1] == 0.25


def test_hill_endpoints_counts_each_end():
    a, b = {"hill": True}, {"hill": False}
    assert tm.hill_endpoints(a, b) == 1 and tm.hill_endpoints(a, a) == 2
    assert tm.hill_endpoints(b, b) == 0


def test_error_summary_and_flags():
    errs = [{"hours": 3.0, "ref_h": 2.0, "h_err_pct": 50.0, "km_err_pct": 10.0,
             "split": "validate", "a": "p", "b": "q"},
            {"hours": 1.0, "ref_h": 2.0, "h_err_pct": -50.0, "km_err_pct": -10.0,
             "split": "validate", "a": "r", "b": "s"},
            {"hours": 2.2, "ref_h": 2.0, "h_err_pct": 10.0, "km_err_pct": 0.0,
             "split": "fit", "a": "t", "b": "u"}]
    s = tm.error_summary(errs[:2])
    assert s == {"n": 2, "mae_h": 1.0, "mape_h_pct": 50.0, "mape_km_pct": 10.0}
    assert tm.error_summary([]) == {"n": 0}
    assert tm.flag_large_errors(errs) == ["p-q +50.0%", "r-s -50.0%"]
    assert tm.flag_large_errors(errs, split="fit") == []
    assert tm.flag_large_errors([{**errs[0], "h_err_pct": 25.0}]) == []  # exactly 25% is not over


def test_route_errors_match_the_model():
    rows = [{"a": "p", "b": "q", "split": "fit", "hill": True, "n_hill": 1, "gc_km": 100.0,
             "ref_km": 130.0, "ref_h": 3.0}]
    e = tm.route_errors(rows, FITTED)[0]
    km = 100 * 1.2
    assert e["km"] == round(km, 1)
    assert e["hours"] == round(100 / 50 + (km - 100) / 100 + 1.0, 2) == round(
        tm.plain_hours(km, FITTED) + 1.0, 2)


def test_leave_one_out_refits_without_each_route_and_skips_emptied_groups():
    loo = tm.leave_one_out(synthetic_rows(), 100.0)
    # every fit route is skipped only if its group would be empty; hill group has 2 so none is
    assert loo["n"] == 8 and loo["mape_h_pct"] < 1.0  # exact synthetic data is predicted exactly
    single_hill = [r for r in synthetic_rows() if r["n_hill"] == 0] + rows_for(
        [(50, 60, 2.2)], n_hill=1)
    assert tm.leave_one_out(single_hill, 100.0)["n"] == 6  # the lone hill route cannot be left out


@needs_raw
def test_matrix_is_complete_in_keys_and_deterministic():
    _, nodes, _ = real_nodes()
    fitted = tm.load_params()["fitted"]
    h1, k1 = tm.matrix_pairs(nodes, fitted)
    h2, _ = tm.matrix_pairs(list(nodes), fitted)
    assert h1 == h2 and len(h1) == len(nodes) * (len(nodes) - 1) // 2
    assert list(h1) == sorted(h1)
    assert all(a < b for a, b in (key.split("|") for key in h1))
    assert all(v >= 0.25 for v in h1.values()) and set(h1) == set(k1)


def test_triangle_violations_counts_not_repairs():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    hours = {"a|b": 1.0, "b|c": 1.0, "a|c": 5.0}
    assert tm.triangle_violations(nodes, hours) == 1
    assert tm.triangle_violations(nodes, {"a|b": 1.0, "b|c": 1.0, "a|c": 2.0}) == 0


def test_airport_aliases_map_non_hub_civil_airports_to_the_nearest_destination():
    nodes = [{"id": "kochi", "kind": "destination", "lat": 9.97, "lon": 76.28},
             {"id": "munnar", "kind": "destination", "lat": 10.09, "lon": 77.06},
             {"id": "hub-trv", "kind": "hub", "iata": "TRV", "lat": 8.48, "lon": 76.92}]
    air = {"COK": {"lat": 10.15, "lon": 76.40, "military": False},
           "TRV": {"lat": 8.48, "lon": 76.92, "military": False},
           "FAR": {"lat": 14.0, "lon": 79.0, "military": False},
           "MIL": {"lat": 9.98, "lon": 76.28, "military": True}}
    assert tm.airport_aliases(nodes, air, 25) == {"COK": "kochi"}  # TRV is a hub, FAR too far


def test_calibration_file_is_unverified_complete_and_consistent_with_hill_flags():
    doc = json.loads(tm.ROUTES_PATH.read_text(encoding="utf-8"))
    assert doc["status"] == "UNVERIFIED" and len(doc["routes"]) == 30
    assert sum(r["split"] == "validate" for r in doc["routes"]) == 6
    held = {"thanjavur", "thekkady", "chikmagalur", "araku-valley"}
    ends = {r["a"] for r in doc["routes"]} | {r["b"] for r in doc["routes"]}
    assert not held & ends
    if not (AIRPORTS_RAW_PATH.exists() and tm.WIKIDATA_RAW.exists()):
        pytest.skip("node coordinates need the gitignored data/raw (fresh clone)")
    _, nodes, _ = real_nodes()
    rows = tm.route_rows(doc["routes"], {n["id"]: n for n in nodes})
    assert sum(r["hill"] for r in rows) == 9
    assert sum(r["hill"] and r["split"] == "validate" for r in rows) == 3
    assert all(r["n_hill"] <= 1 for r in rows)
    for r in rows:  # a reference distance should not be wildly below the straight line
        assert r["ref_km"] >= 0.95 * r["gc_km"], r
