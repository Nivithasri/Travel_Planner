"""Nodes, hill flag, hub rule, fit and matrix (Phase 5; D76 to D79)."""
from __future__ import annotations

import json
import math

import pytest

from src.facts.facts import haversine_km
from src.places.places import Registry
from src.tools import travel_model as tm
from src.tools.routing import load_airports

PARAMS = {"hill_elevation_m": 1000, "hill_overrides": ["Kodagu", "Wayanad", "Srisailam"]}


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


def test_real_hill_nodes_are_exactly_the_six_and_not_plateau_cities():
    _, nodes, report = real_nodes()
    hill = {n["name"] for n in nodes if n["hill"]}
    assert hill == {"Ooty", "Kodaikanal", "Munnar", "Kodagu", "Wayanad", "Srisailam"}
    assert {n for n, _ in report["hill_nodes"]} == hill
    for name in ("Bengaluru", "Mysuru", "Hampi"):
        assert not next(n for n in nodes if n["name"] == name)["hill"]
    assert not any(n["hill"] for n in nodes if n["kind"] == "hub")


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


def test_hub_alias_strips_airport_words():
    assert tm.hub_alias("Coimbatore International Airport") == "Coimbatore"
    assert tm.hub_alias("Salem Airport") == "Salem"
    assert tm.hub_alias("Basant Nagar Airstrip") == "Basant Nagar"


def rows_for(refs, hill=False, split="fit"):
    return [{"ref_km": km, "ref_h": h, "gc_km": gc, "hill": hill, "split": split}
            for gc, km, h in refs]


def test_fit_recovers_known_factor_and_speed():
    plain = rows_for([(100, 120, 2.0), (200, 240, 4.0), (50, 60, 1.0)])  # factor 1.2, speed 60
    hill = rows_for([(100, 150, 5.0), (80, 120, 4.0)], hill=True)  # factor 1.5, speed 30
    f = tm.fit(plain + hill)
    assert f["road_factor"] == 1.2 and f["speed_kmh"] == 60.0
    assert f["hill_road_factor"] == 1.5 and f["hill_speed_kmh"] == 30.0


def test_fit_uses_the_median_and_ignores_validation_routes():
    rows = rows_for([(100, 120, 2.0), (100, 120, 2.0), (100, 300, 2.0)])  # one wrong reference
    rows += rows_for([(100, 999, 2.0)], split="validate")
    rows += rows_for([(100, 150, 5.0)], hill=True)
    assert tm.fit(rows)["road_factor"] == 1.2


def test_fit_needs_both_edge_classes():
    with pytest.raises(ValueError):
        tm.fit(rows_for([(100, 120, 2.0)]))


def test_estimate_hill_edge_is_slower_and_floored():
    fitted = {"road_factor": 1.2, "speed_kmh": 60.0,
              "hill_road_factor": 1.5, "hill_speed_kmh": 30.0}
    km_p, h_p = tm.estimate(10.0, 77.0, 11.0, 77.0, False, fitted)
    km_h, h_h = tm.estimate(10.0, 77.0, 11.0, 77.0, True, fitted)
    assert h_h > h_p and km_h > km_p
    assert math.isclose(h_p, km_p / 60.0)
    assert tm.estimate(10.0, 77.0, 10.001, 77.0, False, fitted)[1] == 0.25


def test_matrix_is_complete_symmetric_in_keys_and_deterministic():
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


def test_calibration_file_is_unverified_complete_and_consistent_with_hill_flags():
    doc = json.loads(tm.ROUTES_PATH.read_text(encoding="utf-8"))
    assert doc["status"] == "UNVERIFIED" and len(doc["routes"]) == 30
    assert sum(r["split"] == "validate" for r in doc["routes"]) == 6
    _, nodes, _ = real_nodes()
    by_id = {n["id"]: n for n in nodes}
    rows = tm.route_rows(doc["routes"], by_id)  # raises KeyError when an endpoint is not a node
    assert sum(r["hill"] for r in rows) == 9
    assert sum(r["hill"] and r["split"] == "validate" for r in rows) == 3
    held = {"thanjavur", "thekkady", "chikmagalur", "araku-valley"}
    assert not held & ({r["a"] for r in rows} | {r["b"] for r in rows})
    for r in rows:  # a reference distance should not be wildly below the straight line
        assert r["ref_km"] >= 0.95 * r["gc_km"], r
