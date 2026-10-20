"""O13: the curated airport allowlist, the dedupe rule and nearest_airport (D79)."""
from __future__ import annotations

import json

import pytest

from src.tools.routing import (
    AIRPORTS_ALLOWLIST_PATH,
    AIRPORTS_RAW_PATH,
    dedupe_airports,
    element_point,
    load_airports,
    load_airports_allowlist,
    nearest_airport,
)

IN_REGION = ("BEP BLR BPM CCJ CDP CJB CNN COK GBI HBX HYD IXE IXG IXM IXX KJB MAA MYQ NVY PNY PUT "
             "RJA RMD RQY SXV TCR TIR TRV TRZ VDY VGA VTZ WGC").split()
OUT_OF_REGION = "ACJ IXU JAF JGB KLH LTU NDC PYB SAG SSE TRR".split()

needs_raw = pytest.mark.skipif(
    not AIRPORTS_RAW_PATH.exists(),
    reason="data/raw/overpass is gitignored and absent on a fresh clone")


def raw_elements():
    raw = json.loads(AIRPORTS_RAW_PATH.read_text(encoding="utf-8"))
    return raw["response"]["elements"]


def el(kind, osm_id, iata, lat=10.0, lon=77.0):
    return {"type": kind, "id": osm_id, "lat": lat, "lon": lon,
            "tags": {"iata": iata, "name": iata}}


def test_allowlist_is_the_33_in_region_codes():
    codes = load_airports_allowlist()
    assert len(codes) == 33 and len(set(codes)) == 33
    assert sorted(codes) == sorted(IN_REGION)
    assert not set(codes) & set(OUT_OF_REGION)
    doc = json.loads(AIRPORTS_ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert sorted(doc["out_of_region_excluded"]) == sorted(OUT_OF_REGION)


@needs_raw
def test_every_allowlist_code_is_in_the_saved_overpass_file():
    present = {(e.get("tags") or {}).get("iata") for e in raw_elements()}
    assert set(IN_REGION) <= present


@needs_raw
def test_load_airports_has_one_entry_per_code_and_no_out_of_region():
    airports = load_airports()
    assert sorted(airports) == sorted(IN_REGION)
    assert not set(airports) & set(OUT_OF_REGION)
    assert "TJV" not in airports  # Thanjavur Air Force Station is a hold-out place


def test_dedupe_prefers_relation_over_way_over_node():
    out = dedupe_airports([el("node", 1, "XXX"), el("way", 2, "XXX"), el("relation", 3, "XXX")],
                          ["XXX"])
    assert (out["XXX"]["osm_type"], out["XXX"]["osm_id"]) == ("relation", 3)
    out = dedupe_airports([el("node", 1, "XXX"), el("way", 2, "XXX")], ["XXX"])
    assert out["XXX"]["osm_type"] == "way"


def test_dedupe_ties_go_to_the_lowest_osm_id():
    out = dedupe_airports([el("way", 20, "XXX"), el("way", 5, "XXX"), el("way", 9, "XXX")], ["XXX"])
    assert out["XXX"]["osm_id"] == 5


def test_dedupe_drops_codes_outside_the_allowlist_and_pointless_elements():
    elements = [el("node", 1, "AAA"), el("node", 2, "BBB"),
                {"type": "node", "id": 3, "tags": {"iata": "AAA"}}]
    assert list(dedupe_airports(elements, ["AAA"])) == ["AAA"]


@needs_raw
def test_the_duplicated_codes_resolve_to_one_entry_each():
    airports = load_airports()
    by_iata = {}
    for e in raw_elements():
        by_iata.setdefault((e.get("tags") or {}).get("iata"), []).append(e["type"])
    for iata in ("HYD", "VDY"):
        assert len(by_iata[iata]) == 2  # the raw file really has two elements
        assert iata in airports
    assert airports["HYD"]["osm_type"] == "relation"  # relation beats way


def test_element_point_uses_center_for_ways_and_relations():
    assert element_point({"type": "way", "center": {"lat": 1.5, "lon": 2.5}}) == (1.5, 2.5)
    assert element_point({"type": "node", "lat": 3.0, "lon": 4.0}) == (3.0, 4.0)
    assert element_point({"type": "way"}) is None


@needs_raw
def test_nearest_airport_never_returns_an_out_of_region_code():
    airports = load_airports()
    # Rameswaram is about 100 km from Jaffna (JAF, Sri Lanka); the allowlist keeps JAF out
    got = nearest_airport(9.2876, 79.3129, airports)
    assert got["iata"] in IN_REGION and got["iata"] != "JAF"
    assert nearest_airport(17.69, 83.29, airports)["iata"] == "VTZ"


def test_military_tags_are_detected():
    from src.tools.routing import is_military
    assert is_military({"military": "airfield"})
    assert is_military({"aerodrome:type": "military/public"})
    assert not is_military({"aerodrome:type": "public"})
    assert not is_military({})


@needs_raw
def test_saved_file_marks_exactly_the_military_airports_among_the_allowlist():
    airports = load_airports()
    assert {k for k, v in airports.items() if v["military"]} == {"IXX", "BEP", "BPM"}


def test_nearest_airport_skips_military_fields():
    airports = {"MIL": {"iata": "MIL", "name": "m", "lat": 10.0, "lon": 77.0, "military": True},
                "CIV": {"iata": "CIV", "name": "c", "lat": 11.0, "lon": 77.0, "military": False}}
    assert nearest_airport(10.0, 77.0, airports)["iata"] == "CIV"


def test_nearest_airport_tie_goes_to_the_lower_iata():
    airports = {"BBB": {"iata": "BBB", "name": "b", "lat": 10.0, "lon": 77.0},
                "AAA": {"iata": "AAA", "name": "a", "lat": 10.0, "lon": 77.0}}
    assert nearest_airport(11.0, 77.0, airports)["iata"] == "AAA"
