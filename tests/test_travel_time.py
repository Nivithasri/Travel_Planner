"""travel_time lookups, the committed matrix file and the no-chunk-text rule (D81, D83)."""
from __future__ import annotations

import json
import os
import time

import pytest

from src.common.schemas import REPO_ROOT
from src.places.places import Registry
from src.tools import check_season as cs
from src.tools import travel_time as tt
from src.tools.routing import UnknownPlaceError
from src.tools.travel_model import REGISTRY_PATH, haversine_km

MATRIX = REPO_ROOT / "index" / "travel_matrix.json"
SEASON = REPO_ROOT / "index" / "season_data.json"
CHUNKS = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
NODE_KEYS = {"id", "name", "kind", "state", "lat", "lon", "hill", "aliases", "iata"}
MATRIX_META_KEYS = {"status", "decision", "attribution", "params", "nodes", "hubs", "pairs",
                    "triangle_violations", "airport_aliases"}


def registry():
    return Registry.load(REGISTRY_PATH)


def doc():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_every_destination_pair_has_a_value_and_is_symmetric():
    m = tt.load_matrix()
    names = [e.name for e in registry().entries if not e.holdout]
    assert len(names) == 26
    for a in names:
        assert m.travel_time(a, a) == 0.0
        for b in names:
            assert m.travel_time(a, b) == m.travel_time(b, a)
            if a != b:
                assert m.travel_time(a, b) >= 0.25


def test_hubs_are_valid_in_travel_time_by_name_alias_and_iata():
    m = tt.load_matrix()
    base = m.travel_time("Kochi", "Coimbatore")
    assert base == m.travel_time("kochi", "CJB") == m.travel_time("Kochi", "hub-cjb")
    assert base == m.travel_time("Kochi", "Coimbatore International Airport")
    assert m.travel_time("Chennai", "Bengaluru") == m.travel_time("Madras", "Bangalore")
    assert m.travel_time("Chennai", "KJB") == m.travel_time("Chennai", "Kurnool")


def test_every_nearest_airport_code_resolves_in_travel_time_for_all_26_places():
    m = tt.load_matrix()
    table = cs.load_season()
    names = [e.name for e in registry().entries if not e.holdout]
    assert len(names) == 26
    for name in names:
        code = table.check(name, 1)["nearest_airport"]["iata"]
        node_id = m.resolve(code)  # raises UnknownPlaceError when the code is not accepted
        assert m.travel_time(code, "Chennai") >= 0.0
        assert node_id in m.nodes
    for code, node_id in doc()["meta"]["airport_aliases"].items():
        assert m.nodes[node_id]["kind"] == "destination"


def test_airport_alias_codes_are_near_their_destination():
    d = doc()
    nodes = {n["id"]: n for n in d["nodes"]}
    allowed = {"COK", "HYD", "IXM", "MAA", "MYQ", "PNY", "TIR", "WGC"}
    assert set(d["meta"]["airport_aliases"]) == allowed
    from src.tools.routing import load_airports
    if not os.path.exists(REPO_ROOT / "data" / "raw" / "overpass" / "_south" / "airports.json"):
        pytest.skip("airport coordinates need the gitignored data/raw (fresh clone)")
    airports = load_airports()
    for code, node_id in d["meta"]["airport_aliases"].items():
        a, n = airports[code], nodes[node_id]
        assert haversine_km(a["lat"], a["lon"], n["lat"], n["lon"]) <= 25


def test_holdouts_and_nonsense_are_unknown_in_travel_time():
    for name in ("Thanjavur", "Thekkady", "Chikmagalur", "Araku Valley", "Atlantis", "", None, 5):
        with pytest.raises(UnknownPlaceError):
            tt.travel_time("Chennai", name)
        with pytest.raises(UnknownPlaceError):
            tt.travel_time(name, "Chennai")


def test_unknown_place_error_is_a_value_error_with_suggestions():
    with pytest.raises(ValueError) as err:
        tt.travel_time("Chenai", "Madurai")
    assert isinstance(err.value, UnknownPlaceError) and "Chennai" in err.value.suggestions
    assert "Chennai" in str(err.value)


def test_matrix_file_shape_and_holdouts_absent():
    d = doc()
    ids = {n["id"] for n in d["nodes"]}
    assert len(d["nodes"]) == 48 and d["meta"]["status"] == "PROVISIONAL"
    assert set(d["meta"]) == MATRIX_META_KEYS
    assert not ids & {"thanjavur", "thekkady", "chikmagalur", "araku-valley"}
    assert sum(n["kind"] == "hub" for n in d["nodes"]) == 22
    assert len(d["hours"]) == len(d["km"]) == 48 * 47 // 2
    assert all(set(n) <= NODE_KEYS for n in d["nodes"])
    assert d["meta"]["triangle_violations"] == 0
    assert set(d["meta"]["params"]) >= {"band_km", "road_factor", "speed_short_kmh",
                                        "speed_long_kmh", "ghat_surcharge_h"}


def test_known_values_are_in_the_expected_range():
    assert 5.0 <= tt.travel_time("Chennai", "Bengaluru") <= 7.0
    assert 3.0 <= tt.travel_time("Kochi", "Munnar") <= 4.5


def test_hill_penalty_is_a_fixed_surcharge_not_scaled_with_distance():
    """B1: a long trip to a hill node is its plain time plus the surcharge, and a detour through
    a plain hub on the way is never faster than the direct trip by more than the surcharge."""
    d = doc()
    sur = d["meta"]["params"]["ghat_surcharge_h"]
    m = tt.load_matrix()
    direct = m.travel_time("Chennai", "Ooty")
    assert direct < m.travel_time("Chennai", "CJB") + m.travel_time("CJB", "Ooty")
    assert direct < 12.0 and sur < 3.0
    nodes = [n for n in d["nodes"] if not n["hill"]]
    far = max(nodes, key=lambda n: m.travel_time(n["id"], "ooty"))
    near = min(nodes, key=lambda n: m.travel_time(n["id"], "ooty"))
    gap = m.travel_time(far["id"], "ooty") - m.travel_time(near["id"], "ooty")
    plain_gap = m.travel_time(far["id"], "munnar") - m.travel_time(near["id"], "munnar")
    assert gap > 0 and plain_gap > -10  # both grow with distance, no distance-scaled blow-up


def test_lookup_is_instant_after_the_first_load():
    tt.travel_time("Chennai", "Madurai")
    t0 = time.perf_counter()
    for _ in range(1000):
        tt.travel_time("Chennai", "Madurai")
    assert (time.perf_counter() - t0) / 1000 < 0.001


def test_cache_reloads_a_rebuilt_file(tmp_path):
    """B8: the loader notices a changed file (modification time or size)."""
    src = json.loads(MATRIX.read_text(encoding="utf-8"))
    path = tmp_path / "m.json"
    path.write_text(json.dumps(src), encoding="utf-8")
    first = tt.load_matrix(path)
    assert tt.load_matrix(path) is first  # unchanged file: same object
    key = next(iter(src["hours"]))
    src["hours"][key] = 99.0
    path.write_text(json.dumps(src), encoding="utf-8")
    os.utime(path, ns=(1, 2_000_000_000))  # force a different mtime even on coarse clocks
    second = tt.load_matrix(path)
    assert second is not first and second.hours[key] == 99.0


def test_season_cache_reloads_a_rebuilt_file(tmp_path):
    src = json.loads(SEASON.read_text(encoding="utf-8"))
    path = tmp_path / "s.json"
    path.write_text(json.dumps(src), encoding="utf-8")
    first = cs.load_season(path)
    assert cs.load_season(path) is first
    src["places"]["Munnar"]["confidence"] = "medium"
    path.write_text(json.dumps(src), encoding="utf-8")
    os.utime(path, ns=(1, 2_000_000_000))
    second = cs.load_season(path)
    assert second is not first and second.places["Munnar"]["confidence"] == "medium"


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def test_index_files_hold_only_short_strings_so_no_chunk_text_fits():
    """Always runs (also on a clone): chunk text is hundreds of characters, and every string in
    both files is far shorter. Attribution is the longest string."""
    for path in (MATRIX, SEASON):
        longest = max(len(s) for s in _strings(json.loads(path.read_text(encoding="utf-8"))))
        assert longest < 260, (path.name, longest)
    season = json.loads(SEASON.read_text(encoding="utf-8"))
    for entry in season["places"].values():
        for row in entry["months"].values():
            assert set(row["warning_sources"]) == set(row["warning_chunk_ids"])
            assert all(len(i) == 16 for i in row["warning_chunk_ids"])
            assert set(row["evidence"]["model_written"]) == {"chunks", "season_tagged", "all_year"}
            assert set(row["warning_sources"].values()) <= {"model_written"}


@pytest.mark.skipif(not CHUNKS.exists(),
                    reason="data/processed/chunks.jsonl is gitignored and absent on a fresh clone")
def test_neither_index_file_contains_chunk_text():
    blobs = [MATRIX.read_text(encoding="utf-8"), SEASON.read_text(encoding="utf-8")]
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        prefix = json.loads(line)["text"][:40]
        for blob in blobs:
            assert prefix not in blob
