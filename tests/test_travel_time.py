"""travel_time lookups, the committed matrix file and the no-chunk-text rule (D81)."""
from __future__ import annotations

import json
import time

import pytest

from src.common.schemas import REPO_ROOT
from src.places.places import Registry
from src.tools import travel_time as tt
from src.tools.routing import UnknownPlaceError
from src.tools.travel_model import REGISTRY_PATH

MATRIX = REPO_ROOT / "index" / "travel_matrix.json"
SEASON = REPO_ROOT / "index" / "season_data.json"
CHUNKS = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
NODE_KEYS = {"id", "name", "kind", "state", "lat", "lon", "hill", "aliases", "iata"}


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
    assert not ids & {"thanjavur", "thekkady", "chikmagalur", "araku-valley"}
    assert sum(n["kind"] == "hub" for n in d["nodes"]) == 22
    assert len(d["hours"]) == len(d["km"]) == 48 * 47 // 2
    assert all(set(n) <= NODE_KEYS for n in d["nodes"])


def test_known_values_are_in_the_expected_range():
    assert 5.0 <= tt.travel_time("Chennai", "Bengaluru") <= 7.0
    assert 3.0 <= tt.travel_time("Kochi", "Munnar") <= 4.5


def test_lookup_is_instant_after_the_first_load():
    tt.travel_time("Chennai", "Madurai")
    t0 = time.perf_counter()
    for _ in range(1000):
        tt.travel_time("Chennai", "Madurai")
    assert (time.perf_counter() - t0) / 1000 < 0.001


def chunk_prefixes():
    out = []
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        out.append(json.loads(line)["text"][:40])
    return out


@pytest.mark.skipif(not CHUNKS.exists(), reason="frozen chunks not present")
def test_neither_index_file_contains_chunk_text():
    blobs = [MATRIX.read_text(encoding="utf-8"), SEASON.read_text(encoding="utf-8")]
    for prefix in chunk_prefixes():
        for blob in blobs:
            assert prefix not in blob
    season = json.loads(blobs[1])
    allowed = {"chunk_id", "chunks", "season_tagged", "all_year"}
    for entry in season["places"].values():
        for row in entry["months"].values():
            assert set(row["warning_sources"]) == set(row["warning_chunk_ids"])
            assert all(len(i) == 16 for i in row["warning_chunk_ids"])
            assert set(row["evidence"]["model_written"]) == allowed - {"chunk_id"}
