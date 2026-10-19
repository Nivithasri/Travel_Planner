from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    BaseChunk,
    EnrichedChunk,
    PlaceRegistryEntry,
    make_chunk_id,
)

BASE = {
    "chunk_id": "abc",
    "text": "The fort is open 9-5.",
    "source_type": "blog",
    "url": "https://example.com/a",
    "date": "2025-03-01",
    "state": "KA",
}
ENRICHED = {
    **BASE,
    "context_note": "Hampi fort timings from a blog.",
    "place": "Hampi",
    "place_type": "heritage",
    "season_tags": ["winter"],
    "aspect": "timing",
    "is_warning": False,
}
PLACE = {
    "name": "Munnar",
    "aliases": ["Munar"],
    "state": "KL",
    "region_cluster": "western_ghats",
    "popularity": "popular",
    "lat": 10.0889,
    "lon": 77.0595,
}


def test_base_chunk_has_no_fake_phase3_values():
    c = BaseChunk(**BASE)
    assert c.aspect is None and c.place is None and c.is_warning is None
    assert c.season_tags is None and c.place_type is None
    assert c.context_note == ""  # contract: "" before Phase 3


def test_enriched_roundtrip_matches_contract_fields():
    c = EnrichedChunk(**ENRICHED)
    assert c.model_dump(mode="json") == ENRICHED


def test_enriched_requires_everything():
    for missing in ("context_note", "place", "place_type", "season_tags", "aspect", "is_warning"):
        data = {k: v for k, v in ENRICHED.items() if k != missing}
        with pytest.raises(ValidationError):
            EnrichedChunk(**data)
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, "context_note": ""})


def test_date_null_ok_but_key_required_and_format_checked():
    assert BaseChunk(**{**BASE, "date": None}).date is None
    no_date = {k: v for k, v in BASE.items() if k != "date"}
    with pytest.raises(ValidationError):
        BaseChunk(**no_date)
    for bad in ("2025-3-1", "01/03/2025", "2025-13-40", ""):
        with pytest.raises(ValidationError):
            BaseChunk(**{**BASE, "date": bad})


@pytest.mark.parametrize(
    "field,value",
    [("source_type", "twitter"), ("state", "MH"), ("aspect", "weather")],
)
def test_enums_rejected(field, value):
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, field: value})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        BaseChunk(**{**BASE, "score": 0.5})


def test_chunk_id_deterministic():
    assert make_chunk_id("u", 0) == make_chunk_id("u", 0)
    assert make_chunk_id("u", 0) != make_chunk_id("u", 1)
    assert make_chunk_id("u", 0) != make_chunk_id("v", 0)


def test_place_registry():
    p = PlaceRegistryEntry(**PLACE)
    assert p.model_dump(mode="json", exclude_none=True) == PLACE
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "popularity": "famous"})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "lat": 123})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "state": "GJ"})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{k: v for k, v in PLACE.items() if k != "aliases"})


def test_reference_is_a_valid_source_type_and_enum_matches_contract():
    from src.common.schemas import SourceType
    assert {s.value for s in SourceType} == {
        "reddit", "blog", "youtube", "wikivoyage", "official", "reference"}
    contract = (Path(__file__).resolve().parents[1] / "docs" / "contracts.md").read_text("utf-8")
    assert all(s.value in contract.split("| source_type |")[1].split("\n")[0] for s in SourceType)


def test_chunk_accepts_reference_source_type():
    assert BaseChunk(**{**BASE, "source_type": "reference"}).source_type.value == "reference"


# ---- merged from tests/test_schemas_strict.py: 

@pytest.mark.parametrize("bad", ["yes", "true", 1, 0, "False", None])
def test_is_warning_is_strict_bool_on_enriched(bad):
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, "is_warning": bad})


@pytest.mark.parametrize("bad", ["yes", "true", 1, 0])
def test_is_warning_is_strict_bool_on_base(bad):
    with pytest.raises(ValidationError):
        BaseChunk(**{**BASE, "is_warning": bad})


def test_is_warning_accepts_real_bools_and_json_booleans():
    assert EnrichedChunk(**{**ENRICHED, "is_warning": True}).is_warning is True
    assert BaseChunk(**BASE).is_warning is None
    raw = (
        '{"chunk_id":"c","text":"t","source_type":"blog","url":"u","date":null,'
        '"state":"TN","is_warning":true}'
    )
    assert BaseChunk.model_validate_json(raw).is_warning is True
