"""Metadata filters (src/retrieval/filters.py, D85, D69)."""
import pytest

from src.retrieval.filters import Filters, chunk_verdict, partition


def chunk(state="KA", tags=("all_year",), place_type="city"):
    return {"state": state, "season_tags": list(tags), "place_type": place_type}


def test_unknown_filter_keys_raise():
    with pytest.raises(ValueError, match="unknown filter keys"):
        Filters.from_dict({"state": ["KA"], "aspect": ["price"]})
    with pytest.raises(ValueError):
        Filters.from_dict({"month": 13})
    with pytest.raises(ValueError):
        Filters.from_dict({"season": "spring"})


def test_a_month_fixes_the_season():
    assert Filters.from_dict({"month": 7}).target_season == "monsoon"
    assert Filters.from_dict({"month": 1, "season": "summer"}).target_season == "winter"
    assert Filters.from_dict({"season": "summer"}).target_season == "summer"


def test_state_filter_applies_to_every_provenance_class():
    f = Filters.from_dict({"state": ["KA"]})
    for prov in ("llm", "fallback", "rule_only"):
        assert chunk_verdict(chunk("KL"), prov, f)[0] == "state"
        assert chunk_verdict(chunk("KA"), prov, f)[0] is None


def test_season_passes_all_year_empty_and_matching_tags_only():
    f = Filters.from_dict({"season": "monsoon"})
    assert chunk_verdict(chunk(tags=("all_year",)), "llm", f)[0] is None
    assert chunk_verdict(chunk(tags=()), "llm", f)[0] is None
    assert chunk_verdict(chunk(tags=("monsoon", "winter")), "llm", f)[0] is None
    assert chunk_verdict(chunk(tags=("winter",)), "llm", f)[0] == "season"


def test_place_type_is_a_hard_filter_on_model_written_rows():
    f = Filters.from_dict({"place_type": ["beach"]})
    assert chunk_verdict(chunk(place_type="beach"), "llm", f) == (None, False)
    assert chunk_verdict(chunk(place_type="city"), "llm", f) == ("place_type", False)


@pytest.mark.parametrize("prov", ["rule_only", "fallback"])
def test_rule_based_rows_are_exempt_from_the_place_type_filter(prov):
    f = Filters.from_dict({"place_type": ["beach"]})
    # a city-typed rule-based row is neither removed nor boosted; the skip is reported
    assert chunk_verdict(chunk(place_type="city"), prov, f) == (None, True)


def test_diagnostic_5b_filters_rule_based_rows_too():
    f = Filters.from_dict({"place_type": ["beach"]})
    assert chunk_verdict(chunk(place_type="city"), "rule_only", f,
                         rule_based_place_type=True) == ("place_type", False)


def test_from_parsed_reads_only_the_parse_fields_and_explicit_overrides():
    parsed = {"states": ["KA"], "season": "monsoon", "month": None, "place_types": ["beach"],
              "time_sensitive": True, "place_scope": "named", "places": ["Hampi"]}
    f = Filters.from_parsed(parsed)
    assert f.state == ("KA",) and f.season == "monsoon" and f.place_type == ("beach",)
    g = Filters.from_parsed(parsed, {"state": ["KL"]})
    assert g.state == ("KL",) and g.place_type == ("beach",)
    assert Filters.from_parsed({"states": [], "place_types": []}).empty


def test_partition_keeps_rank_order_and_reports_what_it_rejected():
    chunks = {"a": chunk("KA"), "b": chunk("KL"), "c": chunk("KA"), "d": chunk("TN")}
    prov = {"a": "llm", "b": "llm", "c": "rule_only", "d": "llm"}
    ranked = [("b", 0.9), ("a", 0.8), ("d", 0.7), ("c", 0.6)]
    passing, rejected, trace = partition(ranked, chunks, prov, Filters.from_dict({"state": ["KA"]}))
    assert [c for c, _ in passing] == ["a", "c"] and [c for c, _ in rejected] == ["b", "d"]
    assert trace["rejected"] == {"state": 2}
    assert trace["rejected_by_provenance"] == {"llm": 2}


def test_exempt_count_is_the_passing_rule_based_rows_whose_check_was_skipped():
    chunks = {"a": chunk(place_type="beach"), "b": chunk(place_type="city"),
              "c": chunk(place_type="city")}
    prov = {"a": "llm", "b": "rule_only", "c": "llm"}
    passing, rejected, trace = partition([("a", 3), ("b", 2), ("c", 1)], chunks, prov,
                                         Filters.from_dict({"place_type": ["beach"]}))
    assert [c for c, _ in passing] == ["a", "b"] and [c for c, _ in rejected] == ["c"]
    assert trace["exempt"] == 1
