"""Freshness ranking (src/retrieval/freshness.py, D87) and the filter+freshness composition."""
from datetime import date

import pytest

from src.retrieval import freshness
from src.retrieval.postprocess import PostConfig, postprocess

REF = date(2026, 10, 18)


def blog(aspect, when):
    return {"date_kind": "published", "date": when, "aspect": aspect, "state": "KA",
            "season_tags": ["all_year"], "place_type": "city"}


def wiki(aspect="history"):
    return {"date_kind": "revised", "date": None, "aspect": aspect, "state": "KA",
            "season_tags": ["all_year"], "place_type": "city"}


def test_half_life_halves_the_factor():
    f, decayed = freshness.freshness_factor(blog("price", "2026-04-21"), REF)  # 180 d old
    assert decayed and f == pytest.approx(0.5 ** 1.0)
    f2, _ = freshness.freshness_factor(blog("food", "2024-10-18"), REF)  # 730 d, food
    assert f2 == pytest.approx(0.5)


def test_fast_aspects_decay_faster_than_slow_ones():
    old = "2024-10-18"
    fast = freshness.freshness_factor(blog("closure", old), REF)[0]
    slow = freshness.freshness_factor(blog("food", old), REF)[0]
    assert fast < slow < 0.5 + 1e-9


@pytest.mark.parametrize("aspect", ["scenery", "history"])
def test_stable_aspects_do_not_decay(aspect):
    assert freshness.freshness_factor(blog(aspect, "2005-01-01"), REF) == (0.5, False)


@pytest.mark.parametrize("c", [
    wiki(), {"date_kind": "none", "date": None, "aspect": "price"},
    {"date_kind": "sitemap_lastmod", "date": None, "aspect": "price"},
    {"date_kind": "revised", "date": "2026-01-01", "aspect": "price"}])
def test_undated_or_revised_chunks_are_neutral_never_old(c):
    assert freshness.freshness_factor(c, REF) == (0.5, False)


def test_a_future_date_is_not_negative_age():
    f, _ = freshness.freshness_factor(blog("price", "2030-01-01"), REF)
    assert f == pytest.approx(1.0)


def test_weight_zero_or_a_stable_query_leaves_the_order_unchanged():
    chunks = {"a": blog("price", "2010-01-01"), "b": wiki(), "c": wiki()}
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    for w, ts in ((0.0, True), (0.3, False)):
        out, trace = freshness.rerank(ranked, chunks, REF, w, ts)
        assert [c for c, _ in out] == ["a", "b", "c"] and trace["w_q"] == 0.0


def test_time_sensitive_query_pushes_an_old_dated_chunk_below_a_near_tie():
    chunks = {"a": blog("price", "2010-01-01"), "b": wiki(), "c": wiki()}
    ranked = [("a", 0.81), ("b", 0.80), ("c", 0.70)]  # a and b nearly tie on relevance
    out, trace = freshness.rerank(ranked, chunks, REF, 0.3, True)
    assert [c for c, _ in out][0] == "b" and trace["w_q"] == 0.3
    assert trace["decayed"] == 1 and trace["moved"] >= 1


def test_a_clear_relevance_leader_stays_first_at_the_largest_grid_weight():
    # w <= 0.3 reorders near-ties only: the old dated chunk keeps a wide lead
    chunks = {"a": blog("price", "2010-01-01"), "b": wiki(), "c": wiki()}
    out, _ = freshness.rerank([("a", 0.9), ("b", 0.5), ("c", 0.1)], chunks, REF,
                              max(freshness.GRID), True)
    assert [c for c, _ in out][0] == "a"


def test_rel_normalise_min_max_and_equal_scores():
    assert freshness.rel_normalise([4.0, 2.0, 0.0]) == [1.0, 0.5, 0.0]
    assert freshness.rel_normalise([2.0, 2.0]) == [1.0, 1.0]
    assert freshness.rel_normalise([]) == []


def test_ties_break_by_chunk_id():
    chunks = {"z": wiki(), "a": wiki()}
    out, _ = freshness.rerank([("z", 1.0), ("a", 1.0)], chunks, REF, 0.2, True)
    assert [c for c, _ in out] == ["a", "z"]


# ---- composition -------------------------------------------------------------------------

CHUNKS = {"a": wiki(), "b": wiki(), "c": wiki()}
CHUNKS["b"]["state"] = "KL"
PROV = {"a": "llm", "b": "llm", "c": "llm"}
RANKED = [("b", 0.9), ("a", 0.8), ("c", 0.7)]
PARSED = {"states": ["KA"], "time_sensitive": False}


def test_off_config_changes_nothing():
    out, trace = postprocess(RANKED, CHUNKS, PROV, PARSED, PostConfig(), 10)
    assert [c for c, _ in out] == ["b", "a", "c"] and trace["backfilled"] == 0


def test_filters_move_rejected_rows_behind_and_backfill_to_k():
    out, trace = postprocess(RANKED, CHUNKS, PROV, PARSED, PostConfig(use_filters=True), 10)
    assert [c for c, _ in out] == ["a", "c", "b"]  # b (KL) rejected but kept as back-fill
    assert trace["filters"]["rejected"] == {"state": 1} and trace["backfilled"] == 1
    out2, trace2 = postprocess(RANKED, CHUNKS, PROV, PARSED, PostConfig(use_filters=True), 2)
    assert [c for c, _ in out2] == ["a", "c"] and trace2["backfilled"] == 0


def test_explicit_filters_apply_even_with_the_config_off():
    out, trace = postprocess(RANKED, CHUNKS, PROV, {"time_sensitive": False}, PostConfig(), 10,
                             explicit_filters={"state": ["KL"]})
    assert [c for c, _ in out][0] == "b" and trace["filters"]["rejected"] == {"state": 2}


def test_freshness_needs_the_reference_date():
    with pytest.raises(ValueError, match="reference date"):
        postprocess(RANKED, CHUNKS, PROV, PARSED, PostConfig(fresh_w=0.2), 10)


def test_gold_fields_in_the_parse_are_ignored():
    gold = {**PARSED, "places": ["Kochi"], "place_scope": "named", "time_sensitive": False}
    a, _ = postprocess(RANKED, CHUNKS, PROV, gold, PostConfig(use_filters=True), 10)
    b, _ = postprocess(RANKED, CHUNKS, PROV, PARSED, PostConfig(use_filters=True), 10)
    assert a == b
