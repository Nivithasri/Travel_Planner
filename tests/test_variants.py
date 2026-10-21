"""Phase 6 variant wiring (eval/variants.py), tested with a fake index so no model is needed."""
import numpy as np
import pytest

from eval import variants


class FakeIndex:
    def __init__(self):
        self.chunks = {c: {"chunk_id": c, "place": "Hampi", "text": f"text {c}"}
                       for c in ("a", "b", "c", "d")}
        self.calls: list[tuple[str, int]] = []

    def table(self, name, query, k):
        self.calls.append((name, k))
        rankings = {"bm25_plain": ["a", "b", "c"], "dense_plain": ["b", "c", "d"],
                    "bm25_contextual": ["c", "a"], "dense_contextual": ["a", "d"]}
        return [{"chunk_id": cid, "score": 1.0 / (i + 1)}
                for i, cid in enumerate(rankings[name][:k])]


def test_hybrid_plain_fuses_the_two_plain_arms(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    out = variants.hybrid_plain("q", None, 10)
    # bm25_plain [a,b,c] + dense_plain [b,c,d]: b is in both top ranks -> first
    assert [r["chunk_id"] for r in out][0] == "b"
    assert out[0]["place"] == "Hampi" and "score" in out[0]  # full chunk fields + score
    assert all(name in ("bm25_plain", "dense_plain") for name, _ in fake.calls)


def test_hybrid_arms_pull_depth_100(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    variants.hybrid_contextual("q", None, 10)
    assert all(k == variants.RRF_DEPTH for _, k in fake.calls)  # fuse deep, cut to k after


def test_hybrid_contextual_uses_the_contextual_tables(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    variants.hybrid_contextual("q", None, 5)
    assert {name for name, _ in fake.calls} == {"bm25_contextual", "dense_contextual"}


def test_control_arms_use_the_control_index(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_control_index", lambda: fake)
    out = variants.dense_control("q", None, 2)
    assert fake.calls == [("dense_contextual", 2)]
    assert [r["chunk_id"] for r in out] == ["a", "d"]


def test_filters_raise_until_row_5():
    for fn in (variants.hybrid_plain, variants.hybrid_contextual, variants.hybrid_control,
               variants.dense_control, variants.bm25_control):
        with pytest.raises(NotImplementedError):
            fn("q", {"state": ["KA"]}, 10)


# ---- row 4: expansion ------------------------------------------------------------------------

class QueryIndex:
    """A fake index whose rankings depend on the query text."""

    def __init__(self, by_query):
        self.chunks = {c: {"chunk_id": c, "place": "P", "text": c} for c in "abcdef"}
        self.by_query = by_query
        self.calls: list[tuple[str, str]] = []

    def table(self, name, query, k):
        self.calls.append((name, query))
        return [{"chunk_id": c, "score": 1.0} for c in self.by_query[query][:k]]


def test_expanded_ids_fuses_one_hybrid_ranking_per_query():
    from src.retrieval.hybrid import CONTEXTUAL_TABLES, expanded_ids

    idx = QueryIndex({"req": ["a", "b"], "sub1": ["c", "b"], "sub2": ["b", "d"]})
    out = expanded_ids(idx, CONTEXTUAL_TABLES, ["req", "sub1", "sub2"], depth=10)
    ids = [c for c, _ in out]
    assert ids[0] == "b"  # b is high in all three lists
    assert set(ids) == {"a", "b", "c", "d"}
    # every query hit both contextual tables
    assert {(n, q) for n, q in idx.calls} == {(t, q) for t in CONTEXTUAL_TABLES
                                             for q in ("req", "sub1", "sub2")}


def test_expanded_hybrid_uses_the_request_and_its_parsed_subqueries(monkeypatch):
    idx = QueryIndex({"req": ["a"], "s1": ["b"], "s2": ["c"], "s3": ["d"]})
    monkeypatch.setattr(variants, "get_index", lambda: idx)
    monkeypatch.setattr(variants, "parsed_by_text",
                        lambda: {"req": {"sub_queries": ["s1", "s2", "s3"]}})
    out = variants.expanded_hybrid("req", None, 10)
    assert {r["chunk_id"] for r in out} == {"a", "b", "c", "d"}
    assert {q for _, q in idx.calls} == {"req", "s1", "s2", "s3"}


def test_expanded_hybrid_without_a_parse_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(variants, "get_index", lambda: QueryIndex({}))
    monkeypatch.setattr(variants, "parsed_by_text", lambda: {})
    with pytest.raises(KeyError, match="parse_queries"):
        variants.expanded_hybrid("unknown", None, 10)


def test_a_fallback_parse_runs_the_request_alone(monkeypatch):
    idx = QueryIndex({"req": ["a", "b"]})
    monkeypatch.setattr(variants, "get_index", lambda: idx)
    monkeypatch.setattr(variants, "parsed_by_text", lambda: {"req": {"sub_queries": []}})
    out = variants.expanded_hybrid("req", None, 10)
    assert [r["chunk_id"] for r in out] == ["a", "b"] and {q for _, q in idx.calls} == {"req"}


# ---- rows 5 and 6 wiring ---------------------------------------------------------------------

def _wire(monkeypatch, parse):
    from datetime import date

    idx = QueryIndex({"req": ["a", "b", "c"], "s1": ["b", "c"]})
    idx.chunks = {c: {"chunk_id": c, "place": "P", "state": st, "season_tags": ["all_year"],
                      "place_type": "city", "date_kind": "revised", "date": None,
                      "aspect": "other"} for c, st in (("a", "KL"), ("b", "KA"), ("c", "KA"))}
    monkeypatch.setattr(variants, "get_index", lambda: idx)
    monkeypatch.setattr(variants, "get_provenance", lambda: {"a": "llm", "b": "llm", "c": "llm"})
    monkeypatch.setattr(variants, "parsed_by_text", lambda: {"req": parse})
    monkeypatch.setattr("src.common.schemas.load_reference_date", lambda: date(2026, 10, 18))
    return idx


PARSE = {"sub_queries": ["s1"], "states": ["KA"], "season": None, "month": None,
         "place_types": [], "time_sensitive": False}


def test_row5_filters_with_the_parsed_state_and_runs_the_subqueries(monkeypatch):
    idx = _wire(monkeypatch, PARSE)
    out = variants.row5_filters("req", None, 10)
    assert [r["chunk_id"] for r in out][:2] == ["b", "c"]  # KL chunk a goes behind
    assert {q for _, q in idx.calls} == {"req", "s1"}


def test_run_post_reports_the_trace(monkeypatch):
    from src.retrieval.postprocess import PostConfig

    _wire(monkeypatch, PARSE)
    rows, trace = variants.run_post("req", PostConfig(use_filters=True), 2)
    assert [r["chunk_id"] for r in rows] == ["b", "c"] and trace["backfilled"] == 0
    assert trace["filters"]["rejected"] == {"state": 1}


def test_hybrid_base_runs_the_request_alone(monkeypatch):
    idx = _wire(monkeypatch, PARSE)
    variants.hybrid_filters("req", None, 10)
    assert {q for _, q in idx.calls} == {"req"}


def test_row6_with_a_stable_query_equals_row5(monkeypatch):
    _wire(monkeypatch, PARSE)
    a = [r["chunk_id"] for r in variants.row5_filters("req", None, 10)]
    b = [r["chunk_id"] for r in variants.row6_fresh_w30("req", None, 10)]
    assert a == b  # time_sensitive is False in the parse, so w_q = 0


def test_post_variants_refuse_a_query_that_was_not_parsed(monkeypatch):
    _wire(monkeypatch, PARSE)
    with pytest.raises(KeyError, match="parse_queries"):
        variants.row5_filters("never parsed", None, 10)


# ---- row 8 wiring ----------------------------------------------------------------------------

class MmrIndex(QueryIndex):
    def __init__(self):
        super().__init__({"req": ["a", "b", "c", "d"], "s1": ["b", "c"]})
        src = {"a": "blog", "b": "blog", "c": "wikivoyage", "d": "blog"}
        self.chunks = {c: {"chunk_id": c, "place": "P" + c, "source_type": src[c], "text": c}
                       for c in "abcd"}
        # a and b are identical vectors, c and d differ
        self._m = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    def dense_matrix(self, kind):
        return self._m, {c: i for i, c in enumerate("abcd")}


def _mmr_wire(monkeypatch, places=()):
    idx = MmrIndex()
    monkeypatch.setattr(variants, "get_index", lambda: idx)
    monkeypatch.setattr(variants, "parsed_by_text", lambda: {
        "req": {"sub_queries": ["s1"], "places": list(places)}})
    return idx


LOGITS = {"a": 3.0, "b": 2.8, "c": 0.0, "d": -0.5}


def scorer(query, ids):
    return {c: LOGITS[c] for c in ids}


def test_row8_lambda_one_is_the_reranked_order(monkeypatch):
    _mmr_wire(monkeypatch)
    rows, trace = variants.run_mmr("req", 1.0, scorer, 4)
    assert [r["chunk_id"] for r in rows] == ["a", "b", "c", "d"]
    assert trace["place_term"] is True and trace["depth"] == 30


def test_row8_lower_lambda_moves_the_duplicate_down(monkeypatch):
    _mmr_wire(monkeypatch)
    rows, _ = variants.run_mmr("req", 0.5, scorer, 4)
    ids = [r["chunk_id"] for r in rows]
    assert ids[0] == "a" and ids.index("b") > ids.index("c")  # b duplicates a


def test_row8_place_named_queries_switch_the_place_term_off(monkeypatch):
    _mmr_wire(monkeypatch, places=["Hampi"])
    _, trace = variants.run_mmr("req", 0.5, scorer, 4)
    assert trace["place_term"] is False


def test_row8_the_scorer_decides_the_order_before_mmr(monkeypatch):
    _mmr_wire(monkeypatch)
    flipped = lambda q, ids: {c: -LOGITS[c] for c in ids}  # noqa: E731
    rows, _ = variants.run_mmr("req", 1.0, flipped, 4)
    assert [r["chunk_id"] for r in rows][0] == "d"


def test_row8_variant_refuses_filters_and_unparsed_queries(monkeypatch):
    _mmr_wire(monkeypatch)
    f = variants.make_mmr_variant(0.8, scorer)
    with pytest.raises(NotImplementedError):
        f("req", {"state": ["KA"]}, 10)
    with pytest.raises(KeyError, match="parse_queries"):
        f("never parsed", None, 10)
