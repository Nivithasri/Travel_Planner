"""run_eval on a toy collection with known answers (D73), plus the baseline adapters' pure parts."""
import math
from datetime import date

import pytest

from eval import adapters, qrels, run_eval

REF = date(2026, 10, 18)
INFO = {
    "c1": {"chunk_id": "c1", "place": "Hampi", "source_type": "blog", "date": "2026-01-05",
           "date_kind": "published", "provenance": "rule_only"},
    "c2": {"chunk_id": "c2", "place": "Hampi", "source_type": "reference", "date": None,
           "date_kind": "revised", "provenance": "llm"},
    "c3": {"chunk_id": "c3", "place": "Mysuru", "source_type": "wikivoyage", "date": None,
           "date_kind": "revised", "provenance": "llm"},
    "c4": {"chunk_id": "c4", "place": "Munnar", "source_type": "wikivoyage", "date": None,
           "date_kind": "revised", "provenance": "fallback"},
    "c5": {"chunk_id": "c5", "place": "Kochi", "source_type": "reference", "date": None,
           "date_kind": "revised", "provenance": "llm"},
}
QUERIES = [
    {"id": "Q1", "text": "one", "split": "dev", "type": "family", "time_sensitive": False,
     "freshness_slice": False, "place_scope": "named", "places": ["Hampi"]},
    {"id": "Q2", "text": "two", "split": "dev", "type": "monsoon", "time_sensitive": False,
     "freshness_slice": False, "place_scope": "free", "places": []},
    {"id": "Q3", "text": "three", "split": "dev", "type": "freshness", "time_sensitive": True,
     "freshness_slice": True, "place_scope": "free", "places": []},
]
QRELS = {"Q1": {"c1": 2, "c2": 1, "c3": 0}, "Q2": {"c5": 2}, "Q3": {"c1": 0}}
RUNS = {"one": ["c2", "c1", "c4"], "two": ["c3", "c4"], "three": ["c1", "c2"]}


def toy_retrieve(query, filters=None, k=10):
    assert filters is None
    return [{**INFO[c], "score": 1.0 / (i + 1)} for i, c in enumerate(RUNS[query][:k])]


def result():
    return run_eval.evaluate(toy_retrieve, QUERIES, QRELS, INFO, REF, {"Munnar"})


def test_known_answers_per_query():
    pq = {r["id"]: r for r in result()["per_query"]}
    # Q1: gains 1,2,0 against ideal 2,1,0
    want = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert pq["Q1"]["ndcg@10"] == pytest.approx(want)
    assert pq["Q1"]["mrr"] == 1.0
    assert pq["Q1"]["p@5"] == pytest.approx(2 / 5)
    assert pq["Q1"]["recall@10"] == pytest.approx(1.0)
    assert pq["Q1"]["unjudged@10"] == pytest.approx(1 / 3)  # c4 has no label
    assert pq["Q1"]["excluded@10"] == pytest.approx(1 / 3)  # c4 is in Munnar
    assert pq["Q1"]["fresh@10"] == pytest.approx(1 / 3)  # c1 is published within 18 months
    assert pq["Q1"]["src_types@10"] == 3 and pq["Q1"]["places@10"] == 2
    # Q2: the only relevant chunk (c5) is never retrieved
    assert pq["Q2"]["ndcg@10"] == 0.0 and pq["Q2"]["mrr"] == 0.0
    assert pq["Q2"]["recall@10"] == 0.0 and pq["Q2"]["unjudged@10"] == 1.0
    # Q3 has no relevant chunk
    assert pq["Q3"]["ndcg@10"] is None and pq["Q3"]["mrr"] is None


def test_means_exclude_zero_relevant_queries_and_count_them():
    g = result()["groups"]
    overall = g["overall"]
    assert overall["n_queries"] == 3 and overall["n_zero_relevant"] == 1
    assert overall["ndcg@10_excluded"] == 1
    q1 = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert overall["ndcg@10"] == pytest.approx((q1 + 0.0) / 2)  # Q3 not in the mean
    assert overall["p@5"] == pytest.approx((2 / 5 + 0 + 0) / 3)  # P@5 keeps every query


def test_groups_cover_types_slice_and_scope():
    g = result()["groups"]
    assert {"overall", "type=family", "type=monsoon", "type=freshness", "freshness_slice",
            "place_scope=named", "place_scope=free"} <= set(g)
    assert g["freshness_slice"]["n_queries"] == 1
    assert g["place_scope=named"]["n_queries"] == 1 and g["place_scope=free"]["n_queries"] == 2
    assert g["type=family"]["ndcg@10"] == pytest.approx(
        (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))


def test_provenance_table_sums_over_queries():
    prov = result()["groups"]["overall"]["provenance"]
    # top-10 slots: Q1 c2(llm) c1(rule_only) c4(fallback); Q2 c3(llm) c4(fallback); Q3 c1 c2
    assert prov["llm"]["n"] == 3 and prov["rule_only"]["n"] == 2 and prov["fallback"]["n"] == 2
    assert prov["llm"]["relevant"] == 1  # only c2 in Q1; c3 is graded 0 or unjudged elsewhere


def test_unknown_chunk_from_retriever_is_an_error():
    def bad(query, filters=None, k=10):
        return [{"chunk_id": "zzz", "score": 1.0}]

    with pytest.raises(ValueError):
        run_eval.evaluate(bad, QUERIES[:1], QRELS, INFO, REF, set())


def test_missing_qrels_entry_means_all_unjudged_and_is_reported():
    out = run_eval.evaluate(toy_retrieve, QUERIES, {"Q1": QRELS["Q1"]}, INFO, REF, set())
    assert out["queries_without_qrels"] == ["Q2", "Q3"]
    pq = {r["id"]: r for r in out["per_query"]}
    assert pq["Q2"]["unjudged@10"] == 1.0


def test_render_states_the_caveats():
    text = run_eval.render(result(), "header")
    assert "LOWER BOUND" in text and "0.000 is expected" in text and "place_scope=named" in text


def test_load_retriever_and_bad_spec():
    assert run_eval.load_retriever("eval.adapters:rrf") is adapters.rrf
    with pytest.raises(ValueError):
        run_eval.load_retriever("nocolon")


def test_dev_run_does_not_allow_test_qrels(monkeypatch):
    calls = []

    def fake_load_qrels(split, *, allow_test=False, directory=None):
        calls.append((split, allow_test))
        return {}

    monkeypatch.setattr(run_eval, "load_qrels", fake_load_qrels)
    monkeypatch.setattr(run_eval, "load_queries", lambda split: QUERIES[:1])
    monkeypatch.setattr(run_eval, "load_reference_date", lambda: REF)
    monkeypatch.setattr(run_eval, "load_chunk_info", lambda *a: INFO)
    monkeypatch.setattr(run_eval, "load_excluded", lambda: set())
    monkeypatch.setattr(run_eval, "load_retriever", lambda spec: toy_retrieve)
    assert run_eval.main(["--retriever", "x:y", "--split", "dev"]) == 0
    assert calls == [("dev", False)]
    assert run_eval.main(["--retriever", "x:y", "--split", "test"]) == 0
    assert calls[-1] == ("test", True)
    assert qrels.TestQrelsRefused  # the refusal itself is covered in test_eval_qrels


def test_rrf_by_hand():
    # a: 1/61, b: 1/62 + 1/61, c: 1/62
    out = adapters.rrf([["a", "b"], ["b", "c"]], k=60)
    assert [c for c, _ in out] == ["b", "a", "c"]
    assert out[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert out[1][1] == pytest.approx(1 / 61)


def test_rrf_ties_break_by_chunk_id():
    out = adapters.rrf([["z"], ["a"]], k=60)  # equal scores
    assert [c for c, _ in out] == ["a", "z"]


def test_adapters_reject_filters_before_touching_the_index():
    with pytest.raises(NotImplementedError):
        adapters.bm25_plain("q", {"state": "KA"})
