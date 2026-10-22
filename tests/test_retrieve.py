"""retrieve() and its config switches (src/retrieval/retrieve.py, D84, D85, D96). Fakes only."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.retrieval import retrieve as R
from src.retrieval.fusion import rrf

ROOT = Path(__file__).resolve().parents[1]
IDS = list("abcdefgh")


class FakeIndex:
    """Rankings by (table, query); every chunk has all the fields the pipeline reads."""

    def __init__(self, rankings=None):
        self.rankings = rankings or {}
        self.calls: list[tuple[str, str, int]] = []
        self.chunks = {c: {"chunk_id": c, "place": "Hampi" if c in "abcd" else "Kochi",
                           "state": "KA" if c in "abcd" else "KL", "source_type": "blog",
                           "url": f"https://x.test/{c}", "text": f"text {c}",
                           "season_tags": ["all_year"], "place_type": "city",
                           "date_kind": "revised", "date": None, "aspect": "other"}
                       for c in IDS}
        self.ids = IDS
        self._m = np.eye(len(IDS), dtype=np.float32)

    def table(self, name, query, k):
        self.calls.append((name, query, k))
        ranking = self.rankings.get((name, query)) or self.rankings.get(query, IDS)
        return [{"chunk_id": c, "score": round(1.0 - 0.1 * i, 4)}
                for i, c in enumerate(ranking[:k])]

    def dense_matrix(self, kind):
        return self._m, {c: i for i, c in enumerate(IDS)}


PROV = {c: "llm" for c in IDS}


def boom(*a, **k):
    raise AssertionError("must not be called by this config")


def parse_for(sub=("s1", "s2"), places=(), states=()):
    return lambda q: {"sub_queries": list(sub), "places": list(places), "states": list(states),
                      "season": None, "month": None, "place_types": [],
                      "time_sensitive": False}


# ---- the frozen config -----------------------------------------------------------------------

def test_the_frozen_config_is_dense_contextual_alone_and_matches_the_committed_record():
    cfg = R.FROZEN_CONFIG
    assert cfg.tables == ("dense_contextual",)
    assert (cfg.expansion, cfg.use_filters, cfg.fresh_w, cfg.rerank, cfg.mmr_lambda,
            cfg.corrective_tau) == (False, False, 0.0, False, None, None)
    rec = json.loads((ROOT / "eval" / "frozen_config.json").read_text(encoding="utf-8"))
    assert rec["frozen"] == R.config_dict(cfg)  # changing the frozen config means a new decision


def test_the_hybrid_is_kept_as_a_measured_but_not_frozen_config():
    assert R.HYBRID_CONFIG.tables == ("bm25_contextual", "dense_contextual")
    assert R.HYBRID_CONFIG is not R.FROZEN_CONFIG and R.HYBRID_CONFIG.name != R.FROZEN_CONFIG.name


def test_the_frozen_config_searches_only_the_dense_arm_and_keeps_its_own_scores():
    idx = FakeIndex({("dense_contextual", "q"): ["c", "a", "b"]})
    rows, trace = R.Retriever(R.FROZEN_CONFIG, index=idx, provenance=PROV).search("q", None, 3)
    assert [r["chunk_id"] for r in rows] == ["c", "a", "b"]
    assert [r["score"] for r in rows] == [1.0, 0.9, 0.8]  # cosine scores, not fused ones
    assert {name for name, _, _ in idx.calls} == {"dense_contextual"}  # no BM25, no fusion
    assert trace["config"] == "dense_contextual"


@pytest.mark.parametrize("kw", [{"tables": ("bm25_contextual",)}, {"tables": ("dense_plain",)},
                                {"tables": ("dense_contextual",), "expansion": True},
                                {"tables": ("dense_contextual",), "rerank": True},
                                {"tables": ("dense_contextual",), "use_filters": True},
                                {"tables": ("dense_contextual",), "fresh_w": 0.1}])
def test_the_dense_arm_is_only_accepted_alone_and_only_the_two_ablated_arm_sets_exist(kw):
    with pytest.raises(ValueError):
        R.RetrieveConfig(name="bad", **kw)


def test_the_full_chain_carries_the_dev_chosen_parameters():
    lam = json.loads((ROOT / "eval" / "mmr_lambda.json").read_text())["lam"]
    tau = json.loads((ROOT / "eval" / "corrective_tau.json").read_text())["tau"]
    assert R.FULL_CONFIG.mmr_lambda == lam and R.FULL_CONFIG.corrective_tau == tau
    assert R.FULL_CONFIG.expansion and R.FULL_CONFIG.rerank


def test_retrieve_has_the_contract_signature():
    sig = inspect.signature(R.retrieve)
    assert list(sig.parameters) == ["query", "filters", "k"]
    assert sig.parameters["filters"].default is None and sig.parameters["k"].default == 10


def test_the_frozen_config_needs_no_parser_scorer_or_llm():
    r = R.Retriever(R.FROZEN_CONFIG, index=FakeIndex(), provenance=PROV)
    rows, trace = r.search("any query", None, 5)
    assert [x["chunk_id"] for x in rows] == IDS[:5] and trace["config"] == R.FROZEN_CONFIG.name


def test_the_hybrid_config_fuses_both_arms_by_rrf():
    idx = FakeIndex({("bm25_contextual", "q"): ["b", "a", "c"],
                     ("dense_contextual", "q"): ["a", "b", "d"]})
    rows, _ = R.Retriever(R.HYBRID_CONFIG, index=idx, provenance=PROV).search("q", None, 3)
    want = [c for c, _ in rrf([["b", "a", "c"], ["a", "b", "d"]])][:3]
    assert [r["chunk_id"] for r in rows] == want
    assert all({"text", "url", "place", "state", "score"} <= set(r) for r in rows)
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert {name for name, _, _ in idx.calls} == {"bm25_contextual", "dense_contextual"}


# ---- explicit filters (D85) -------------------------------------------------------------------

def test_explicit_filters_move_failing_chunks_behind_and_back_fill():
    r = R.Retriever(index=FakeIndex(), provenance=PROV)
    rows, trace = r.search("q", {"state": ["KL"]}, 8)
    ids = [x["chunk_id"] for x in rows]
    assert ids[:4] == list("efgh") and ids[4:] == list("abcd")  # Kochi first, rest back-filled
    assert trace["post"]["filters"]["rejected"] == {"state": 4}
    assert [x["chunk_id"] for x in r.search("q", {"state": ["KL"]}, 4)[0]] == list("efgh")


def test_an_unknown_filter_key_is_refused():
    with pytest.raises(ValueError, match="unknown filter keys"):
        R.Retriever(index=FakeIndex(), provenance=PROV).search("q", {"aspect": ["price"]}, 5)


def test_explicit_filters_with_a_reranked_pipeline_are_refused():
    r = R.Retriever(R.RetrieveConfig(name="rr", rerank=True), index=FakeIndex(), provenance=PROV,
                    scorer=lambda q, ids: {c: 0.0 for c in ids})
    with pytest.raises(NotImplementedError):
        r.search("q", {"state": ["KA"]}, 5)


# ---- configuration is validated ---------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"mmr_lambda": 0.9}, {"corrective_tau": 0.5},
                                {"expansion": True, "mmr_lambda": 0.9},
                                {"rerank": True, "mmr_lambda": 0.9},
                                {"rerank": True, "use_filters": True},
                                {"rerank": True, "fresh_w": 0.2},
                                {"expansion": True, "rerank": True, "mmr_lambda": 1.5},
                                {"fresh_w": -0.1}])
def test_combinations_that_were_not_ablated_are_refused(kw):
    with pytest.raises(ValueError):
        R.RetrieveConfig(name="bad", **kw)


def test_missing_parts_are_named():
    with pytest.raises(ValueError, match="parser"):
        R.Retriever(R.RetrieveConfig(name="e", expansion=True), index=FakeIndex())
    with pytest.raises(ValueError, match="scorer"):
        R.Retriever(R.RetrieveConfig(name="r", rerank=True), index=FakeIndex())
    with pytest.raises(ValueError, match="rewriter"):
        R.Retriever(R.FULL_CONFIG, index=FakeIndex(), parser=parse_for(), scorer=lambda q, i: {})


# ---- the optional components, as ablated ------------------------------------------------------

def test_expansion_runs_the_request_and_the_subqueries_and_passes_only_text_to_the_parser():
    seen = []

    def parser(q):
        seen.append(q)
        return parse_for()(q)

    idx = FakeIndex()
    r = R.Retriever(R.RetrieveConfig(name="e", expansion=True), index=idx, provenance=PROV,
                    parser=parser)
    r.search("the request", None, 5)
    assert seen == ["the request"]  # the text only: no id, no gold field
    assert {q for _, q, _ in idx.calls} == {"the request", "s1", "s2"}


def test_rerank_orders_the_head_by_the_scorer():
    logits = {c: float(ord(c)) for c in IDS}  # h highest
    r = R.Retriever(R.RetrieveConfig(name="rr", rerank=True), index=FakeIndex(), provenance=PROV,
                    scorer=lambda q, ids: {c: logits[c] for c in ids})
    assert [x["chunk_id"] for x in r.search("q", None, 8)[0]] == list("hgfedcba")


def mmr_cfg(lam):
    return R.RetrieveConfig(name="m", expansion=True, rerank=True, mmr_lambda=lam)


def test_mmr_lambda_one_keeps_the_reranked_order_and_lower_lambda_can_change_it():
    logits = {"a": 3.0, "b": 2.9, "c": 0.0, "d": -1.0, "e": -2.0, "f": -3.0, "g": -4.0, "h": -5.0}
    idx = FakeIndex()
    idx._m = np.array([[1, 0, 0]] * 2 + [[0, 1, 0], [0, 0, 1]] + [[0.5, 0.5, 0]] * 4,
                      dtype=np.float32)  # a and b are identical vectors
    common = dict(index=idx, provenance=PROV, parser=parse_for(places=("Hampi",)),
                  scorer=lambda q, ids: {c: logits[c] for c in ids})
    one = [x["chunk_id"] for x in R.Retriever(mmr_cfg(1.0), **common).search("q", None, 8)[0]]
    low = [x["chunk_id"] for x in R.Retriever(mmr_cfg(0.5), **common).search("q", None, 8)[0]]
    assert one == list("abcdefgh") and low != one and low.index("b") > low.index("c")


def test_the_corrective_pass_runs_once_at_depth_300_with_the_rewrite():
    idx = FakeIndex()
    asked = []

    def rewriter(q, triggers):
        asked.append((q, list(triggers)))
        return "a better query"

    cfg = R.RetrieveConfig(name="c", expansion=True, rerank=True, mmr_lambda=0.9,
                           corrective_tau=0.99)
    r = R.Retriever(cfg, index=idx, provenance=PROV, parser=parse_for(), rewriter=rewriter,
                    scorer=lambda q, ids: {c: -5.0 for c in ids})  # weak: low_confidence fires
    _, trace = r.search("q", None, 5)
    assert trace["passes"] == 1 and trace["rewrite"] == "a better query"
    assert "low_confidence" in trace["triggers"]
    assert asked == [("q", trace["triggers"])]
    assert any(q == "a better query" for _, q, _ in idx.calls)
    assert any(k == 300 for _, _, k in idx.calls) and len(asked) == 1  # never a second rewrite


def test_no_trigger_means_no_second_pass_and_no_rewrite_call():
    cfg = R.RetrieveConfig(name="c", expansion=True, rerank=True, mmr_lambda=0.9,
                           corrective_tau=0.0001)
    idx = FakeIndex()
    for i, c in enumerate(IDS):  # five different pages in the top five: few_sources stays quiet
        idx.chunks[c]["url"] = f"https://x.test/{i}"
    r = R.Retriever(cfg, index=idx, provenance=PROV, parser=parse_for(), rewriter=boom,
                    scorer=lambda q, ids: {c: 6.0 for c in ids})
    _, trace = r.search("q", None, 5)
    assert trace["passes"] == 0 and trace["triggers"] == []


def test_filters_follow_the_ablated_post_processing():
    cfg = R.RetrieveConfig(name="f", use_filters=True)
    rows, trace = R.Retriever(cfg, index=FakeIndex(), provenance=PROV,
                              parser=parse_for(sub=(), states=("KL",))).search("q", None, 4)
    assert [x["chunk_id"] for x in rows] == list("efgh") and "post" in trace


# ---- FULL_CONFIG outside the eval queries ----------------------------------------------------

def test_full_config_on_a_non_eval_query_raises_a_clear_error_not_keyerror():
    retriever = R.Retriever(R.FULL_CONFIG, index=FakeIndex(), provenance=PROV,
                            parser=R.parser_from_file(), scorer=boom,
                            rewriter=R.rewriter_from_file())
    with pytest.raises(R.QueryNotPrepared, match="eval queries") as err:
        retriever.search("a query that is not one of the forty eval queries", None, 10)
    assert not isinstance(err.value, KeyError)


def test_the_file_rewriter_raises_the_same_clear_error():
    with pytest.raises(R.QueryNotPrepared, match="rewrite") as err:
        R.rewriter_from_file()("not an eval query", ["low_confidence"])
    assert not isinstance(err.value, KeyError)
