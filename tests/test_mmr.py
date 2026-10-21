"""MMR (src/retrieval/mmr.py, D89) and the reranker's pure part (src/retrieval/rerank.py, D88)."""
import math

import numpy as np
import pytest

from src.retrieval import mmr, rerank

# a and b are near-duplicates (sim 0.9); c is different (sim 0.1 to both)
SIMS = {frozenset("ab"): 0.9, frozenset("ac"): 0.1, frozenset("bc"): 0.1}
REL = {"a": 0.9, "b": 0.85, "c": 0.5}


def sim(x, y):
    return SIMS[frozenset((x, y))]


def test_lambda_one_is_pure_relevance_and_keeps_the_input_order():
    assert mmr.mmr_order(["a", "b", "c"], REL, sim, 1.0) == ["a", "b", "c"]


def test_known_answer_at_lambda_half_puts_the_different_chunk_before_the_duplicate():
    # step 1: a (0.45 vs 0.425 vs 0.25). step 2: b = .425-.45 = -0.025 ; c = .25-.05 = 0.2 -> c
    assert mmr.mmr_order(["a", "b", "c"], REL, sim, 0.5) == ["a", "c", "b"]


def test_lambda_zero_picks_by_redundancy_only_with_ties_to_the_prior_rank():
    assert mmr.mmr_order(["a", "b", "c"], REL, sim, 0.0) == ["a", "c", "b"]


def test_a_lambda_outside_zero_one_is_refused():
    with pytest.raises(ValueError):
        mmr.mmr_order(["a"], {"a": 1.0}, sim, 1.2)


def test_the_first_pick_is_the_most_relevant_even_if_the_input_is_not_sorted():
    assert mmr.mmr_order(["c", "a", "b"], REL, sim, 0.8)[0] == "a"


def test_ties_go_to_the_better_prior_rank_then_the_id():
    flat = {"x": 0.5, "y": 0.5}
    assert mmr.mmr_order(["y", "x"], flat, lambda a, b: 0.0, 0.7) == ["y", "x"]


def test_sigmoid_known_values_and_stability():
    assert mmr.sigmoid(0.0) == 0.5
    assert mmr.sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2)))
    assert mmr.sigmoid(-1000) == pytest.approx(0.0, abs=1e-12)  # no overflow
    assert mmr.sigmoid(1000) == pytest.approx(1.0)


def chunks():
    return {"a": {"source_type": "blog", "place": "Hampi"},
            "b": {"source_type": "blog", "place": "Hampi"},
            "c": {"source_type": "wikivoyage", "place": "Kochi"}}


MATRIX = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
ROW = {"a": 0, "b": 1, "c": 2}


def test_similarity_weights_cosine_source_type_and_place():
    s = mmr.make_similarity(MATRIX, ROW, chunks(), place_term=True)
    assert s("a", "b") == pytest.approx(0.6 * 0.6 + 0.2 + 0.2)  # cos .6, same source, same place
    assert s("a", "c") == pytest.approx(0.0)  # cos 0, different source, different place


def test_the_place_term_is_off_for_place_named_queries():
    on = mmr.make_similarity(MATRIX, ROW, chunks(), place_term=True)
    off = mmr.make_similarity(MATRIX, ROW, chunks(), place_term=False)
    assert on("a", "b") - off("a", "b") == pytest.approx(0.2)


def test_apply_mmr_reorders_the_head_and_leaves_the_tail():
    ranked = [("a", 5.0), ("b", 4.0), ("c", 3.0), ("t1", 2.0), ("t2", 1.0)]
    logits = {"a": 2.0, "b": 1.8, "c": 0.0}
    out = mmr.apply_mmr(ranked, logits, sim, 0.5, depth=3)
    assert [c for c, _ in out] == ["a", "c", "b", "t1", "t2"]
    assert [s for _, s in out[:3]] == [3.0, 2.0, 1.0]  # descending rank scores
    assert out[3:] == ranked[3:]


def test_select_lambda_takes_the_smallest_within_tolerance():
    curve = {1.0: 0.80, 0.9: 0.79, 0.8: 0.785, 0.7: 0.77, 0.5: 0.70}
    assert mmr.select_lambda(curve) == 0.8  # 0.785 >= 0.78 ; 0.77 < 0.78
    assert mmr.select_lambda({1.0: 0.8, 0.5: 0.5}) == 1.0
    assert mmr.select_lambda({1.0: 0.8, 0.5: 0.79, 0.7: None}) == 0.5


def test_select_lambda_needs_the_baseline():
    with pytest.raises(ValueError):
        mmr.select_lambda({0.9: 0.7})


# ---- rerank.apply_scores -----------------------------------------------------------------

def test_apply_scores_orders_the_head_by_logit_and_keeps_the_tail_order():
    ranked = [("a", 3.0), ("b", 2.0), ("c", 1.0), ("t1", 0.5), ("t2", 0.4)]
    out = rerank.apply_scores(ranked, {"a": -1.0, "b": 2.0, "c": 0.5}, depth=3)
    assert [c for c, _ in out] == ["b", "c", "a", "t1", "t2"]
    assert out[0] == ("b", 2.0) and out[3:] == ranked[3:]


def test_apply_scores_ties_keep_the_prior_rank():
    out = rerank.apply_scores([("z", 2.0), ("a", 1.0)], {"z": 1.0, "a": 1.0}, depth=2)
    assert [c for c, _ in out] == ["z", "a"]


def test_apply_scores_requires_a_score_for_every_head_candidate():
    with pytest.raises(KeyError):
        rerank.apply_scores([("a", 1.0), ("b", 0.5)], {"a": 1.0}, depth=2)


def test_apply_scores_default_depth_is_30():
    assert rerank.DEPTH == 30
    ranked = [(f"c{i:02d}", 1.0 - i / 100) for i in range(40)]
    logits = {cid: float(-i) for i, (cid, _) in enumerate(ranked[:30])}
    logits["c29"] = 100.0
    out = rerank.apply_scores(ranked, logits)
    assert out[0][0] == "c29" and [c for c, _ in out[30:]] == [c for c, _ in ranked[30:]]


# ---- CrossEncoderScorer (fake model, no torch) -------------------------------------------------

from src.common.cache import DiskCache  # noqa: E402

CHUNKS = {"c1": {"context_note": "note one", "text": "text one"},
          "c2": {"context_note": "note two", "text": "text two"},
          "c3": {"context_note": "note three", "text": "text three"}}


def make_dir(tmp_path, weights=b"w1"):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_bytes(b"{}")
    (d / "pytorch_model.bin").write_bytes(weights)
    return d


class FakeModel:
    """Scores a passage by its length; records every call."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, query, passages):
        self.calls.append((query, list(passages)))
        return [float(len(p)) for p in passages]


def scorer(tmp_path, model, **kw):
    kw.setdefault("score_fn", model)
    return rerank.CrossEncoderScorer(make_dir(tmp_path, kw.pop("weights", b"w1")), CHUNKS,
                                     DiskCache(tmp_path / "cache"), **kw)


def test_scores_are_the_raw_logits_for_the_requested_chunks(tmp_path):
    m = FakeModel()
    out = scorer(tmp_path, m)("q", ["c1", "c2"])
    want = {"c1": len("note one\n\ntext one"), "c2": len("note two\n\ntext two")}
    assert out == {k: float(v) for k, v in want.items()}


def test_the_passage_is_note_then_text_for_contextual_and_text_for_plain(tmp_path):
    m = FakeModel()
    scorer(tmp_path, m)("q", ["c1"])
    assert m.calls[0][1] == ["note one\n\ntext one"]
    m2 = FakeModel()
    scorer(tmp_path, m2, text_kind="plain")("q", ["c1"])
    assert m2.calls[0][1] == ["text one"]
    with pytest.raises(ValueError):
        rerank.passage_text(CHUNKS["c1"], "other")


def test_a_second_call_is_served_from_the_cache_and_never_touches_the_model(tmp_path):
    m = FakeModel()
    s = scorer(tmp_path, m)
    first = s("q", ["c1", "c2", "c3"])
    second = s("q", ["c3", "c1"])
    assert len(m.calls) == 1 and second == {"c3": first["c3"], "c1": first["c1"]}
    assert (s.n_scored, s.n_cached) == (3, 2)


def test_only_the_missing_pairs_are_scored(tmp_path):
    m = FakeModel()
    s = scorer(tmp_path, m)
    s("q", ["c1"])
    s("q", ["c1", "c2"])
    assert len(m.calls) == 2 and m.calls[1][1] == ["note two\n\ntext two"]


def test_another_query_does_not_reuse_a_score(tmp_path):
    m = FakeModel()
    s = scorer(tmp_path, m)
    s("q1", ["c1"])
    s("q2", ["c1"])
    assert len(m.calls) == 2


def test_other_weights_never_reuse_cached_scores(tmp_path):
    m1, m2 = FakeModel(), FakeModel()
    scorer(tmp_path, m1, weights=b"AAAA")("q", ["c1"])
    s2 = scorer(tmp_path, m2, weights=b"BBBB")
    s2("q", ["c1"])
    assert len(m2.calls) == 1  # same cache folder, different weight hash -> a miss
    assert rerank.model_id_from({"a": "1"}) != rerank.model_id_from({"a": "2"})


def test_a_changed_passage_text_is_a_miss(tmp_path):
    m = FakeModel()
    s = scorer(tmp_path, m)
    s("q", ["c1"])
    CHUNKS["c1"]["text"] = "edited text"
    try:
        s("q", ["c1"])
    finally:
        CHUNKS["c1"]["text"] = "text one"
    assert len(m.calls) == 2


def test_the_model_is_never_loaded_without_a_cache_miss(tmp_path, monkeypatch):
    s = scorer(tmp_path, FakeModel())
    s("q", ["c1"])
    s2 = rerank.CrossEncoderScorer(s.model_dir, CHUNKS, s.cache, score_fn=None)
    monkeypatch.setattr(rerank, "load_model", lambda d: (_ for _ in ()).throw(AssertionError))
    assert s2("q", ["c1"])  # fully cached: load_model is never reached


def test_a_wrong_number_of_scores_is_an_error_and_nothing_is_cached(tmp_path):
    s = scorer(tmp_path, lambda q, ps: [1.0])
    with pytest.raises(RuntimeError):
        s("q", ["c1", "c2"])
    assert s.cache.get(rerank.CACHE_KIND, s.model_id, s._key("q", "c1", "note one\n\ntext one")) \
        is None


def test_an_empty_model_folder_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        rerank.CrossEncoderScorer(tmp_path / "empty", CHUNKS, DiskCache(tmp_path / "c"))


def test_weight_hashes_cover_the_files_that_exist(tmp_path):
    d = make_dir(tmp_path)
    h = rerank.weight_hashes(d)
    assert set(h) == {"config.json", "pytorch_model.bin"}
    assert h["config.json"] == rerank.sha256_file(d / "config.json")
