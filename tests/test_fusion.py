"""RRF (src/retrieval/fusion.py, D72/D84). Behaviour must match the old eval.adapters.rrf."""
import pytest

from eval import adapters
from src.retrieval import fusion


def test_rrf_known_scores_and_order():
    # a: 1/61 ; b: 1/62 + 1/61 ; c: 1/62
    out = fusion.rrf([["a", "b"], ["b", "c"]], k=60)
    assert [c for c, _ in out] == ["b", "a", "c"]
    assert out[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert out[1][1] == pytest.approx(1 / 61)


def test_rrf_default_k_is_60():
    assert fusion.RRF_K == 60
    assert fusion.rrf([["a"]])[0][1] == pytest.approx(1 / 61)


def test_rrf_ties_break_by_chunk_id_ascending():
    assert [c for c, _ in fusion.rrf([["z"], ["a"]], k=60)] == ["a", "z"]


def test_rank_counts_from_one_not_zero():
    # if ranks started at 0 the first item would score 1/60, not 1/61
    assert fusion.rrf([["a"]], k=60)[0][1] == pytest.approx(1 / (60 + 1))


def test_adapters_reexports_the_same_function():
    assert adapters.rrf is fusion.rrf
