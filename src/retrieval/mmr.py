"""Maximal Marginal Relevance over the reranked top 30 (D89; docs/papers/mmr.md). Pure.

    pick next = argmax over the remaining candidates of  lam * rel(d) - (1 - lam) * max_s sim(d, s)

s ranges over the already selected chunks (the max is 0 while nothing is selected). rel is the
sigmoid of the cross-encoder logit, so it lies in (0, 1). sim is document-to-document:

    sim = 0.6 * cos(dense_contextual) + 0.2 * [same source_type] + 0.2 * [same place]

and the place term is 0 when the parsed query names a place: for a place-named query another
place's chunk is not relevant (labelling rule 1), so "different place" must not count as diversity.
Ties go to the better prior rank, then to the chunk id.

SELECTION of lam (fixed before any run, D89): the smallest lam in LAMBDAS whose dev nDCG@10 is
within TOLERANCE of lam = 1.0. The popular/hidden-gem axis cannot be evaluated (0 hidden_gem in the
frozen corpus, D44); only source-type and place diversity are measured.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.5)
TOLERANCE = 0.02
SIM_WEIGHTS = {"cosine": 0.6, "source_type": 0.2, "place": 0.2}

Similarity = Callable[[str, str], float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def make_similarity(matrix: np.ndarray, row_of: dict[str, int], chunks: dict[str, dict], *,
                    place_term: bool) -> Similarity:
    """Document-document similarity from unit-norm dense vectors and two metadata matches."""
    w = SIM_WEIGHTS

    def sim(a: str, b: str) -> float:
        cos = float(matrix[row_of[a]] @ matrix[row_of[b]])
        same_src = chunks[a]["source_type"] == chunks[b]["source_type"]
        same_place = place_term and chunks[a]["place"] == chunks[b]["place"]
        return w["cosine"] * cos + w["source_type"] * same_src + w["place"] * bool(same_place)

    return sim


def mmr_order(ids: list[str], rel: dict[str, float], sim: Similarity, lam: float) -> list[str]:
    """All of `ids` in MMR selection order. `ids` is the reranked order (best first); with
    lam = 1 and rel non-increasing along it, the order is unchanged."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    prior = {cid: i for i, cid in enumerate(ids)}
    remaining = list(ids)
    selected: list[str] = []
    while remaining:
        def key(cid: str) -> tuple[float, int, str]:
            redundancy = max((sim(cid, s) for s in selected), default=0.0)
            return (-(lam * rel[cid] - (1 - lam) * redundancy), prior[cid], cid)

        best = min(remaining, key=key)
        selected.append(best)
        remaining.remove(best)
    return selected


def apply_mmr(ranked: list[tuple[str, float]], logits: dict[str, float], sim: Similarity,
              lam: float, depth: int = 30) -> list[tuple[str, float]]:
    """Re-order the first `depth` of `ranked` (already reranked, scores = logits) by MMR and keep
    the rest below them unchanged. The returned score of a head chunk is its MMR rank score
    (descending: depth for the first pick down to 1), so downstream sorts by score stay valid."""
    head, tail = ranked[:depth], ranked[depth:]
    ids = [cid for cid, _ in head]
    rel = {cid: sigmoid(logits[cid]) for cid in ids}
    order = mmr_order(ids, rel, sim, lam)
    n = len(order)
    return [(cid, float(n - i)) for i, cid in enumerate(order)] + tail


def select_lambda(ndcg_by_lambda: dict[float, float | None], tolerance: float = TOLERANCE
                  ) -> float:
    """The smallest lam whose nDCG@10 is within `tolerance` of the lam = 1.0 value. Requires the
    lam = 1.0 entry; a lam with no value (None) is skipped."""
    if ndcg_by_lambda.get(1.0) is None:
        raise ValueError("the lam = 1.0 value is needed")
    ref = ndcg_by_lambda[1.0]
    floor = ref - tolerance - 1e-12
    return min(lam for lam, v in ndcg_by_lambda.items() if v is not None and v >= floor)
