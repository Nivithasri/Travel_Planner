"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009; docs/papers/rrf.md).

RRF(d) = sum over the input rankings of 1 / (k + rank(d)), rank counted from 1, with k = 60. A
document absent from a ranking contributes nothing from it. Ties are broken by chunk id so the
fused order is reproducible (our addition, not the paper's). Pure; no I/O.
"""
from __future__ import annotations

from collections import defaultdict

RRF_K = 60


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked chunk-id lists into one (chunk_id, score) list, best first."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))
