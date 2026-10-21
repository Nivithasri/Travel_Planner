"""Cross-encoder reranking, the model-independent part (D88; docs/papers/cross_encoder.md). Pure.

The reranker scores (query, chunk) pairs; `apply_scores` re-orders the top `depth` candidates by
those scores and leaves everything below them where it was (plan.md item 7: rerank the top 30,
ranks 31-100 keep their order). The model itself (bge-reranker-v2-m3, local, offline, cached per
(query, chunk)) is added to this module with row 7.

A scorer is any callable (query: str, chunk_ids: list[str]) -> {chunk_id: logit}. Tests and the
MMR wiring check use fakes; no model is needed here.
"""
from __future__ import annotations

from collections.abc import Callable

DEPTH = 30
Scorer = Callable[[str, list[str]], dict[str, float]]


def apply_scores(ranked: list[tuple[str, float]], logits: dict[str, float], depth: int = DEPTH
                 ) -> list[tuple[str, float]]:
    """The first `depth` of `ranked`, ordered by logit descending (ties by their prior rank, then
    chunk id), carrying the LOGIT as the new score; the rest unchanged below them. Every
    candidate in the head must have a score."""
    head, tail = ranked[:depth], ranked[depth:]
    missing = [cid for cid, _ in head if cid not in logits]
    if missing:
        raise KeyError(f"no reranker score for {missing[:3]}")
    pos = {cid: i for i, (cid, _) in enumerate(head)}
    order = sorted(head, key=lambda t: (-logits[t[0]], pos[t[0]], t[0]))
    return [(cid, logits[cid]) for cid, _ in order] + tail
