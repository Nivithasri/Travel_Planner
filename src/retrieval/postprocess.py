"""Filters then freshness over a ranked candidate list: rows 5 and 6 (D85, D87). Pure.

    candidates (ranked) -> filters.partition -> passing -> freshness.rerank -> passing + rejected

The rejected candidates stay behind the passing ones in rank order, so a query with fewer than k
passing candidates is topped up from the unfiltered ranking (back-fill). The trace reports the
filters used, what each rejected, how many place_type checks were skipped for rule-based rows
(D69), the freshness weight actually applied, and how many results were back-filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.retrieval import freshness
from src.retrieval.filters import Filters, partition


@dataclass(frozen=True)
class PostConfig:
    use_filters: bool = False
    rule_based_place_type: bool = False  # diagnostic 5b: filter place_type on every row
    fresh_w: float = 0.0  # 0 means freshness is off


def postprocess(ranked: list[tuple[str, float]], chunks: dict[str, dict],
                provenance: dict[str, str], parsed: dict, config: PostConfig, k: int,
                reference: date | None = None, explicit_filters: dict | None = None
                ) -> tuple[list[tuple[str, float]], dict]:
    """Returns (top-k (chunk_id, score), trace). `parsed` is a validated parse
    (ValidatedQuery.to_dict()); only its states, season, month, place_types and time_sensitive
    fields are read."""
    trace: dict = {}
    rejected: list[tuple[str, float]] = []
    passing = list(ranked)
    f = Filters.from_parsed(parsed, explicit_filters)
    if config.use_filters or explicit_filters:
        passing, rejected, trace["filters"] = partition(
            ranked, chunks, provenance, f, rule_based_place_type=config.rule_based_place_type)
    if config.fresh_w > 0:
        if reference is None:
            raise ValueError("freshness needs the reference date (EVAL_REFERENCE_DATE, D4)")
        passing, trace["freshness"] = freshness.rerank(
            passing, chunks, reference, config.fresh_w, bool(parsed.get("time_sensitive")))
    out = passing + rejected
    trace["backfilled"] = min(len(rejected), max(0, k - len(passing)))
    return out[:k], trace
