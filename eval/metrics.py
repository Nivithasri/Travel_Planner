"""Ranking metrics for the Phase 4 harness (D73). Pure Python, no I/O.

Conventions (docs/decisions.md D73):
- `ranked` is a list of chunk ids, best first; a repeated id keeps its first position only.
- `qrel` is {chunk_id: 0|1|2} for ONE query. A chunk with no label is unjudged and counts as 0.
- Relevant means label >= `threshold` (default 1).
- nDCG uses linear gain (the label); the ideal ranking uses every judged chunk.
- nDCG, MRR and Recall return None for a query with no relevant chunk; callers exclude those from
  means and count them. P@k is defined for every query (its denominator is always k).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from datetime import date

DEFAULT_THRESHOLD = 1


def dedupe(ranked: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for cid in ranked:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def grade(qrel: dict[str, int], chunk_id: str) -> int:
    return int(qrel.get(chunk_id, 0))


def n_relevant(qrel: dict[str, int], threshold: int = DEFAULT_THRESHOLD) -> int:
    return sum(1 for v in qrel.values() if v >= threshold)


def has_relevant(qrel: dict[str, int], threshold: int = DEFAULT_THRESHOLD) -> bool:
    return n_relevant(qrel, threshold) > 0


def _dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int = 10,
              threshold: int = DEFAULT_THRESHOLD) -> float | None:
    if not has_relevant(qrel, threshold):
        return None
    top = dedupe(ranked)[:k]
    ideal = sorted((int(v) for v in qrel.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return None
    return _dcg([grade(qrel, c) for c in top]) / ideal_dcg


def reciprocal_rank(ranked: list[str], qrel: dict[str, int], k: int | None = None,
                    threshold: int = DEFAULT_THRESHOLD) -> float | None:
    if not has_relevant(qrel, threshold):
        return None
    top = dedupe(ranked)[:k] if k else dedupe(ranked)
    for i, cid in enumerate(top, 1):
        if grade(qrel, cid) >= threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: list[str], qrel: dict[str, int], k: int = 5,
                   threshold: int = DEFAULT_THRESHOLD) -> float:
    top = dedupe(ranked)[:k]
    return sum(1 for c in top if grade(qrel, c) >= threshold) / k


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int,
                threshold: int = DEFAULT_THRESHOLD) -> float | None:
    total = n_relevant(qrel, threshold)
    if total == 0:
        return None
    top = dedupe(ranked)[:k]
    return sum(1 for c in top if grade(qrel, c) >= threshold) / total


def unjudged_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    """Share of the returned top-k that has no label (0.0 when nothing was returned)."""
    top = dedupe(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if c not in qrel) / len(top)


def months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + (d.month - 1) - months, 12)
    m += 1
    last = [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31][m - 1]
    return date(y, m, min(d.day, last))


def freshness_share(rows: list[dict], reference: date, months: int = 18) -> float:
    """Fraction of `rows` with date_kind == "published" and a date inside the window
    [reference - months, reference]. Revised and sitemap dates never count (D48)."""
    if not rows:
        return 0.0
    start = months_before(reference, months)
    n = 0
    for r in rows:
        if r.get("date_kind") != "published" or not r.get("date"):
            continue
        if start <= date.fromisoformat(r["date"]) <= reference:
            n += 1
    return n / len(rows)


def diversity(rows: list[dict]) -> dict[str, int]:
    return {"source_types": len({r.get("source_type") for r in rows}),
            "places": len({r.get("place") for r in rows})}


def excluded_share(rows: list[dict], excluded: set[str]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("place") in excluded) / len(rows)


def provenance_counts(rows: list[dict], qrel: dict[str, int],
                      threshold: int = DEFAULT_THRESHOLD) -> dict[str, dict[str, int]]:
    """Per provenance class: results, judged results, relevant results, grade sum over judged
    results. Summed over queries by the caller (D69)."""
    out: dict[str, Counter] = {}
    for r in rows:
        c = out.setdefault(r.get("provenance") or "unknown", Counter())
        c["n"] += 1
        cid = r["chunk_id"]
        if cid in qrel:
            c["judged"] += 1
            c["grade_sum"] += int(qrel[cid])
            if qrel[cid] >= threshold:
                c["relevant"] += 1
    return {k: dict(v) for k, v in out.items()}


def mean(values: Iterable[float | None]) -> tuple[float | None, int]:
    """(mean over the non-None values or None if there are none, number of None values)."""
    items = list(values)
    vals = [v for v in items if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(items) - len(vals)
