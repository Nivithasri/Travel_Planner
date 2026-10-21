"""Aspect-aware, query-aware freshness ranking (D87; docs/papers/time_decay.md). Pure.

    final = rel_norm * (1 - w_q) + w_q * f

- rel_norm: min-max of the candidates' relevance scores (equal scores all map to 1.0).
- w_q = w when the PARSED query is time_sensitive, else 0 (the gold flag is never read).
- f = 0.5 ** (age_days / half_life(aspect)) for date_kind == "published" only. Every other chunk
  (no date, a revision date, a sitemap date) gets the NEUTRAL f = 0.5, never "old" (D3, D48), and
  so does an aspect without decay (scenery, history).

Half-lives, fixed before any run: closure, timing, price, safety 180 d; food, other 730 d;
scenery, history none. MEASURED CAVEAT (D87): in the frozen corpus every published chunk is a blog
chunk with a rule-based aspect, and none is inside the 18-month window, so a dated chunk is OLD
(f < 0.5) and the row can only push old blog chunks below undated wiki chunks; it cannot show a
freshness gain. With w_q = 0 the order is unchanged (ties by chunk_id, as everywhere).
"""
from __future__ import annotations

from datetime import date

HALF_LIFE_DAYS: dict[str, float | None] = {
    "closure": 180.0, "timing": 180.0, "price": 180.0, "safety": 180.0,
    "food": 730.0, "other": 730.0, "scenery": None, "history": None,
}
NEUTRAL = 0.5
GRID = (0.0, 0.1, 0.2, 0.3)  # w, tuned on dev only (D87)


def freshness_factor(chunk: dict, reference: date) -> tuple[float, bool]:
    """(f, decayed). decayed is False for the neutral cases."""
    if chunk.get("date_kind") != "published" or not chunk.get("date"):
        return NEUTRAL, False
    half_life = HALF_LIFE_DAYS.get(chunk.get("aspect") or "other", HALF_LIFE_DAYS["other"])
    if half_life is None:
        return NEUTRAL, False
    age = max(0, (reference - date.fromisoformat(chunk["date"])).days)
    return 0.5 ** (age / half_life), True


def rel_normalise(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def rerank(ranked: list[tuple[str, float]], chunks: dict[str, dict], reference: date,
           w: float, time_sensitive: bool) -> tuple[list[tuple[str, float]], dict]:
    """Re-score `ranked` ((chunk_id, relevance), best first) and re-sort. Returns the new
    (chunk_id, final score) list and a trace {"w_q", "decayed": n chunks that got a decay factor,
    "moved": n chunks whose position changed}."""
    w_q = w if time_sensitive else 0.0
    if not ranked:
        return [], {"w_q": w_q, "decayed": 0, "moved": 0}
    norm = rel_normalise([s for _, s in ranked])
    out = []
    decayed = 0
    for (cid, _), r in zip(ranked, norm):
        f, did = freshness_factor(chunks[cid], reference)
        decayed += did
        out.append((cid, r * (1 - w_q) + w_q * f))
    out.sort(key=lambda t: (-t[1], t[0]))
    moved = sum(1 for a, b in zip(ranked, out) if a[0] != b[0])
    return out, {"w_q": w_q, "decayed": decayed, "moved": moved}
