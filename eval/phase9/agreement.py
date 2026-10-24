"""Agreement between two scorers (D124). Pure functions, no I/O.

Scores are integers 1 to 5. Reported: exact agreement, within-one agreement and the quadratic
weighted Cohen's kappa (None when it is undefined because both scorers used one score only).
Binary agreement (supported 0/1) uses the same kappa with two labels."""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence


def _pairs(a: Mapping[Hashable, int], b: Mapping[Hashable, int]) -> list[tuple[int, int]]:
    return [(int(a[k]), int(b[k])) for k in sorted(set(a) & set(b), key=str)]


def exact(pairs: Sequence[tuple[int, int]]) -> float | None:
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else None


def within_one(pairs: Sequence[tuple[int, int]]) -> float | None:
    return sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs) if pairs else None


def weighted_kappa(pairs: Sequence[tuple[int, int]], labels: Sequence[int] = (1, 2, 3, 4, 5),
                   weights: str = "quadratic") -> float | None:
    """Cohen's kappa with agreement weights w = 1 - (d / (k - 1)) ** 2 (quadratic), 1 - d/(k - 1)
    (linear) or exact match (none), d the distance between the two labels' positions."""
    n = len(pairs)
    k = len(labels)
    if n == 0 or k < 2:
        return None
    pos = {lab: i for i, lab in enumerate(labels)}
    table = [[0] * k for _ in range(k)]
    for x, y in pairs:
        table[pos[x]][pos[y]] += 1
    rows = [sum(r) for r in table]
    cols = [sum(table[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1)
        if weights == "none":
            return float(i == j)
        return 1 - d if weights == "linear" else 1 - d * d

    po = sum(w(i, j) * table[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def binary_kappa(pairs: Sequence[tuple[int, int]]) -> float | None:
    return weighted_kappa(pairs, labels=(0, 1), weights="none")


def agreement(a: Mapping[Hashable, int], b: Mapping[Hashable, int]) -> dict:
    """Agreement of two scorers on the keys both scored."""
    pairs = _pairs(a, b)
    return {"n": len(pairs), "exact": exact(pairs), "within_one": within_one(pairs),
            "weighted_kappa": weighted_kappa(pairs)}


def agreement_by_dimension(a: Mapping[Hashable, Mapping[str, int]],
                           b: Mapping[Hashable, Mapping[str, int]],
                           dims: Sequence[str]) -> dict:
    """{dimension: agreement, "pooled": agreement over all dimensions}. Keys are rows (the
    request and system pair); each value maps a dimension to a score."""
    out: dict = {}
    pooled_a: dict = {}
    pooled_b: dict = {}
    for d in dims:
        ka = {k: v[d] for k, v in a.items() if d in v}
        kb = {k: v[d] for k, v in b.items() if d in v}
        out[d] = agreement(ka, kb)
        for k in set(ka) & set(kb):
            pooled_a[(k, d)], pooled_b[(k, d)] = ka[k], kb[k]
    out["pooled"] = agreement(pooled_a, pooled_b)
    return out
