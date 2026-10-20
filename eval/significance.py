"""Paired significance testing for the Phase 6 ablation table (D92; docs/papers/significance.md).

Pure, no I/O. The unit is a per-query metric (nDCG@10) for two retrievers on the SAME queries, so
every test is paired on the query. A query with no value under either system (e.g. no relevant
judged chunk, metric None) is dropped pairwise.

- randomization_test: two-sided paired sign-flip test. Exact enumeration of all 2^n sign patterns
  when n <= EXACT_MAX_N (dev, n=13 -> 8,192), else seeded Monte Carlo (test, n=27).
- bootstrap_ci: 95% paired bootstrap CI of the mean per-query difference.
- holm: Holm-Bonferroni step-down adjusted p-values over a family of comparisons.
- ttest_rel_p: paired t-test p-value (scipy) as a cross-check only, never the primary test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

EXACT_MAX_N = 20  # 2^20 ~ 1e6 patterns; dev (13) and anything smaller is enumerated exactly
DEFAULT_MC = 100_000
DEFAULT_BOOTSTRAP = 10_000
SEED = 42
_EPS = 1e-12


def paired_differences(a: dict[str, float | None], b: dict[str, float | None]) -> list[float]:
    """b - a over the query ids present in both with a non-None value in each, in sorted id order
    (so the result is reproducible)."""
    out = []
    for qid in sorted(set(a) & set(b)):
        va, vb = a[qid], b[qid]
        if va is not None and vb is not None:
            out.append(float(vb) - float(va))
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def randomization_test(diffs: list[float], *, exact_max_n: int = EXACT_MAX_N,
                       n_mc: int = DEFAULT_MC, seed: int = SEED) -> dict:
    """Two-sided paired sign-flip test on per-query differences. Returns {p, n, mean_diff, method,
    n_resamples}. H0: the sign of each difference is exchangeable; the statistic is |mean|. A diff
    of exactly 0 contributes 0 under any flip and never changes the statistic."""
    n = len(diffs)
    if n == 0:
        return {"p": 1.0, "n": 0, "mean_diff": 0.0, "method": "none", "n_resamples": 0}
    obs = abs(_mean(diffs))
    total = sum(diffs)
    if n <= exact_max_n:
        ge = 0
        for mask in range(1 << n):
            s = 0.0
            for i, d in enumerate(diffs):
                s += d if (mask >> i) & 1 else -d
            if abs(s) / n >= obs - _EPS:
                ge += 1
        return {"p": ge / (1 << n), "n": n, "mean_diff": total / n, "method": "exact",
                "n_resamples": 1 << n}
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_mc):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s) / n >= obs - _EPS:
            ge += 1
    return {"p": (ge + 1) / (n_mc + 1), "n": n, "mean_diff": total / n, "method": "monte_carlo",
            "n_resamples": n_mc}


def bootstrap_ci(diffs: list[float], *, n: int = DEFAULT_BOOTSTRAP, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile 95% CI of the mean difference by resampling query pairs with replacement."""
    m = len(diffs)
    if m == 0:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(m)] for _ in range(m)) / m)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-DOWN adjusted p-values, in the input order. For m sorted ascending
    p-values, the i-th (0-based) is multiplied by (m - i) and the running maximum is carried
    forward so the adjusted values stay monotone; each is capped at 1.0."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def ttest_rel_p(diffs: list[float]) -> float | None:
    """Paired t-test p-value (two-sided), a cross-check only (D92). None if scipy is absent or n<2
    or the differences have no spread."""
    if len(diffs) < 2:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    if max(diffs) - min(diffs) < _EPS:
        return None
    zeros = [0.0] * len(diffs)
    return float(stats.ttest_rel(diffs, zeros).pvalue)


@dataclass(frozen=True)
class Comparison:
    """One consecutive-row comparison before Holm adjustment."""

    name: str
    n: int
    mean_diff: float
    p: float
    method: str
    ci_low: float | None
    ci_high: float | None


def compare(name: str, a: dict[str, float | None], b: dict[str, float | None], *,
            n_mc: int = DEFAULT_MC, seed: int = SEED) -> Comparison:
    diffs = paired_differences(a, b)
    rt = randomization_test(diffs, n_mc=n_mc, seed=seed)
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return Comparison(name=name, n=rt["n"], mean_diff=rt["mean_diff"], p=rt["p"],
                      method=rt["method"], ci_low=lo, ci_high=hi)


def holm_over(comparisons: list[Comparison]) -> list[tuple[Comparison, float]]:
    """Pair each comparison with its Holm-adjusted p over the whole family."""
    adj = holm([c.p for c in comparisons])
    return list(zip(comparisons, adj))
