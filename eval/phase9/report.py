"""Phase 9 tables: paired system comparisons, the live comparison, latency and case candidates.

Everything here is a pure function of saved records and scores, so every committed number is
reproducible by `python scripts/phase9.py report`. Nothing in this module returns answer text or
source excerpts: only scores, counts and ids (D100, O12).
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from eval.phase9.sheets import DIMENSIONS
from eval.significance import (
    Comparison,
    bootstrap_ci,
    holm_over,
    paired_differences,
    randomization_test,
)

SYSTEM_PAIRS = (("full", "plain_rag"), ("full", "llm_only"), ("plain_rag", "llm_only"))


def per_system(table: dict[tuple[str, str], dict[str, int]]) -> dict[str, dict[str, dict]]:
    """{system: {dimension: {request_id: score}}} from {(request_id, system): scores}."""
    out: dict[str, dict[str, dict]] = {}
    for (rid, system), scores in table.items():
        for d, v in scores.items():
            out.setdefault(system, {}).setdefault(d, {})[rid] = v
    return out


def means(by_system: dict[str, dict[str, dict]], dims: Sequence[str] = DIMENSIONS) -> list[dict]:
    rows = []
    for system, dd in sorted(by_system.items()):
        row = {"system": system, "n": max((len(v) for v in dd.values()), default=0)}
        for d in dims:
            vals = list(dd.get(d, {}).values())
            row[d] = round(statistics.fmean(vals), 2) if vals else None
        rows.append(row)
    return rows


def compare_pairs(by_system: dict[str, dict[str, dict]], pairs: Iterable[tuple[str, str]],
                  dims: Sequence[str] = DIMENSIONS, *, n_mc: int = 100_000) -> list[dict]:
    """For each (a, b) and dimension: the mean of a minus b, paired by request, with a 95% bootstrap
    CI, the sign-flip p and the Holm-adjusted p over the whole family."""
    comps: list[Comparison] = []
    labels: list[tuple[str, str, str]] = []
    for a, b in pairs:
        for d in dims:
            sa, sb = by_system.get(a, {}).get(d, {}), by_system.get(b, {}).get(d, {})
            diffs = paired_differences(sb, sa)  # a minus b
            rt = randomization_test(diffs, n_mc=n_mc)
            lo, hi = bootstrap_ci(diffs)
            comps.append(Comparison(f"{a} - {b} / {d}", rt["n"], rt["mean_diff"], rt["p"],
                                    rt["method"], lo, hi))
            labels.append((a, b, d))
    rows = []
    for (c, adj), (a, b, d) in zip(holm_over(comps), labels, strict=True):
        rows.append({"first": a, "second": b, "dimension": d, "n": c.n,
                     "mean_diff": round(c.mean_diff, 3),
                     "ci": [None if c.ci_low is None else round(c.ci_low, 3),
                            None if c.ci_high is None else round(c.ci_high, 3)],
                     "p": round(c.p, 4), "p_holm": round(adj, 4)})
    return rows


def live_summary(records: Iterable[dict]) -> dict:
    """Counts of the live arm (D126): how often the step returned results, what reached the writer
    and what was cited. Counts only."""
    recs = [r for r in records if r["arm"] == "full_live"]
    dates = sorted({r["meta"]["utc_date"] for r in recs})
    status: dict[str, int] = {}
    keys = ("searches", "results", "in_scope", "in_plan", "dropped_excluded", "merged", "cited",
            "not_cited")
    tot = dict.fromkeys(keys, 0)
    for r in recs:
        live = r.get("live") or {}
        status[live.get("status", "none")] = status.get(live.get("status", "none"), 0) + 1
        for k in keys:
            tot[k] += int(live.get(k, 0) or 0)
    return {"requests": len(recs), "utc_dates": dates, "same_day": len(dates) <= 1,
            "status": status, "totals": tot,
            "requests_with_live_cited": sum(1 for r in recs
                                            if int((r.get("live") or {}).get("cited", 0) or 0) > 0)}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def latency_table(records: Iterable[dict]) -> list[dict]:
    """Median and p90 per node and total, per arm; `limiter_wait` is reported as its own row when
    the records carry it (D127)."""
    by_arm: dict[str, dict[str, list[float]]] = {}
    for r in records:
        nodes = by_arm.setdefault(r["arm"], {})
        for node, secs in r.get("timings", {}).items():
            nodes.setdefault(node, []).append(float(secs))
        total = r.get("meta", {}).get("wall_s")
        if total is not None:
            nodes.setdefault("total", []).append(float(total))
        wait = r.get("meta", {}).get("limiter_wait_s")
        if wait is not None:
            nodes.setdefault("limiter_wait", []).append(float(wait))
    rows = []
    for arm in sorted(by_arm):
        for node, vals in sorted(by_arm[arm].items()):
            rows.append({"arm": arm, "node": node, "n": len(vals),
                         "median_s": round(statistics.median(vals), 2),
                         "p90_s": round(percentile(vals, 90), 2)})
    return rows


def case_candidates(records: Iterable[dict], table: dict[tuple[str, str], dict[str, int]]) -> dict:
    """Request ids to read for the case studies (ids only, no text): failures are the full
    system's lowest-scoring or critic-failed requests; live cases are the requests whose live arm
    cited a live item or whose scores differ from the corpus arm."""
    recs = {(r["request_id"], r["arm"]): r for r in records}
    failures, live_cases = [], []
    for (rid, arm), r in sorted(recs.items()):
        if arm == "full_corpus" and (r.get("critic") == "failed_rules"
                                     or r["meta"].get("repaired")):
            failures.append({"request_id": rid, "why": f"critic {r.get('critic')}"})
        if arm == "full_live":
            cited = int((r.get("live") or {}).get("cited", 0) or 0)
            a, b = table.get((rid, "full_live")), table.get((rid, "full_corpus"))
            changed = bool(a and b and a != b)
            if cited or changed:
                live_cases.append({"request_id": rid, "live_cited": cited,
                                   "scores_changed": changed})
    low = sorted(((sum(v.values()), rid) for (rid, system), v in table.items()
                  if system == "full"))[:4]
    failures += [{"request_id": rid, "why": f"lowest total score {tot}"} for tot, rid in low]
    return {"failures": failures, "live_cases": live_cases}


def md_table(rows: list[dict], cols: Sequence[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |\n"
                   for r in rows)
    return head + body
