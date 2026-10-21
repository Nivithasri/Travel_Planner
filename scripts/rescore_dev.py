"""Re-score the dev rows with the MERGED judgments (round 1 human + round 2 chat, D91, D95) and
re-apply the keep rules (D84, D87, D89, D90). Dev only; offline (reranker scores and rewrites come
from their caches); reads qrels_dev and qrels_dev_r2, never the test judgments.

    python scripts/rescore_dev.py            # writes eval/ablation/dev_merged.{md,json}

Every row is scored twice from ONE set of rankings (memoised): against round 1 alone and against
the merged judgments, so the effect of round 2 is visible row by row. The rules are applied as
written BEFORE this run:
  * D84 keep rule: a component stays when its dev nDCG@10 delta over its predecessor is >= -0.005;
  * D87: the freshness weight is the one with the highest dev nDCG@10 (ties to the smaller);
  * D89: lambda is the smallest in {1.0, 0.9, 0.8, 0.7, 0.5} within 0.02 of lambda = 1.0;
  * D90: tau is the candidate with the highest dev nDCG@10 (ties to the smaller).
It REPORTS what the rules pick on the merged judgments and compares with the frozen choices
(eval/mmr_lambda.json, eval/corrective_tau.json); it never overwrites them. Changing a chosen
parameter is a decision for the owner.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import ablation as A  # noqa: E402
from eval import variants as V  # noqa: E402
from eval.qrels import load_qrels, load_qrels_merged, load_queries  # noqa: E402
from eval.run_eval import (  # noqa: E402
    ENRICHED,
    PROVENANCE,
    evaluate,
    load_chunk_info,
    load_excluded,
    load_retriever,
)
from scripts.tau_curve import choose_tau as choose_highest  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402
from src.retrieval import mmr  # noqa: E402

KEEP_TOLERANCE = 0.005  # D84
OUT = REPO_ROOT / "eval" / "ablation"
CHAIN = [("hybrid_plain", "row 2"), ("hybrid_contextual", "row 3"), ("expanded_hybrid", "row 4"),
         ("row7_rerank", "row 7"), ("row8_mmr", "row 8"), ("row9_crag", "row 9")]
# (component, from row, to row): the delta that the D84 keep rule looks at
DECISIONS = [
    ("row 3: contextual notes", "hybrid_plain", "hybrid_contextual"),
    ("row 4: query expansion", "hybrid_contextual", "expanded_hybrid"),
    ("row 5: metadata filters", "expanded_hybrid", "row5_filters"),
    ("row 5 on the row-3 base (diagnostic)", "hybrid_contextual", "hybrid_filters"),
    ("row 7: cross-encoder rerank", "expanded_hybrid", "row7_rerank"),
    ("row 7 on the row-3 base (diagnostic)", "hybrid_contextual", "hybrid_rerank"),
    ("row 8: MMR at the frozen lambda", "row7_rerank", "row8_mmr"),
    ("row 9: corrective pass at the frozen tau", "row8_mmr", "row9_crag"),
]


def verdict(delta: float | None) -> str:
    """The D84 keep rule."""
    if delta is None:
        return "n/a"
    return "KEEP" if delta >= -KEEP_TOLERANCE - 1e-12 else "DROP"


def memo(fn):
    """Cache a retriever's results per (query, k): the same rankings are scored twice."""
    cache: dict = {}

    def wrapped(query, filters=None, k=10):
        if (query, k) not in cache:
            cache[(query, k)] = fn(query, filters, k)
        return cache[(query, k)]

    return wrapped


def ndcg(res: dict) -> float | None:
    return res["groups"]["overall"]["ndcg@10"]


def f3(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    queries = load_queries("dev")
    q1, qm = load_qrels("dev"), load_qrels_merged("dev")
    info, excluded, ref = load_chunk_info(ENRICHED, PROVENANCE), load_excluded(), \
        load_reference_date()
    spec = []
    extra = [["hybrid_filters", "eval.variants:hybrid_filters"]]  # in no rows file (a diagnostic)
    for rows in (json.loads((REPO_ROOT / "eval" / f).read_text(encoding="utf-8"))
                 for f in ("rows_dev.json", "rows_dev_diagnostics.json")):
        for name, fn in rows + extra:
            if name not in {n for n, _ in spec}:
                spec.append((name, fn))
    r1: dict[str, dict] = {}
    rm: dict[str, dict] = {}
    for name, fn in spec:
        retr = memo(load_retriever(fn))
        r1[name] = evaluate(retr, queries, q1, info, ref, excluded)
        rm[name] = evaluate(retr, queries, qm, info, ref, excluded)
        print(f"scored {name}: round 1 {f3(ndcg(r1[name]))}  merged {f3(ndcg(rm[name]))}",
              flush=True)

    lines = ["# Dev rows re-scored with the merged judgments (round 1 human + round 2 chat)", "",
             "Round 2 is labelled by a chat assistant (D95); 4 of its labels were corrected on the "
             "owner's instruction. Dev, 13 queries: descriptive, not significant (D92).", "",
             "## All rows", "",
             "| row | nDCG@10 round 1 | nDCG@10 merged | cond. merged | MRR | P@5 | R@10 | "
             "unjudged@10 r1 | unjudged@10 merged | places@10 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in spec:
        g1, gm = r1[name]["groups"]["overall"], rm[name]["groups"]["overall"]
        lines.append(f"| {name} | {f3(g1['ndcg@10'])} | {f3(gm['ndcg@10'])} | "
                     f"{f3(A.condensed_ndcg(rm[name], qm))} | {f3(gm['mrr'])} | {f3(gm['p@5'])} | "
                     f"{f3(gm['recall@10'])} | {f3(g1['unjudged@10'])} | "
                     f"{f3(gm['unjudged@10'])} | {f3(gm['places@10'])} |")

    # ---- lambda and tau curves on the merged judgments (report only)
    lam0 = V.chosen_lambda()
    tau0 = V.chosen_tau()
    scorer = V.get_ce_scorer()
    curve_l = {}
    for lam in mmr.LAMBDAS:
        curve_l[lam] = ndcg(evaluate(memo(V.make_mmr_variant(lam, scorer)), queries, qm, info,
                                     ref, excluded))
    lam_pick = mmr.select_lambda(curve_l)
    cands = json.loads(V.TAU_FILE.read_text(encoding="utf-8"))["candidates"]
    rewrites = V.grader_mod.load_rewrites(V.REWRITES_FILE)
    curve_t = {}
    for tau in cands:
        def crag(query, filters=None, k=10, _tau=tau):
            return V.run_corrective(query, _tau, scorer, lam0, k, rewrites=rewrites)[0]

        curve_t[tau] = ndcg(evaluate(memo(crag), queries, qm, info, ref, excluded))
    tau_pick = choose_highest(curve_t)
    w_arms = {0.0: "row5_filters", 0.1: "row6_fresh_w10", 0.2: "row6_fresh_w20",
              0.3: "row6_fresh_w30"}
    curve_w = {w: ndcg(rm[n]) for w, n in w_arms.items() if n in rm}
    w_pick = choose_highest(curve_w) if curve_w else None

    lines += ["", "## Keep rules re-applied (D84: delta >= -0.005 over the predecessor)", "",
              "| component | from -> to | delta round 1 | delta merged | verdict (merged) |",
              "|---|---|---|---|---|"]
    verdicts = {}
    for comp, a, b in DECISIONS:
        d1, dm = delta(ndcg(r1[a]), ndcg(r1[b])), delta(ndcg(rm[a]), ndcg(rm[b]))
        v = verdict(dm)
        if comp.startswith("row 8"):
            v = f"by the lambda rule: lambda {lam_pick}" + (" (MMR off)" if lam_pick == 1.0
                                                           else " (MMR on)")
        verdicts[comp] = {"delta_round1": d1, "delta_merged": dm, "verdict": v}
        lines.append(f"| {comp} | {a} -> {b} | "
                     f"{'-' if d1 is None else f'{d1:+.3f}'} | "
                     f"{'-' if dm is None else f'{dm:+.3f}'} | {v} |")
    lines += ["", "## Parameters: what the pre-stated rules pick on the merged judgments", "",
              f"* freshness weight w (D87, highest dev nDCG@10, ties to the smaller): "
              f"{ {w: round(v, 4) for w, v in curve_w.items()} } -> picks {w_pick} "
              f"(frozen: 0, freshness off)",
              f"* MMR lambda (D89): { {k: round(v, 4) for k, v in curve_l.items()} } -> picks "
              f"{lam_pick} (frozen: {lam0})",
              f"* corrective tau (D90): { {k: round(v, 4) for k, v in curve_t.items()} } -> picks "
              f"{tau_pick} (frozen: {tau0})",
              "", "The frozen files eval/mmr_lambda.json and eval/corrective_tau.json were NOT "
              "changed. A different pick here is information for the owner, not an automatic "
              "change."]
    chain = [(n, rm[n]) for n, _ in CHAIN if n in rm]
    sig = A.significance_rows(chain)
    lines += ["", "## Paired nDCG@10 significance between consecutive chain rows (merged)", "",
              "| comparison | n | delta | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for s in sig:
        lines.append(f"| {s['comparison']} | {s['n']} | {s['delta_ndcg@10']:+.3f} | "
                     f"[{f3(s['ci95'][0])}, {f3(s['ci95'][1])}] | {s['p']:.4f} | "
                     f"{s['p_holm']:.4f} |")
    md = "\n".join(lines)
    print("\n" + md)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_merged.md").write_text(md + "\n", encoding="utf-8")
    (OUT / "dev_merged.json").write_text(json.dumps(
        {"ndcg_round1": {n: ndcg(r1[n]) for n, _ in spec},
         "ndcg_merged": {n: ndcg(rm[n]) for n, _ in spec}, "verdicts": verdicts,
         "lambda_curve_merged": {str(k): v for k, v in curve_l.items()}, "lambda_pick": lam_pick,
         "tau_curve_merged": {str(k): v for k, v in curve_t.items()}, "tau_pick": tau_pick,
         "w_curve_merged": {str(k): v for k, v in curve_w.items()}, "w_pick": w_pick,
         "frozen": {"lambda": lam0, "tau": tau0}}, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
