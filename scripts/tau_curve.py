"""Choose the corrective-retrieval threshold tau on DEV by the rule fixed in LOG.md (D90).

    python scripts/tau_curve.py        # dev only; offline; needs eval/corrective_rewrites.json

The candidates are the 20th, 40th and 60th percentile of the dev best-reranker-scores. For each,
row 9 is run on dev (the stored rewrites; reranker scores come from the cache or the local model)
and compared with row 8 (no correction). RULE (fixed before any row-9 metric existed): tau = the
candidate with the highest dev nDCG@10 as charged (unjudged = 0), ties to the smaller tau. The
D84 keep rule then decides whether row 9 is in the pipeline: delta against row 8 >= -0.005,
re-decided on the round-2 qrels (D91). Writes eval/corrective_tau.json and
eval/ablation/tau_curve_dev.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import variants as V  # noqa: E402
from eval.ablation import condensed_ndcg  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from eval.run_eval import ENRICHED, PROVENANCE, evaluate, load_chunk_info, load_excluded  # noqa: E402
from scripts.corrective_rewrites import first_pass, tau_candidates  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402
from src.retrieval import grader  # noqa: E402


def choose_tau(ndcg_by_tau: dict[float, float | None]) -> float:
    """Highest nDCG@10, ties to the smaller tau (the rule in LOG.md)."""
    best = max(v for v in ndcg_by_tau.values() if v is not None)
    return min(t for t, v in ndcg_by_tau.items() if v is not None and v >= best - 1e-12)


def main() -> int:
    queries, qrels = load_queries("dev"), load_qrels("dev")
    info, excluded, ref = load_chunk_info(ENRICHED, PROVENANCE), load_excluded(), \
        load_reference_date()
    lam, scorer = V.chosen_lambda(), V.get_ce_scorer()
    signals = first_pass(queries, scorer, lam)
    taus = tau_candidates([s["confidence"] for s in signals.values()])
    rewrites = grader.load_rewrites(V.REWRITES_FILE)

    base = evaluate(V.make_mmr_variant(lam, scorer), queries, qrels, info, ref, excluded)
    base_by_q = {r["id"]: r for r in base["per_query"]}
    rows = [("row 8 (no correction)", None, base)]
    curve: dict[float, float | None] = {}
    traces: dict[float, dict[str, dict]] = {}
    for tau in taus:
        seen: dict[str, dict] = {}

        def retrieve(query, filters=None, k=10, _tau=tau, _seen=seen):
            rows_, trace = V.run_corrective(query, _tau, scorer, lam, k, rewrites=rewrites)
            _seen[query] = trace
            return rows_

        res = evaluate(retrieve, queries, qrels, info, ref, excluded)
        curve[tau] = res["groups"]["overall"]["ndcg@10"]
        traces[tau] = seen
        rows.append((f"row 9, tau {tau}", tau, res))
    chosen = choose_tau(curve)

    lines = ["# Row 9: tau on dev (rule in LOG.md; row 8 = no correction)", "",
             "| arm | triggered | nDCG@10 | cond nDCG@10 | MRR | P@5 | unjudged@10 | places@10 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, tau, res in rows:
        g = res["groups"]["overall"]
        trig = "-" if tau is None else f"{sum(1 for t in traces[tau].values() if t['passes'])}/13"
        lines.append(f"| {name} | {trig} | {g['ndcg@10']:.3f} | "
                     f"{condensed_ndcg(res, qrels):.3f} | {g['mrr']:.3f} | {g['p@5']:.3f} | "
                     f"{g['unjudged@10']:.3f} | {g['places@10']:.3f} |")
    lines += ["", f"chosen tau by the rule (highest dev nDCG@10, ties to the smaller): {chosen}",
              "", f"## Per query at the chosen tau {chosen} (triggered queries only)", "",
              "| query | triggers | rewrite | nDCG@10 before | after |", "|---|---|---|---|---|"]
    chosen_res = next(res for _, tau, res in rows if tau == chosen)
    after = {r["id"]: r for r in chosen_res["per_query"]}
    for q in queries:
        t = traces[chosen][q["text"]]
        if t["passes"]:
            lines.append(f"| {q['id']} | {'+'.join(t['triggers'])} | {t['rewrite'] or '(none)'} | "
                         f"{base_by_q[q['id']]['ndcg@10']:.3f} | {after[q['id']]['ndcg@10']:.3f} |")
    delta = curve[chosen] - base["groups"]["overall"]["ndcg@10"]
    lines += ["", f"row 9 minus row 8 at tau {chosen}: {delta:+.3f} nDCG@10 (D84 keep rule: >= "
              "-0.005)"]
    out = "\n".join(lines)
    print(out)
    (REPO_ROOT / "eval" / "ablation" / "tau_curve_dev.md").write_text(out + "\n", encoding="utf-8")
    V.TAU_FILE.write_text(json.dumps(
        {"tau": chosen, "split": "dev", "candidates": taus,
         "rule": "highest dev nDCG@10 as charged, ties to the smaller tau (LOG.md, D90)",
         "curve": {str(k): v for k, v in curve.items()},
         "row8_ndcg": base["groups"]["overall"]["ndcg@10"]}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
