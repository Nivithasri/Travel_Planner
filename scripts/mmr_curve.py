"""The MMR lambda curve on dev and the pre-stated selection (D89).

    python scripts/mmr_curve.py --scorer proxy      # WIRING CHECK ONLY: stand-in scores, not row 8
    python scripts/mmr_curve.py --scorer ce         # row 8 proper (needs the reranker, row 7)

For each lambda in {1.0, 0.9, 0.8, 0.7, 0.5}: nDCG@10, condensed nDCG@10 (unjudged removed),
src_types@10, places@10 and unjudged@10 on the DEV split, then the selected lambda by the rule
fixed in D89 (the smallest lambda within 0.02 of lambda = 1.0, by nDCG@10 as charged). Dev only;
no network. With --scorer proxy the table is headed so and nothing is written to eval/ablation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import variants  # noqa: E402
from eval.ablation import condensed_ndcg  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from eval.run_eval import ENRICHED, PROVENANCE, evaluate, load_chunk_info, load_excluded  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402
from src.retrieval import mmr  # noqa: E402


def scorer_for(name: str):
    if name == "proxy":
        return variants.proxy_scorer
    raise SystemExit("--scorer ce needs the cross-encoder (row 7), which is not built yet")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorer", choices=("proxy", "ce"), required=True)
    args = ap.parse_args(argv)
    scorer = scorer_for(args.scorer)
    queries, qrels = load_queries("dev"), load_qrels("dev")
    info, excluded, ref = load_chunk_info(ENRICHED, PROVENANCE), load_excluded(), \
        load_reference_date()
    head = ("STAND-IN SCORES: a wiring check, NOT row 8" if args.scorer == "proxy"
            else "row 8 (cross-encoder), dev")
    lines = [f"# MMR lambda curve ({head})", "",
             "| lambda | nDCG@10 | cond nDCG@10 | src_types@10 | places@10 | unjudged@10 |",
             "|---|---|---|---|---|---|"]
    curve = {}
    for lam in mmr.LAMBDAS:
        res = evaluate(variants.make_mmr_variant(lam, scorer), queries, qrels, info, ref, excluded)
        g = res["groups"]["overall"]
        curve[lam] = g["ndcg@10"]
        cond = condensed_ndcg(res, qrels)
        lines.append(f"| {lam} | {g['ndcg@10']:.3f} | {cond:.3f} | {g['src_types@10']:.3f} | "
                     f"{g['places@10']:.3f} | {g['unjudged@10']:.3f} |")
    chosen = mmr.select_lambda(curve)
    lines += ["", f"selected lambda by the D89 rule: {chosen} (smallest within {mmr.TOLERANCE} "
              "of lambda = 1.0)"]
    out = "\n".join(lines)
    print(out)
    if args.scorer == "ce":
        (REPO_ROOT / "eval" / "ablation" / "mmr_curve_dev.md").write_text(out + "\n",
                                                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
