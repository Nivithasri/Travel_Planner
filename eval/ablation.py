"""Run the Phase 6 ablation rows in order on one split and build the comparison table (D84, D92).

    python eval/ablation.py --split dev --rows eval/rows_dev.json
    python eval/ablation.py --split dev --rows eval/rows_dev.json --json eval/ablation

`--rows` is a JSON list of [row_name, "module:function"] pairs in ablation order; each function is a
retrieve-contract callable. Every row is scored with eval.run_eval.evaluate on the split, the
headline metrics go into one table, and a paired sign-flip test (eval.significance, D92) compares
each row with the one before it on per-query nDCG@10, with Holm-Bonferroni over that family.

Pure helpers (headline_row, significance_rows, to_markdown, to_csv) are unit-tested; running the
rows needs the index. Nothing here reads qrels_test unless --split test. Dev tuning only (CLAUDE.md
rule 3); the test split is run once per frozen row at the end of the phase.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import significance as S  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from eval.run_eval import (  # noqa: E402
    ENRICHED,
    PROVENANCE,
    load_chunk_info,
    load_excluded,
    load_retriever,
)
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402

HEADLINE = ("ndcg@10", "mrr", "p@5", "recall@10", "unjudged@10", "excluded@10",
            "src_types@10", "places@10")


def ndcg_by_query(result: dict) -> dict[str, float | None]:
    """{query_id: nDCG@10 or None} from an evaluate() result, for the paired tests."""
    return {r["id"]: r["ndcg@10"] for r in result["per_query"]}


def headline_row(name: str, result: dict) -> dict:
    g = result["groups"]["overall"]
    row = {"row": name, "n": g["n_queries"], "0rel": g["n_zero_relevant"]}
    row.update({c: g[c] for c in HEADLINE})
    return row


def significance_rows(ordered: list[tuple[str, dict]], *, n_mc: int = S.DEFAULT_MC,
                      seed: int = S.SEED) -> list[dict]:
    """One paired comparison per consecutive pair of rows, with Holm-adjusted p over the family."""
    comps = []
    for (name_a, res_a), (name_b, res_b) in zip(ordered, ordered[1:]):
        comps.append(S.compare(f"{name_a} -> {name_b}", ndcg_by_query(res_a),
                               ndcg_by_query(res_b), n_mc=n_mc, seed=seed))
    out = []
    for comp, holm_p in S.holm_over(comps):
        out.append({"comparison": comp.name, "n": comp.n, "delta_ndcg@10": comp.mean_diff,
                    "ci95": [comp.ci_low, comp.ci_high], "p": comp.p, "p_holm": holm_p,
                    "method": comp.method})
    return out


def _fmt(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def to_markdown(rows: list[dict], sig: list[dict], split: str) -> str:
    cols = ["row", "n", "0rel", *HEADLINE]
    lines = [f"# Phase 6 ablation ({split}; dev tuning only, D84/D92)", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            str(r[c]) if c in ("row", "n", "0rel") else _fmt(r[c]) for c in cols) + " |")
    lines += ["", "## Paired nDCG@10 significance between consecutive rows (D92)", "",
              "| comparison | n | delta | 95% CI | p | p (Holm) | method |",
              "|---|---|---|---|---|---|---|"]
    for s in sig:
        ci = f"[{_fmt(s['ci95'][0])}, {_fmt(s['ci95'][1])}]"
        lines.append(f"| {s['comparison']} | {s['n']} | {s['delta_ndcg@10']:+.3f} | {ci} | "
                     f"{s['p']:.4f} | {s['p_holm']:.4f} | {s['method']} |")
    lines += ["", "unjudged@10 > 0 means the row retrieved chunks round 1 never pooled; those "
              "count as 0 until round 2 is labelled (D91). nDCG on 13 dev queries is descriptive, "
              "not significant (D92)."]
    return "\n".join(lines)


def to_csv(rows: list[dict]) -> str:
    cols = ["row", "n", "0rel", *HEADLINE]
    out = [",".join(cols)]
    for r in rows:
        out.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
    return "\n".join(out) + "\n"


def run(rows_spec: list[list[str]], split: str, out_dir: Path | None) -> int:
    reference = load_reference_date()
    if reference is None:
        print("EVAL_REFERENCE_DATE is not set in .env (D4)")
        return 2
    queries = load_queries(split)
    qrels = load_qrels(split, allow_test=(split == "test"))
    info = load_chunk_info(ENRICHED, PROVENANCE)
    excluded = load_excluded()
    from eval.run_eval import evaluate

    ordered: list[tuple[str, dict]] = []
    for name, spec in rows_spec:
        result = evaluate(load_retriever(spec), queries, qrels, info, reference, excluded)
        ordered.append((name, result))
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{split}_{name}.json").write_text(
                json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print(f"ran {name}: nDCG@10 {_fmt(result['groups']['overall']['ndcg@10'])}")
    table = [headline_row(name, res) for name, res in ordered]
    sig = significance_rows(ordered) if len(ordered) > 1 else []
    md = to_markdown(table, sig, split)
    print("\n" + md)
    if out_dir:
        (out_dir / f"ablation_{split}.md").write_text(md + "\n", encoding="utf-8")
        (out_dir / f"ablation_{split}.csv").write_text(to_csv(table), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--rows", type=Path, required=True, help="JSON [[name, module:function], ...]")
    ap.add_argument("--json", type=Path, help="directory for per-row JSON and the table")
    args = ap.parse_args(argv)
    rows_spec = json.loads(args.rows.read_text(encoding="utf-8"))
    out_dir = args.json or (REPO_ROOT / "eval" / "ablation")
    return run(rows_spec, args.split, out_dir)


if __name__ == "__main__":
    sys.exit(main())
