"""Evidence for rows 5 and 6 on dev: which filters fired, what they rejected, and the effect.

    python scripts/rows56_report.py            # dev only; writes eval/ablation/rows56_dev.md

Per dev query: the filters the parse produced, how many of the 100 candidates each rejected (by
reason and provenance), the number of rule-based rows the place_type check skipped (D69), the
condensed nDCG@10 of row 4 (expanded) and row 5 (filters) and their difference. Then each filter
kind on its own (state only, season only, place_type only) so the loss can be attributed, and the
diagnostics on the plain hybrid base and 5b. Dev only; reads qrels_dev. No network.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M  # noqa: E402
from eval import variants as V  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402
from src.retrieval.filters import Filters  # noqa: E402
from src.retrieval.postprocess import PostConfig  # noqa: E402

OUT = REPO_ROOT / "eval" / "ablation" / "rows56_dev.md"
K = 20


def cond(ranked: list[str], qrel: dict[str, int]) -> float | None:
    return M.ndcg_at_k([c for c in ranked if c in qrel], qrel, 10)


def run(query: str, config: PostConfig, base: str, explicit: dict | None = None):
    rows, trace = V.run_post(query, config, K, base=base, explicit_filters=explicit)
    return [r["chunk_id"] for r in rows], trace


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    queries, qrels = load_queries("dev"), load_qrels("dev")
    parsed = V.parsed_by_text()
    lines = ["# Rows 5 and 6 on dev: which filters fired and what they cost", "",
             "condensed nDCG@10 (unjudged removed before the cut) of row 4 (expanded) and row 5 "
             "(+ filters), per dev query.", "",
             "| query | filters from the parse | rejected of 100 (state/season/place_type) | "
             "exempt | row4 | row5 | delta |", "|---|---|---|---|---|---|---|"]
    r4s, r5s = [], []
    for q in queries:
        qr = qrels[q["id"]]
        base_ids, _ = run(q["text"], PostConfig(), "expanded")
        ids, trace = run(q["text"], PostConfig(use_filters=True), "expanded")
        a, b = cond(base_ids, qr), cond(ids, qr)
        r4s.append(a)
        r5s.append(b)
        f, rej = trace["filters"], trace["filters"]["rejected"]
        shown = {k: v for k, v in f["filters"].items() if v}
        lines.append(f"| {q['id']} | {shown or 'none'} | "
                     f"{rej.get('state', 0)}/{rej.get('season', 0)}/{rej.get('place_type', 0)} | "
                     f"{f['exempt']} | {fmt(a)} | {fmt(b)} | "
                     f"{'' if a is None or b is None else f'{b - a:+.3f}'} |")
    lines += ["", f"mean condensed nDCG@10: row 4 {fmt(mean(r4s))}, row 5 {fmt(mean(r5s))}", "",
              "## Each filter kind alone (explicit filters taken from the parse), on row 4", "",
              "| arm | cond nDCG@10 | queries where it fired |", "|---|---|---|"]
    arms = {"state only": ("state",), "season/month only": ("season", "month"),
            "place_type only": ("place_type",)}
    for name, keys in arms.items():
        vals, fired = [], 0
        for q in queries:
            f = Filters.from_parsed(parsed[q["text"]])
            explicit = {"state": f.state, "season": f.season, "month": f.month,
                        "place_type": f.place_type}
            explicit = {k: v for k, v in explicit.items() if k in keys and v}
            fired += bool(explicit)
            ids, _ = run(q["text"], PostConfig(), "expanded", explicit or None)
            vals.append(cond(ids, qrels[q["id"]]))
        lines.append(f"| {name} | {fmt(mean(vals))} | {fired} of {len(queries)} |")
    lines += ["", "## Diagnostics (not frozen variants)", "",
              "| arm | cond nDCG@10 | note |", "|---|---|---|"]
    diag = [("5b: place_type also on rule-based rows", PostConfig(use_filters=True,
            rule_based_place_type=True), "expanded", "uses RULE-BASED place_type (D69)"),
            ("filters on the plain hybrid base", PostConfig(use_filters=True), "hybrid",
             "row 4 is provisional"),
            ("hybrid base without filters", PostConfig(), "hybrid", "= hybrid_contextual")]
    for name, cfg, base, note in diag:
        vals = [cond(run(q["text"], cfg, base)[0], qrels[q["id"]]) for q in queries]
        lines.append(f"| {name} | {fmt(mean(vals))} | {note} |")
    md = "\n".join(lines)
    print(md)
    OUT.write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
