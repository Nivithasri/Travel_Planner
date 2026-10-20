"""Evaluate any retrieve-contract function on the Phase 4 test collection (D70 to D73).

    python eval/run_eval.py --retriever eval.adapters:dense_plain --split dev
    python eval/run_eval.py --retriever my_module:retrieve --split test      (final runs only)

The retriever is `retrieve(query, filters=None, k=10) -> list[dict]` (chunk fields + "score"). Every
query is run with k=20 and no filters. Chunk fields for the metrics (place, source type, date kind,
provenance) come from the enriched corpus, not from the retriever's dicts. Output: overall, per
query type, the freshness slice, place-named versus place-free queries (D69), and a provenance
table.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402

RUN_K = 20
COLUMNS = ("ndcg@10", "mrr", "p@5", "recall@10", "recall@20", "unjudged@10", "unjudged@20",
           "fresh@10", "src_types@10", "places@10", "excluded@10")
ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
PROVENANCE = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"
EXCLUDED = REPO_ROOT / "data" / "manifests" / "d33_excluded.json"

Retriever = Callable[..., list[dict]]


def load_retriever(spec: str) -> Retriever:
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError("--retriever must look like module:function")
    return getattr(importlib.import_module(module), name)


def load_chunk_info(enriched: Path, provenance: Path) -> dict[str, dict]:
    prov = {}
    for line in provenance.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prov[r["chunk_id"]] = r["source"]  # llm | fallback | rule_only
    info = {}
    for line in enriched.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            info[c["chunk_id"]] = {"chunk_id": c["chunk_id"], "place": c["place"],
                                   "source_type": c["source_type"], "date": c.get("date"),
                                   "date_kind": c.get("date_kind"),
                                   "provenance": prov.get(c["chunk_id"], "unknown")}
    return info


def load_excluded(path: Path = EXCLUDED) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8"))["excluded"])


def run_query(retrieve: Retriever, query: dict, qrel: dict[str, int], info: dict[str, dict],
              reference: date, excluded: set[str], threshold: int, k: int = RUN_K) -> dict:
    results = retrieve(query["text"], None, k)
    ranked = M.dedupe(r["chunk_id"] for r in results)[:k]
    unknown = [c for c in ranked if c not in info]
    if unknown:
        raise ValueError(f"retriever returned chunk ids outside the corpus: {unknown[:3]}")
    rows = [info[c] for c in ranked]
    top10 = rows[:10]
    div = M.diversity(top10)
    return {
        "id": query["id"], "type": query["type"], "freshness_slice": query["freshness_slice"],
        "place_scope": query["place_scope"], "ranked": ranked,
        "ndcg@10": M.ndcg_at_k(ranked, qrel, 10, threshold),
        "mrr": M.reciprocal_rank(ranked, qrel, None, threshold),
        "p@5": M.precision_at_k(ranked, qrel, 5, threshold),
        "recall@10": M.recall_at_k(ranked, qrel, 10, threshold),
        "recall@20": M.recall_at_k(ranked, qrel, 20, threshold),
        "unjudged@10": M.unjudged_at_k(ranked, qrel, 10),
        "unjudged@20": M.unjudged_at_k(ranked, qrel, 20),
        "fresh@10": M.freshness_share(top10, reference),
        "src_types@10": float(div["source_types"]), "places@10": float(div["places"]),
        "excluded@10": M.excluded_share(top10, excluded),
        "has_relevant": M.has_relevant(qrel, threshold),
        "provenance": M.provenance_counts(top10, qrel, threshold),
    }


def summarise(per_query: list[dict]) -> dict:
    out: dict = {"n_queries": len(per_query),
                 "n_zero_relevant": sum(1 for r in per_query if not r["has_relevant"])}
    for col in COLUMNS:
        mean, skipped = M.mean(r[col] for r in per_query)
        out[col] = mean
        if col in ("ndcg@10", "mrr", "recall@10", "recall@20"):
            out[f"{col}_excluded"] = skipped
    prov: dict[str, Counter] = {}
    for r in per_query:
        for cls, c in r["provenance"].items():
            prov.setdefault(cls, Counter()).update(c)
    out["provenance"] = {k: dict(v) for k, v in sorted(prov.items())}
    return out


def evaluate(retrieve: Retriever, queries: list[dict], qrels: dict[str, dict[str, int]],
             info: dict[str, dict], reference: date, excluded: set[str],
             threshold: int = M.DEFAULT_THRESHOLD) -> dict:
    per_query = [run_query(retrieve, q, qrels.get(q["id"], {}), info, reference, excluded,
                           threshold) for q in queries]
    groups = {"overall": per_query}
    for t in sorted({q["type"] for q in queries}):
        groups[f"type={t}"] = [r for r in per_query if r["type"] == t]
    groups["freshness_slice"] = [r for r in per_query if r["freshness_slice"]]
    for scope in ("named", "free"):
        groups[f"place_scope={scope}"] = [r for r in per_query if r["place_scope"] == scope]
    return {"groups": {k: summarise(v) for k, v in groups.items() if v},
            "per_query": [{k: v for k, v in r.items() if k != "provenance"} for r in per_query],
            "queries_without_qrels": sorted(q["id"] for q in queries if q["id"] not in qrels)}


def fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def render(result: dict, header: str) -> str:
    lines = [header, ""]
    head = f"{'group':<22}{'n':>3}{'0rel':>5}" + "".join(f"{c:>13}" for c in COLUMNS)
    lines += [head, "-" * len(head)]
    for name, g in result["groups"].items():
        lines.append(f"{name:<22}{g['n_queries']:>3}{g['n_zero_relevant']:>5}"
                     + "".join(f"{fmt(g[c]):>13}" for c in COLUMNS))
    lines += ["", "0rel = queries with no relevant judged chunk (excluded from the nDCG, MRR and "
              "Recall means).",
              "Recall@20 is a LOWER BOUND: ranks 11 to 20 are mostly unjudged (see unjudged@20).",
              "fresh@10 counts published dates in the 18 months before the reference date; the "
              "frozen corpus has none (D32), so 0.000 is expected for every retriever.", ""]
    lines.append("top-10 by provenance (D69), summed over the queries above (overall):")
    lines.append(f"{'class':<12}{'results':>8}{'judged':>8}{'relevant':>9}{'precision':>10}"
                 f"{'mean grade':>11}")
    prov = result["groups"]["overall"]["provenance"]
    total = sum(v.get("n", 0) for v in prov.values()) or 1
    for cls, v in prov.items():
        judged = v.get("judged", 0)
        prec = v.get("relevant", 0) / v["n"] if v["n"] else 0.0
        grade = v.get("grade_sum", 0) / judged if judged else None
        lines.append(f"{cls:<12}{v['n']:>8}{judged:>8}{v.get('relevant', 0):>9}{prec:>10.3f}"
                     f"{fmt(grade):>11}   (share of top-10 slots {v['n'] / total:.3f})")
    if result["queries_without_qrels"]:
        lines += ["", "queries with no qrels entry (all results unjudged): "
                  + ", ".join(result["queries_without_qrels"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retriever", required=True, help="module:function")
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--threshold", type=int, default=M.DEFAULT_THRESHOLD,
                    help="smallest label that counts as relevant (default 1)")
    ap.add_argument("--json", type=Path, help="also write the full result as JSON")
    args = ap.parse_args(argv)

    reference = load_reference_date()
    if reference is None:
        print("EVAL_REFERENCE_DATE is not set in .env (D4)")
        return 2
    queries = load_queries(args.split)
    qrels = load_qrels(args.split, allow_test=(args.split == "test"))
    info = load_chunk_info(ENRICHED, PROVENANCE)
    result = evaluate(load_retriever(args.retriever), queries, qrels, info, reference,
                      load_excluded(), args.threshold)
    print(render(result, f"retriever {args.retriever}  split {args.split}  "
                         f"reference date {reference.isoformat()}  relevant = label >= "
                         f"{args.threshold}"))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
