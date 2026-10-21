"""Round-2 pooling and blind judging sheets for the Phase 6 variants (D91). No network, no API call.

    python scripts/pool2.py build     # run every variant on all 40 queries -> eval/pool2.json
    python scripts/pool2.py build --warm   # first: embed all queries without the reranker (cache)
    python scripts/pool2.py report    # counts and workload from the committed eval/pool2.json
    python scripts/pool2.py sheets    # blind sheets, NEW pairs only (eval/sheets/judge2_*.csv)

The new pairs are the top 10 of each Phase 6 variant minus the 963 pairs already in
eval/pool.json (round 1). The judged set is read from that pool file, so this script never
opens a relevance judgments file of any split. `build` refuses to overwrite an existing
eval/pool2.json without --force because labels will refer to it. The sheets hold chunk text and
are gitignored (O12). The test sheet is written only when no variant has a gap on the test
queries (a system that could not run, such as row 9 on a query whose rewrite has not been
generated); the dev sheet is always written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import pooling, pooling2  # noqa: E402
from eval import variants as V  # noqa: E402
from eval.qrels import load_queries  # noqa: E402
from eval.run_eval import ENRICHED, PROVENANCE, load_chunk_info, load_excluded  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402

POOL1 = REPO_ROOT / "eval" / "pool.json"
POOL2 = REPO_ROOT / "eval" / "pool2.json"
SHEETS = REPO_ROOT / "eval" / "sheets"
ATTRIBUTION = REPO_ROOT / "data" / "processed" / "attribution.json"
SEED = 43
MISSING_REWRITE = "corrective_rewrites"
# Systems that load the cross-encoder. bge-m3 (to embed new query text) and the cross-encoder do not
# fit in memory together on a 7 GB machine, so `build --warm` first embeds every query and
# sub-query without them; the real build then finds every query vector in the disk cache.
NEEDS_RERANKER = {"row7_rerank", "row8_mmr", "row9_crag", "hybrid_rerank"}

# The Phase 6 variants pooled in round 2 (D91): rows 2-9, the place_type-on-every-row diagnostic
# (5b), the D69 control arms, and the diagnostics reported in the LOG. The baselines are in round 1.
SYSTEMS = [
    ("hybrid_plain", V.hybrid_plain), ("hybrid_contextual", V.hybrid_contextual),
    ("dense_control", V.dense_control), ("bm25_control", V.bm25_control),
    ("hybrid_control", V.hybrid_control), ("expanded_hybrid", V.expanded_hybrid),
    ("row5_filters", V.row5_filters), ("row5b_filters_all_rows", V.row5b_filters_all_rows),
    ("row6_fresh_w10", V.row6_fresh_w10), ("row6_fresh_w20", V.row6_fresh_w20),
    ("row6_fresh_w30", V.row6_fresh_w30), ("row7_rerank", V.row7_rerank),
    ("row8_mmr", V.row8_mmr), ("row9_crag", V.row9_crag),
    ("hybrid_rerank", V.hybrid_rerank), ("hybrid_filters", V.hybrid_filters),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_system(fn, text: str) -> list[str] | None:
    """The system's top-10 chunk ids, or None when it cannot run for this query because a stored
    corrective rewrite is missing (reported as a gap, never filled)."""
    try:
        ids = [r["chunk_id"] for r in fn(text, None, pooling2.DEPTH)]
    except KeyError as exc:
        if MISSING_REWRITE in str(exc):
            return None
        raise
    return ids[:pooling2.DEPTH]


def warm(queries: list[dict]) -> None:
    """Run every system that does not need the reranker, so all query and sub-query vectors are in
    the disk cache. Writes nothing else."""
    for q in queries:
        for name, fn in SYSTEMS:
            if name not in NEEDS_RERANKER:
                run_system(fn, q["text"])
        print(f"{q['id']} warmed", flush=True)


def build(queries: list[dict]) -> dict:
    info = load_chunk_info(ENRICHED, PROVENANCE)
    pool1 = json.loads(POOL1.read_text(encoding="utf-8"))
    tops: dict[str, dict[str, list[str] | None]] = {}
    for q in queries:
        tops[q["id"]] = {}
        for name, fn in SYSTEMS:
            tops[q["id"]][name] = run_system(fn, q["text"])
        print(f"{q['id']} ({q['split']}) done", flush=True)
    pool2 = pooling2.build_pool2(queries, tops, pooling2.judged_pairs(pool1), info, load_excluded())
    inputs = {"queries": REPO_ROOT / "eval" / "queries.json", "pool1": POOL1,
              "enriched": ENRICHED, "parsed_queries": V.PARSED_QUERIES,
              "rewrites": V.REWRITES_FILE, "mmr_lambda": V.MMR_LAMBDA_FILE,
              "corrective_tau": V.TAU_FILE,
              "index_manifest": REPO_ROOT / "index" / "index_manifest.json",
              "rerank_manifest": REPO_ROOT / "data" / "manifests" / "rerank_manifest.json"}
    meta = {"round": pooling2.ROUND, "depth": pooling2.DEPTH, "sheet_seed": SEED,
            "systems": [n for n, _ in SYSTEMS], "inputs_sha256": {k: sha(p) for k, p in
                                                                   inputs.items()}}
    return {"meta": meta, **pool2}


def write_sheets(pool2: dict, queries: list[dict], only: str | None = None
                 ) -> dict[str, int | str]:
    """Write the blind sheets. `only` limits it to one split, so a sheet that is already being
    labelled is never overwritten."""
    chunks = {c["chunk_id"]: c for c in map(json.loads, ENRICHED.read_text(
        encoding="utf-8").splitlines()) if c}
    attribution = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    split_of = {q["id"]: q["split"] for q in queries}
    test_incomplete = any(split_of[qid] == "test" for qids in pool2["gaps"].values()
                          for qid in qids)
    out: dict[str, int | str] = {}
    for split in ("dev", "test"):
        if only and split != only:
            out[split] = "left as it is (--only)"
            continue
        if split == "test" and test_incomplete:
            out[split] = "NOT WRITTEN: a variant could not run on some test queries (see report)"
            continue
        rows = pooling.sheet_rows(pool2, queries, chunks, attribution, split, SEED)
        pooling.write_sheet(SHEETS / f"judge2_{split}.csv", rows)
        out[split] = len(rows)
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("build", "report", "sheets"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing eval/pool2.json")
    ap.add_argument("--only", choices=("dev", "test"),
                    help="sheets only: write just this split and leave the other sheet untouched")
    ap.add_argument("--warm", action="store_true",
                    help="build only: embed all queries without the reranker, write nothing")
    args = ap.parse_args(argv)

    queries = load_queries()
    if args.action == "build" and args.warm:
        warm(queries)
        print("warm-up done: all query vectors are cached; now run `build` without --warm")
        return 0
    if args.action == "build":
        if POOL2.exists() and not args.force:
            print(f"{POOL2} exists; labels refer to it. Use --force only to start over.")
            return 2
        pool2 = build(queries)
        POOL2.write_text(json.dumps(pool2, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {POOL2.relative_to(REPO_ROOT)}")
    else:
        pool2 = json.loads(POOL2.read_text(encoding="utf-8"))
    if args.action == "sheets":
        counts = write_sheets(pool2, queries, args.only)
        print("sheets:", counts)
    pool1 = json.loads(POOL1.read_text(encoding="utf-8"))
    print(pooling2.report(pool2, queries, pooling2.judged_pairs(pool1),
                          load_chunk_info(ENRICHED, PROVENANCE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
