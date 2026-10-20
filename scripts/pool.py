"""Round-1 pooling and judging sheets (D72). Local only: no network, no API call, no qrels.

    python scripts/pool.py build      # run the five systems, write eval/pool.json and the sheets
    python scripts/pool.py report     # print the counts from the committed eval/pool.json
    python scripts/pool.py sheets     # rewrite the sheets from eval/pool.json (same seed and rows)

`build` embeds the 40 queries with the local bge-m3 folder (EMBED_LOCAL_DIR in .env; query vectors
are cached in .cache) and refuses to overwrite an existing eval/pool.json without --force, because
labels refer to it. The sheets (eval/sheets/judge_dev.csv, judge_test.csv) hold chunk text and are
gitignored. Reads data/enriched/ and index/ only; writes nothing there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the local model folder is read, never downloaded
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import pooling  # noqa: E402
from eval.adapters import system_ids  # noqa: E402
from eval.qrels import load_queries  # noqa: E402
from eval.run_eval import ENRICHED, PROVENANCE, load_chunk_info, load_excluded  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402

POOL = REPO_ROOT / "eval" / "pool.json"
SHEETS = REPO_ROOT / "eval" / "sheets"
ATTRIBUTION = REPO_ROOT / "data" / "processed" / "attribution.json"
INDEX_MANIFEST = REPO_ROOT / "index" / "index_manifest.json"
SEED = 42


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in map(json.loads, ENRICHED.read_text(
        encoding="utf-8").splitlines()) if c}


def write_sheets(pool: dict, queries: list[dict]) -> dict[str, int]:
    chunks = load_chunks()
    attribution = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    counts = {}
    for split in ("dev", "test"):
        rows = pooling.sheet_rows(pool, queries, chunks, attribution, split, SEED)
        pooling.write_sheet(SHEETS / f"judge_{split}.csv", rows)
        counts[split] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("build", "report", "sheets"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing eval/pool.json")
    args = ap.parse_args(argv)

    queries = load_queries()
    info = load_chunk_info(ENRICHED, PROVENANCE)
    excluded = load_excluded()
    if args.action == "build":
        if POOL.exists() and not args.force:
            print(f"{POOL} exists; labels refer to it. Use --force only to start over.")
            return 2
        result = pooling.build_pool(queries, system_ids, info, excluded)
        pool = {"meta": {"round": pooling.ROUND, "depth": pooling.DEPTH,
                         "systems": list(pooling.SYSTEMS), "rrf_k": 60, "rrf_depth": 100,
                         "sheet_seed": SEED, "queries_sha256": sha(REPO_ROOT / "eval" /
                                                                   "queries.json"),
                         "enriched_sha256": sha(ENRICHED),
                         "index_manifest_sha256": sha(INDEX_MANIFEST)},
                **result}
        POOL.write_text(json.dumps(pool, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {POOL.relative_to(REPO_ROOT)}")
    else:
        pool = json.loads(POOL.read_text(encoding="utf-8"))
    if args.action in ("build", "sheets"):
        counts = write_sheets(pool, queries)
        print(f"wrote eval/sheets/judge_dev.csv ({counts['dev']} rows) and judge_test.csv "
              f"({counts['test']} rows)")
    print(pooling.report(pool, queries, info, excluded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
