"""Query the Phase 3 indexes: top-k from BM25 (contextual, plain) and dense (plain, contextual).

    python scripts/query_index.py "quiet beaches in Kerala in monsoon"
    python scripts/query_index.py "Hampi ruins timings" --k 10 [--index-dir DIR --enriched FILE]

Frozen-corpus only; no API call. Dense search embeds the query on CPU with the local bge-m3 folder
(EMBED_LOCAL_DIR in .env or --model-dir) and caches the query vector on disk. Columns: rank,
chunk_id, place, source_type, score, snippet. Ties: score descending, then chunk_id ascending.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.cache import DiskCache  # noqa: E402
from src.common.schemas import REPO_ROOT, load_embed_local_dir  # noqa: E402
from src.retrieval.build_index import load_encoder  # noqa: E402
from src.retrieval.search import TABLES, Index, model_id_for  # noqa: E402


def format_table(name: str, rows: list[dict]) -> str:
    lines = [f"== {name}"]
    if not rows:
        return "\n".join(lines + ["   (no results)"])
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:>2}  {r['chunk_id']}  {r['place']:<14.14} {r['source_type']:<10} "
                     f"{r['score']:>8.4f}  {r['snippet']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--index-dir", type=Path, default=REPO_ROOT / "index")
    ap.add_argument("--enriched", type=Path,
                    default=REPO_ROOT / "data" / "enriched" / "chunks.jsonl")
    ap.add_argument("--model-dir", type=Path)
    args = ap.parse_args(argv)
    if not (args.index_dir / "index_manifest.json").exists():
        print(f"no index in {args.index_dir}: build it with `python scripts/build_index.py` "
              "after the full enrichment run")
        return 2
    d = args.model_dir or load_embed_local_dir()
    if d is None:
        print("EMBED_LOCAL_DIR is not set in .env and --model-dir was not given")
        return 2
    index = Index(args.index_dir, args.enriched, encode=load_encoder(Path(d)),
                  cache=DiskCache(REPO_ROOT / ".cache"), model_id=model_id_for(Path(d)))
    print(f"query: {args.query!r}  (top {args.k})")
    for name, _, _ in TABLES:
        print(format_table(name, index.table(name, args.query, args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
