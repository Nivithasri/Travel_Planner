"""Phase 6 Step 0: do corpus vectors embedded in THIS environment match the committed index?

    python scripts/embed_parity.py            # 50 seeded chunks per dense kind (plain, contextual)
    python scripts/embed_parity.py --n 20

Re-embeds a seeded sample of enriched chunks with the local bge-m3 folder (EMBED_LOCAL_DIR) and
compares each vector with its row in index/dense_<kind>.npy (cosine; both are unit-norm). Also
checks that the local model files hash to the ones the index manifest recorded. Reads data/enriched
and index/ only; writes nothing, no cache, no network (HF_HUB_OFFLINE).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the local model folder is read, never downloaded
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common.schemas import REPO_ROOT, load_embed_local_dir  # noqa: E402
from src.retrieval.build_index import (  # noqa: E402
    DENSE_KINDS,
    IDS_FILE,
    MANIFEST_FILE,
    dense_text,
    library_versions,
    load_chunks,
    load_encoder,
    model_file_hashes,
)

INDEX_DIR = REPO_ROOT / "index"
ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
SEED = 42
PASS_MIN_COSINE = 0.999  # the Step 0 gate (plan); fixed before the run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50, help="chunks per dense kind")
    args = ap.parse_args(argv)

    model_dir = load_embed_local_dir()
    if model_dir is None:
        print("EMBED_LOCAL_DIR is not set in .env")
        return 2
    manifest = json.loads((INDEX_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
    recorded = manifest.get("model_files_sha256") or {}
    local = model_file_hashes(Path(model_dir))
    same = {name: local.get(name) == digest for name, digest in recorded.items()}
    print(f"model files: {sum(same.values())} of {len(same)} match the index manifest"
          + ("" if all(same.values()) else f"; differ: {[n for n, ok in same.items() if not ok]}"))
    print(f"versions now:      {library_versions()}")
    print(f"versions at build: {manifest.get('versions')}")

    chunks = load_chunks(ENRICHED)
    ids = json.loads((INDEX_DIR / IDS_FILE).read_text(encoding="utf-8"))
    row = {cid: i for i, cid in enumerate(ids)}
    sample = random.Random(SEED).sample(range(len(chunks)), args.n)
    t0 = time.monotonic()
    encode = load_encoder(Path(model_dir))
    print(f"encoder loaded in {time.monotonic() - t0:.1f} s; {args.n} chunks, seed {SEED}")

    worst = 1.0
    for kind in DENSE_KINDS:
        matrix = np.load(INDEX_DIR / f"dense_{kind}.npy", mmap_mode="r")
        picked = [chunks[i] for i in sample]
        t1 = time.monotonic()
        fresh = encode([dense_text(c, kind) for c in picked])
        stored = np.stack([matrix[row[c["chunk_id"]]] for c in picked])
        cos = (fresh * stored).sum(axis=1)
        worst = min(worst, float(cos.min()))
        print(f"dense_{kind}: cosine min {cos.min():.6f}  mean {cos.mean():.6f}  "
              f"max {cos.max():.6f}  max |diff| {np.abs(fresh - stored).max():.2e}  "
              f"({time.monotonic() - t1:.1f} s)")
    verdict = "PASS" if worst >= PASS_MIN_COSINE and all(same.values()) else "FAIL"
    print(f"{verdict}: min cosine {worst:.6f} (gate >= {PASS_MIN_COSINE}), model files "
          f"{'identical' if all(same.values()) else 'DIFFERENT'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
