"""Build the D69 CONTROL index: context_note replaced by "about <place>, <state>" (D69, D84).

    python scripts/build_control_index.py            # -> work/phase6/control/ (gitignored)
    python scripts/build_control_index.py --verify

The control separates the effect of the model-written note from the effect of the place name
alone (owner point 4, D69): its "contextual" text is `about <place>, <state>` + the chunk text,
nothing else. One BM25 index and one dense matrix (the dense build resumes on a rerun). Reads
data/enriched and the local bge-m3 folder (EMBED_LOCAL_DIR); writes only under work/phase6. No
network, no API call. The artifacts are named bm25_contextual/ and dense_contextual.npy so
eval.variants can load them through the same Index class as the real index.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common.cache import atomic_write_text  # noqa: E402
from src.common.schemas import REPO_ROOT, load_embed_local_dir  # noqa: E402
from src.retrieval.build_index import (  # noqa: E402
    DENSE_PARAMS,
    EXPECTED_TOKENIZER_SHA,
    IDS_FILE,
    build_bm25,
    build_dense,
    library_versions,
    load_bm25,
    load_chunks,
    load_encoder,
    model_file_hashes,
    sha256_file,
)

ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
OUT = REPO_ROOT / "work" / "phase6" / "control"
MANIFEST = "control_manifest.json"


def control_note(chunk: dict) -> str:
    return f"about {chunk['place']}, {chunk['state']}"


def control_bm25_text(chunk: dict) -> str:
    return f"{control_note(chunk)}\n{chunk['text']}"


def control_dense_text(chunk: dict) -> str:
    return f"{control_note(chunk)}{DENSE_PARAMS['contextual_separator']}{chunk['text']}"


def model_dir(arg: Path | None) -> Path:
    d = arg or load_embed_local_dir()
    if d is None:
        sys.exit("EMBED_LOCAL_DIR is not set in .env and --model-dir was not given")
    d = Path(d)
    if sha256_file(d / "tokenizer.json") != EXPECTED_TOKENIZER_SHA:
        sys.exit("STOP: the model folder's tokenizer.json is not the frozen-corpus tokenizer")
    return d


def verify(out: Path) -> list[str]:
    problems = []
    chunks = load_chunks(ENRICHED)
    ids = json.loads((out / IDS_FILE).read_text(encoding="utf-8")) if (out / IDS_FILE).exists() \
        else None
    if ids != [c["chunk_id"] for c in chunks]:
        problems.append("ids.json missing or not equal to the enriched chunk ids in order")
    dense = out / "dense_contextual.npy"
    if not dense.exists():
        problems.append("dense_contextual.npy is missing")
    else:
        m = np.load(dense)
        if m.shape != (len(chunks), DENSE_PARAMS["dim"]):
            problems.append(f"dense shape {m.shape}, expected {(len(chunks), DENSE_PARAMS['dim'])}")
        elif not np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-3):
            problems.append("dense rows are not unit-norm")
    try:
        r = load_bm25(out / "bm25_contextual")
        if r.scores["num_docs"] != len(chunks):
            problems.append(f"bm25 holds {r.scores['num_docs']} docs, expected {len(chunks)}")
    except Exception as exc:
        problems.append(f"bm25_contextual cannot be loaded: {type(exc).__name__}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        problems = verify(args.out)
        print("VERIFIED" if not problems else "VERIFY FAILED")
        for p in problems:
            print(" -", p)
        return 1 if problems else 0

    chunks = load_chunks(ENRICHED)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    bm = build_bm25([control_bm25_text(c) for c in chunks], args.out / "bm25_contextual")
    print(f"bm25_contextual (control): {bm}")
    atomic_write_text(args.out / IDS_FILE, json.dumps([c["chunk_id"] for c in chunks]) + "\n")
    d = model_dir(args.model_dir)
    encode = load_encoder(d)
    dense = build_dense([control_dense_text(c) for c in chunks], args.out / "dense_contextual.npy",
                        encode)
    print(f"dense_contextual (control): {dense}")
    manifest = {
        "kind": "D69 control (about <place>, <state> + text)",
        "enriched_sha256": sha256_file(ENRICHED), "rows": len(chunks),
        "versions": library_versions(), "model_files_sha256": model_file_hashes(d),
        "artifacts_sha256": {
            "dense_contextual.npy": sha256_file(args.out / "dense_contextual.npy"),
            **{f"bm25_contextual/{f.name}": sha256_file(f)
               for f in sorted((args.out / "bm25_contextual").iterdir())}},
        "build_seconds": round(time.monotonic() - t0, 1)}
    atomic_write_text(args.out / MANIFEST, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    problems = verify(args.out)
    print("VERIFIED" if not problems else f"VERIFY FAILED: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
