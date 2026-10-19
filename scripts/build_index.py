"""Build and verify the Phase 3 indexes from the enriched chunks (D51, D52, D55, D61). No API call.

    python scripts/build_index.py             # build all from data/enriched/chunks.jsonl
    python scripts/build_index.py --verify    # re-check an existing index
    python scripts/build_index.py --bench 200 # time the CPU encoder on 200 frozen texts; no output
    python scripts/build_index.py --source X --out DIR [--model-dir D]   # scratch build

The real build needs the COMPLETE enriched file (it must pass check_enriched_matches_frozen); any
other --source or --out is a SCRATCH build and is labelled so. The bge-m3 folder comes from
EMBED_LOCAL_DIR in .env or --model-dir; its tokenizer.json must hash to the frozen-corpus
tokenizer (D48) or the build stops. Dense builds checkpoint every 200 rows and resume on a rerun.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.cache import atomic_write_text  # noqa: E402
from src.common.schemas import REPO_ROOT, load_embed_local_dir  # noqa: E402
from src.process.process import check_enriched_matches_frozen  # noqa: E402
from src.retrieval.build_index import (  # noqa: E402
    BM25_KINDS,
    DENSE_KINDS,
    EXPECTED_TOKENIZER_SHA,
    IDS_FILE,
    MANIFEST_FILE,
    bm25_text,
    build_bm25,
    build_dense,
    dense_text,
    library_versions,
    load_chunks,
    load_encoder,
    sha256_file,
    verify_index,
    write_manifest,
)

DEFAULT_SOURCE = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
DEFAULT_OUT = REPO_ROOT / "index"
FROZEN = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
COMMITTED_MANIFEST = REPO_ROOT / "data" / "manifests" / "index_manifest.json"


def model_dir(args) -> Path:
    d = args.model_dir or load_embed_local_dir()
    if d is None:
        sys.exit("EMBED_LOCAL_DIR is not set in .env and --model-dir was not given")
    d = Path(d)
    tok = d / "tokenizer.json"
    if not tok.exists():
        sys.exit(f"{tok} is missing: is this the bge-m3 folder?")
    if sha256_file(tok) != EXPECTED_TOKENIZER_SHA:
        sys.exit("STOP: the model folder's tokenizer.json does not hash to the frozen-corpus "
                 f"tokenizer ({EXPECTED_TOKENIZER_SHA[:12]}...); refusing to embed")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", type=int, metavar="N")
    ap.add_argument("--bm25-only", action="store_true")
    args = ap.parse_args(argv)

    if args.bench:
        d = model_dir(args)
        texts = [c["text"] for c in load_chunks(FROZEN)[: args.bench]]
        t0 = time.monotonic()
        encode = load_encoder(d)
        t1 = time.monotonic()
        vectors = encode(texts)
        t2 = time.monotonic()
        print(f"model load {t1 - t0:.1f} s; encoded {len(texts)} frozen texts in {t2 - t1:.1f} s "
              f"= {len(texts) / (t2 - t1):.2f} chunks/s on CPU; shape {vectors.shape}; "
              f"versions {library_versions()}")
        print(f"ESTIMATE for 1,696 x 2 passages: {2 * 1696 / (len(texts) / (t2 - t1)) / 60:.1f} "
              "minutes (contextual passages are longer, so a little more)")
        return 0

    real = args.out.resolve() == DEFAULT_OUT.resolve()
    if args.verify:
        problems = verify_index(args.out, args.source)
        print("VERIFIED" if not problems else "VERIFY FAILED")
        for p in problems:
            print(" -", p)
        return 1 if problems else 0

    if not args.source.exists():
        print(f"{args.source} does not exist: the enriched file is written by "
              "`python scripts/enrich.py --assemble` after the full run")
        return 2
    chunks = load_chunks(args.source)
    if real:
        if args.source.resolve() != DEFAULT_SOURCE.resolve():
            print("the real index (index/) is built only from data/enriched/chunks.jsonl; "
                  "use --out for a scratch build")
            return 2
        check_enriched_matches_frozen(chunks, load_chunks(FROZEN))
    else:
        print(f"SCRATCH BUILD ({len(chunks)} rows from {args.source.name} into {args.out}): "
              "not the real index")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"scratch": not real}
    for kind in BM25_KINDS:
        stats[f"bm25_{kind}"] = build_bm25([bm25_text(c, kind) for c in chunks],
                                           out / f"bm25_{kind}")
        print(f"bm25_{kind}: {stats[f'bm25_{kind}']}")
    atomic_write_text(out / IDS_FILE, json.dumps([c["chunk_id"] for c in chunks]) + "\n")
    d = None
    if not args.bm25_only:
        d = model_dir(args)
        encode = load_encoder(d)
        for kind in DENSE_KINDS:
            stats[f"dense_{kind}"] = build_dense([dense_text(c, kind) for c in chunks],
                                                 out / f"dense_{kind}.npy", encode)
            print(f"dense_{kind}: {stats[f'dense_{kind}']}")
    manifest = write_manifest(out, args.source, d, stats)
    problems = verify_index(out, args.source) if not args.bm25_only else []
    print(f"manifest written ({len(manifest['artifacts_sha256'])} artifacts); verify: "
          f"{'OK' if not problems else problems}")
    if real and not problems:
        shutil.copyfile(out / MANIFEST_FILE, COMMITTED_MANIFEST)
        print("copied the manifest (hashes only, no text) to "
              f"{COMMITTED_MANIFEST.relative_to(REPO_ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
