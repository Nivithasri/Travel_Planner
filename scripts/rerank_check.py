"""Check the local cross-encoder and record its hashes (D88). No network, no API call.

    python scripts/rerank_check.py                   # hashes, load, 4-pair sanity check, timing
    python scripts/rerank_check.py --write-manifest  # also data/manifests/rerank_manifest.json

Reads RERANK_LOCAL_DIR from .env. The sanity pairs are invented text about a made-up place
("Sampleville", as in the labelling rubric), never corpus text: a relevant passage must outscore an
irrelevant one for the same query. The manifest holds hashes, versions and settings only. Exit
codes: 0 ok, 2 the folder is missing, 3 the sanity check failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata as md
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.cache import atomic_write_text  # noqa: E402
from src.common.schemas import REPO_ROOT, load_rerank_local_dir  # noqa: E402
from src.retrieval import rerank  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "manifests" / "rerank_manifest.json"
QUERY = "Sampleville temple visiting hours and dress code"
PAIRS = [
    ("relevant", "The Sampleville Shiva temple opens at 5 a.m. and closes at 9 p.m. Visitors must "
                 "cover their shoulders and knees; shorts are not allowed inside the sanctum."),
    ("on topic, partial", "The Sampleville Shiva temple was built in the 12th century and is "
                          "known for its tall gateway towers and carved pillars."),
    ("irrelevant", "Sampleville had a population of 640,000 at the last census, and its main "
                   "industry is textile weaving."),
    ("irrelevant", "A luxury resort near Sample Port offers private pool villas starting at "
                   "25,000 rupees a night."),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args(argv)

    model_dir = load_rerank_local_dir()
    if model_dir is None or not Path(model_dir).is_dir():
        print("RERANK_LOCAL_DIR is not set in .env or is not a folder (download "
              "BAAI/bge-reranker-v2-m3 into models/ yourself; this never downloads)")
        return 2
    t0 = time.monotonic()
    hashes = rerank.weight_hashes(Path(model_dir))
    print(f"{len(hashes)} model files hashed in {time.monotonic() - t0:.1f} s: "
          + ", ".join(sorted(hashes)))
    print(f"cache namespace: {rerank.model_id_from(hashes)}")
    t1 = time.monotonic()
    tok, model = rerank.load_model(Path(model_dir))
    print(f"loaded in {time.monotonic() - t1:.1f} s; num_labels {model.config.num_labels}, "
          f"max_position_embeddings {model.config.max_position_embeddings}")
    import torch

    t2 = time.monotonic()
    with torch.no_grad():
        enc = tok([QUERY] * len(PAIRS), [p for _, p in PAIRS], padding=True, truncation=True,
                  max_length=rerank.MAX_LENGTH, return_tensors="pt")
        logits = [float(x) for x in model(**enc).logits.view(-1).tolist()]
    per_pair = (time.monotonic() - t2) / len(PAIRS)
    for (label, _), logit in zip(PAIRS, logits):
        print(f"  {logit:+8.3f}  sigmoid {1 / (1 + 2.718281828 ** -logit):.3f}  {label}")
    print(f"~{per_pair:.2f} s per pair on these short passages (corpus passages are longer)")
    ok = logits[0] > logits[1] > max(logits[2], logits[3])
    print("SANITY " + ("PASS: relevant > partial > irrelevant" if ok else "FAIL"))
    if args.write_manifest and ok:
        manifest = {"model": "BAAI/bge-reranker-v2-m3 (local copy; its revision is not recorded "
                             "by this check, only the file hashes)",
                    "files_sha256": hashes, "cache_namespace": rerank.model_id_from(hashes),
                    "max_length": rerank.MAX_LENGTH, "batch_size": rerank.BATCH_SIZE,
                    "device": "cpu",
                    "versions": {d: md.version(d) for d in ("transformers", "torch", "tokenizers")}}
        atomic_write_text(MANIFEST, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
