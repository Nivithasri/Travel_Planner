"""Phase 3 indexes over the enriched chunks (D51, D52, D55, D61).

Two BM25 indexes (bm25s, lucene method, k1 1.5, b 0.75, English stopwords, Snowball English
stemmer from PyStemmer): `contextual` over context_note + "\\n" + text and `plain` over text only.
Two dense matrices (sentence-transformers bge-m3 on CPU, L2-normalised float32, max_seq_length
512): `plain` embeds text, `contextual` embeds context_note + "\\n\\n" + text. Search is exact numpy
cosine (search.py). Every artifact is hashed in an index manifest; `verify_index` re-checks them.

Layout under the index directory: ids.json (chunk ids in enriched-file order), bm25_contextual/,
bm25_plain/, dense_plain.npy, dense_contextual.npy, index_manifest.json.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

# torch only: the preinstalled TensorFlow/protobuf pair breaks `import sentence_transformers`
for _var, _val in (("USE_TF", "0"), ("USE_FLAX", "0"), ("TRANSFORMERS_NO_TF", "1")):
    os.environ.setdefault(_var, _val)

BM25_PARAMS = {"library": "bm25s", "method": "lucene", "k1": 1.5, "b": 0.75,
               "stopwords": "english", "stemmer": "snowball english (PyStemmer)",
               "token_pattern": "bm25s default (?u)\\b\\w\\w+\\b", "lowercase": True}
DENSE_PARAMS = {"library": "sentence-transformers", "model": "BAAI/bge-m3", "dim": 1024,
                "max_seq_length": 512, "normalize": True, "dtype": "float32", "device": "cpu",
                "batch_size": 16, "contextual_separator": "\n\n", "query_prefix": ""}
CHECKPOINT_ROWS = 200
BM25_KINDS = ("contextual", "plain")
DENSE_KINDS = ("plain", "contextual")
IDS_FILE = "ids.json"
MANIFEST_FILE = "index_manifest.json"
MODEL_FILES = ("pytorch_model.bin", "tokenizer.json", "config.json", "sentencepiece.bpe.model",
               "tokenizer_config.json", "special_tokens_map.json", "modules.json",
               "sentence_bert_config.json", "config_sentence_transformers.json",
               "1_Pooling/config.json")
EXPECTED_TOKENIZER_SHA = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


# ==== passage texts =================================================================

def bm25_text(chunk: dict, kind: str) -> str:
    return f"{chunk['context_note']}\n{chunk['text']}" if kind == "contextual" else chunk["text"]


def dense_text(chunk: dict, kind: str) -> str:
    if kind == "contextual":
        return f"{chunk['context_note']}{DENSE_PARAMS['contextual_separator']}{chunk['text']}"
    return chunk["text"]


# ==== BM25 ==========================================================================

def make_stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


def bm25_tokens(texts: list[str]) -> list[list[str]]:
    import bm25s

    return bm25s.tokenize(texts, stopwords="english", stemmer=make_stemmer(), return_ids=False,
                          show_progress=False)


def build_bm25(texts: list[str], out_dir: Path) -> dict:
    import bm25s

    started = time.monotonic()
    retriever = bm25s.BM25(method="lucene", k1=BM25_PARAMS["k1"], b=BM25_PARAMS["b"])
    retriever.index(bm25_tokens(texts), show_progress=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    retriever.save(str(out_dir), show_progress=False)
    return {"documents": len(texts), "seconds": round(time.monotonic() - started, 2)}


def load_bm25(index_dir: Path):
    import bm25s

    return bm25s.BM25.load(str(index_dir), load_corpus=False, show_progress=False)


def bm25_scores(retriever, query: str, n_docs: int) -> np.ndarray:
    """BM25 score of every document for `query` (zeros when no query token is in the index)."""
    tokens = bm25_tokens([query])[0]
    known = [t for t in tokens if t in retriever.vocab_dict]
    if not known:
        return np.zeros(n_docs, dtype=np.float32)
    return np.asarray(retriever.get_scores(known), dtype=np.float32)


# ==== dense =========================================================================

def load_encoder(model_dir: Path) -> Callable[[list[str]], np.ndarray]:
    """sentence-transformers bge-m3 on CPU, normalised float32 vectors (D51)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = DENSE_PARAMS["max_seq_length"]

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=DENSE_PARAMS["batch_size"],
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)

    return encode


def _texts_sha(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()


def _atomic_save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def build_dense(texts: list[str], out_path: Path, encode: Callable[[list[str]], np.ndarray], *,
                checkpoint_rows: int = CHECKPOINT_ROWS, log: Callable[[str], None] = print) -> dict:
    """Embed `texts` in order, saving a partial matrix after every `checkpoint_rows` rows. A rerun
    resumes from the partial file when the texts it covered are unchanged; otherwise it starts
    over. The final matrix replaces the partial one atomically."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial, meta = out_path.with_name(out_path.name + ".partial.npy"), out_path.with_name(
        out_path.name + ".partial.json")
    rows: list[np.ndarray] = []
    done = 0
    if partial.exists() and meta.exists():
        saved = np.load(partial)
        info = json.loads(meta.read_text(encoding="utf-8"))
        if (info["rows"] == len(saved) <= len(texts)
                and info["texts_sha256"] == _texts_sha(texts[: len(saved)])):
            rows, done = [saved], len(saved)
            log(f"resuming {out_path.name} from row {done} of {len(texts)}")
        else:
            log(f"discarding a stale partial file for {out_path.name}")
    started, resumed_from = time.monotonic(), done
    while done < len(texts):
        block = texts[done: done + checkpoint_rows]
        vectors = encode(block)
        if vectors.shape[0] != len(block):
            raise RuntimeError("the encoder returned a different number of rows")
        rows.append(vectors)
        done += len(block)
        joined = np.concatenate(rows) if len(rows) > 1 else rows[0]
        rows = [joined]
        _atomic_save(partial, joined)
        meta.write_text(json.dumps({"rows": done, "texts_sha256": _texts_sha(texts[:done])}),
                        encoding="utf-8")
        log(f"{out_path.name}: {done}/{len(texts)} rows")
    matrix = rows[0] if rows else np.zeros((0, DENSE_PARAMS["dim"]), dtype=np.float32)
    _atomic_save(out_path, matrix.astype(np.float32))
    partial.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    seconds = time.monotonic() - started
    new_rows = len(texts) - resumed_from
    return {"rows": len(texts), "resumed_from": resumed_from, "seconds": round(seconds, 2),
            "rows_per_second": round(new_rows / seconds, 3) if seconds > 0 and new_rows else None}


# ==== manifest and verification =====================================================

def library_versions() -> dict[str, str]:
    out = {}
    for dist in ("bm25s", "PyStemmer", "sentence-transformers", "torch", "numpy", "transformers",
                 "tokenizers"):
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    """sha256 of every weight and config file of the local bge-m3 folder (those that exist)."""
    return {name: sha256_file(Path(model_dir) / name) for name in MODEL_FILES
            if (Path(model_dir) / name).exists()}


def artifact_hashes(index_dir: Path) -> dict[str, str]:
    index_dir = Path(index_dir)
    hashes = {}
    for sub in (index_dir / "bm25_contextual", index_dir / "bm25_plain"):
        if sub.is_dir():
            for f in sorted(sub.iterdir()):
                hashes[f"{sub.name}/{f.name}"] = sha256_file(f)
    for name in (IDS_FILE, "dense_plain.npy", "dense_contextual.npy"):
        if (index_dir / name).exists():
            hashes[name] = sha256_file(index_dir / name)
    return hashes


def write_manifest(index_dir: Path, enriched_path: Path, model_dir: Path | None,
                   stats: dict, extra: dict | None = None) -> dict:
    index_dir = Path(index_dir)
    ids = json.loads((index_dir / IDS_FILE).read_text(encoding="utf-8"))
    manifest = {
        "enriched_file": str(enriched_path.name), "enriched_sha256": sha256_file(enriched_path),
        "rows": len(ids), "versions": library_versions(), "bm25_params": BM25_PARAMS,
        "dense_params": DENSE_PARAMS,
        "model_files_sha256": model_file_hashes(model_dir) if model_dir else {},
        "artifacts_sha256": artifact_hashes(index_dir), "build_stats": stats,
        **(extra or {}),
    }
    (index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return manifest


def verify_index(index_dir: Path, enriched_path: Path) -> list[str]:
    """Problems found (empty list = verified): ids == enriched order, enriched sha256 equals the
    manifest's, artifact hashes, dense shape/dtype/NaN/unit norms, BM25 index loads with the right
    document count."""
    index_dir, problems = Path(index_dir), []
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return [f"{MANIFEST_FILE} is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = load_chunks(enriched_path)
    enriched_ids = [c["chunk_id"] for c in chunks]
    if sha256_file(enriched_path) != manifest["enriched_sha256"]:
        problems.append("the enriched chunks.jsonl is not the file the index was built from "
                        "(sha256 differs)")
    ids_path = index_dir / IDS_FILE
    ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else None
    if ids is None:
        problems.append(f"{IDS_FILE} is missing")
    elif ids != enriched_ids:
        problems.append("ids.json does not equal the chunk ids of the enriched file, in order")
    if manifest["rows"] != len(enriched_ids):
        problems.append(f"manifest rows {manifest['rows']} != enriched rows {len(enriched_ids)}")
    now = artifact_hashes(index_dir)
    for name, digest in manifest["artifacts_sha256"].items():
        if now.get(name) != digest:
            problems.append(f"artifact {name} is missing or changed since the build")
    for kind in DENSE_KINDS:
        path = index_dir / f"dense_{kind}.npy"
        if not path.exists():
            problems.append(f"dense_{kind}.npy is missing")
            continue
        m = np.load(path)
        if m.shape != (len(enriched_ids), DENSE_PARAMS["dim"]):
            problems.append(f"dense_{kind}.npy shape {m.shape}, expected "
                            f"{(len(enriched_ids), DENSE_PARAMS['dim'])}")
        if m.dtype != np.float32:
            problems.append(f"dense_{kind}.npy dtype {m.dtype}, expected float32")
        if not np.isfinite(m).all():
            problems.append(f"dense_{kind}.npy contains NaN or inf")
        elif m.size and not np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-3):
            problems.append(f"dense_{kind}.npy rows are not unit-norm")
    for kind in BM25_KINDS:
        sub = index_dir / f"bm25_{kind}"
        try:
            retriever = load_bm25(sub)
            if retriever.scores["num_docs"] != len(enriched_ids):
                problems.append(f"bm25_{kind} holds {retriever.scores['num_docs']} documents, "
                                f"expected {len(enriched_ids)}")
        except Exception as exc:  # report, do not crash the verification
            problems.append(f"bm25_{kind} cannot be loaded: {type(exc).__name__}")
    return problems
