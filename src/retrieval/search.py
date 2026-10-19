"""Ranked search over the Phase 3 indexes (D52, D55): BM25 contextual/plain and exact numpy cosine
over the dense plain/contextual matrices. Frozen-corpus only. Ties are broken by chunk id (score
descending, then chunk_id ascending), so a ranking is reproducible.

The query vector comes from the same local bge-m3 encoder as the corpus; it is cached on disk
(CLAUDE.md rule 4), keyed by the model's tokenizer hash and the query text.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.common.cache import DiskCache
from src.retrieval.build_index import (
    BM25_KINDS,
    DENSE_KINDS,
    DENSE_PARAMS,
    IDS_FILE,
    bm25_scores,
    load_bm25,
    load_chunks,
    sha256_file,
)

SNIPPET_CHARS = 110
TABLES = (("bm25_contextual", "bm25", "contextual"), ("bm25_plain", "bm25", "plain"),
          ("dense_plain", "dense", "plain"), ("dense_contextual", "dense", "contextual"))


def rank(scores: np.ndarray, ids: list[str], k: int, *, positive_only: bool = False
         ) -> list[tuple[str, float]]:
    """Top-k (chunk_id, score), score descending then chunk_id ascending."""
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    out = [(ids[i], float(scores[i])) for i in order]
    if positive_only:
        out = [t for t in out if t[1] > 0]
    return out[:k]


class Index:
    def __init__(self, index_dir: Path, enriched_path: Path,
                 encode: Callable[[list[str]], np.ndarray] | None = None,
                 cache: DiskCache | None = None, model_id: str = "bge-m3-local"):
        self.index_dir = Path(index_dir)
        self.chunks = {c["chunk_id"]: c for c in load_chunks(enriched_path)}
        self.ids: list[str] = json.loads((self.index_dir / IDS_FILE).read_text(encoding="utf-8"))
        self._bm25 = {}
        self._dense = {}
        self._encode = encode
        self._cache = cache
        self._model_id = model_id

    # ---- BM25 ------------------------------------------------------------------------
    def bm25(self, kind: str, query: str, k: int = 10) -> list[dict]:
        if kind not in BM25_KINDS:
            raise ValueError(f"bm25 kind must be one of {BM25_KINDS}")
        if kind not in self._bm25:
            self._bm25[kind] = load_bm25(self.index_dir / f"bm25_{kind}")
        scores = bm25_scores(self._bm25[kind], query, len(self.ids))
        return self._rows(rank(scores, self.ids, k, positive_only=True))

    # ---- dense -----------------------------------------------------------------------
    def query_vector(self, query: str) -> np.ndarray:
        if self._encode is None:
            raise RuntimeError("no encoder: pass encode= (src.retrieval.build_index.load_encoder)")
        inputs = {"query": query, "max_seq_length": DENSE_PARAMS["max_seq_length"],
                  "prefix": DENSE_PARAMS["query_prefix"]}

        def compute() -> list[float]:
            return [float(x) for x in self._encode([query])[0]]

        if self._cache is not None:
            vec = self._cache.get_or_compute("embed_query", self._model_id, inputs, compute)
        else:
            vec = compute()
        return np.asarray(vec, dtype=np.float32)

    def dense(self, kind: str, query: str, k: int = 10) -> list[dict]:
        if kind not in DENSE_KINDS:
            raise ValueError(f"dense kind must be one of {DENSE_KINDS}")
        if kind not in self._dense:
            self._dense[kind] = np.load(self.index_dir / f"dense_{kind}.npy")
        scores = self._dense[kind] @ self.query_vector(query)
        return self._rows(rank(scores, self.ids, k))

    # ---- shared ----------------------------------------------------------------------
    def table(self, name: str, query: str, k: int = 10) -> list[dict]:
        for table_name, family, kind in TABLES:
            if table_name == name:
                return self.bm25(kind, query, k) if family == "bm25" else self.dense(kind, query, k)
        raise ValueError(f"unknown table {name!r}")

    def _rows(self, ranked: list[tuple[str, float]]) -> list[dict]:
        out = []
        for chunk_id, score in ranked:
            c = self.chunks[chunk_id]
            text = " ".join(c["text"].split())
            out.append({"chunk_id": chunk_id, "place": c["place"], "source_type": c["source_type"],
                        "score": round(score, 4),
                        "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS
                                                           else "")})
        return out


def model_id_for(model_dir: Path) -> str:
    """Cache namespace of the local encoder: a hash of its tokenizer file, never a path."""
    return "bge-m3-local-" + sha256_file(Path(model_dir) / "tokenizer.json")[:12]
