"""The four Phase 3 indexes and their RRF fusion as `retrieve`-contract functions (D72):

    python eval/run_eval.py --retriever eval.adapters:dense_plain --split dev

Each function is retrieve(query, filters=None, k=10) -> list of chunk dicts with a "score". They
search the whole frozen corpus; filters are not supported and raise (Phase 6 adds them).
Dense search needs the local bge-m3 folder (EMBED_LOCAL_DIR in .env); query vectors are cached
on disk (CLAUDE.md rule 4).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.common.cache import DiskCache
from src.common.schemas import REPO_ROOT, load_embed_local_dir
from src.retrieval.build_index import load_encoder
from src.retrieval.search import TABLES, Index, model_id_from_manifest

INDEX_DIR = REPO_ROOT / "index"
ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
RRF_K = 60
RRF_DEPTH = 100
SYSTEMS = tuple(name for name, _, _ in TABLES) + ("rrf4",)

_index: Index | None = None


def get_index() -> Index:
    global _index
    if _index is None:
        model_dir = load_embed_local_dir()
        if model_dir is None:
            raise RuntimeError("EMBED_LOCAL_DIR is not set in .env")
        _index = Index(INDEX_DIR, ENRICHED, encode=load_encoder(Path(model_dir)),
                       cache=DiskCache(REPO_ROOT / ".cache"),
                       model_id=model_id_from_manifest(INDEX_DIR))
    return _index


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1. Ties are
    broken by chunk id, so the order is reproducible."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda t: (-t[1], t[0]))


def system_ids(system: str, query: str, k: int,
               index: Index | None = None) -> list[tuple[str, float]]:
    """(chunk_id, score) top-k of one of SYSTEMS."""
    index = index or get_index()
    if system == "rrf4":
        lists = [[r["chunk_id"] for r in index.table(name, query, RRF_DEPTH)]
                 for name, _, _ in TABLES]
        return rrf(lists)[:k]
    return [(r["chunk_id"], r["score"]) for r in index.table(system, query, k)]


def _retrieve(system: str, query: str, filters: dict | None, k: int) -> list[dict]:
    if filters:
        raise NotImplementedError("the baseline adapters do not support filters")
    index = get_index()
    ranked = system_ids(system, query, k, index)
    return [{**index.chunks[cid], "score": score} for cid, score in ranked]


def bm25_contextual(query, filters=None, k=10):
    return _retrieve("bm25_contextual", query, filters, k)


def bm25_plain(query, filters=None, k=10):
    return _retrieve("bm25_plain", query, filters, k)


def dense_plain(query, filters=None, k=10):
    return _retrieve("dense_plain", query, filters, k)


def dense_contextual(query, filters=None, k=10):
    return _retrieve("dense_contextual", query, filters, k)


def rrf4(query, filters=None, k=10):
    return _retrieve("rrf4", query, filters, k)
