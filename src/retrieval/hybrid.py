"""Hybrid and expanded retrieval over an Index (D84 rows 2-4). Pure given the index.

- hybrid_ids: RRF of a query's rankings from several index tables (BM25 + dense), each taken to
  `depth` (D72), cut to `limit` after fusion.
- expanded_ids: one hybrid ranking per query (the request and its sub-queries), then RRF across
  those rankings. The request counts as one list like each sub-query; nothing is weighted.

`index` needs `.table(name, query, k) -> [{"chunk_id", ...}]` (src.retrieval.search.Index).
"""
from __future__ import annotations

from src.retrieval.fusion import rrf

DEPTH = 100
CONTEXTUAL_TABLES = ("bm25_contextual", "dense_contextual")
PLAIN_TABLES = ("bm25_plain", "dense_plain")


def hybrid_ids(index, tables: tuple[str, ...] | list[str], query: str, *, depth: int = DEPTH,
               limit: int | None = None) -> list[tuple[str, float]]:
    lists = [[r["chunk_id"] for r in index.table(name, query, depth)] for name in tables]
    fused = rrf(lists)
    return fused if limit is None else fused[:limit]


def expanded_ids(index, tables: tuple[str, ...] | list[str], queries: list[str], *,
                 depth: int = DEPTH, limit: int | None = None) -> list[tuple[str, float]]:
    per_query = [[cid for cid, _ in hybrid_ids(index, tables, q, depth=depth, limit=depth)]
                 for q in queries]
    fused = rrf(per_query)
    return fused if limit is None else fused[:limit]
