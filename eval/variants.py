"""Phase 6 ablation variants as retrieve-contract functions (D84).

Each is `f(query, filters=None, k=10) -> list[dict]` (chunk fields + "score"), searching the frozen
corpus. Filters raise NotImplementedError until row 5 adds them. Rows so far:

- row 2: hybrid_plain       RRF(bm25_plain, dense_plain)          (the two plain baselines)
- row 3: hybrid_contextual  RRF(bm25_contextual, dense_contextual)
         hybrid_control     RRF of the D69 control index (about <place>, <state> + text)
- D69 single arms for the plain/contextual/control comparison: dense_control, bm25_control
  (dense_plain, dense_contextual, bm25_plain, bm25_contextual are eval.adapters).

RRF fuses the top RRF_DEPTH (100) of each arm (D72). The control index is built by
scripts/build_control_index.py into work/phase6/control and shares the main model's query-vector
cache namespace (same bge-m3), so control dense reuses the cached query vectors.
"""
from __future__ import annotations

from pathlib import Path

from eval.adapters import ENRICHED, INDEX_DIR, RRF_DEPTH, get_index
from src.common.cache import DiskCache
from src.common.schemas import REPO_ROOT, load_embed_local_dir
from src.retrieval.build_index import load_encoder
from src.retrieval.hybrid import CONTEXTUAL_TABLES, expanded_ids, hybrid_ids
from src.retrieval.search import Index, model_id_from_manifest

CONTROL_DIR = REPO_ROOT / "work" / "phase6" / "control"

_control: Index | None = None


def get_control_index() -> Index:
    global _control
    if _control is None:
        model_dir = load_embed_local_dir()
        if model_dir is None:
            raise RuntimeError("EMBED_LOCAL_DIR is not set in .env")
        if not (CONTROL_DIR / "dense_contextual.npy").exists():
            raise RuntimeError("the control index is missing: run scripts/build_control_index.py")
        _control = Index(CONTROL_DIR, ENRICHED, encode=load_encoder(Path(model_dir)),
                         cache=DiskCache(REPO_ROOT / ".cache"),
                         model_id=model_id_from_manifest(INDEX_DIR))  # same bge-m3 -> shared cache
    return _control


def _guard(filters: dict | None) -> None:
    if filters:
        raise NotImplementedError("the row 2/3 variants do not support filters (row 5 adds them)")


def _rows(index: Index, ranked: list[tuple[str, float]]) -> list[dict]:
    return [{**index.chunks[cid], "score": score} for cid, score in ranked]


def _hybrid(index: Index, tables: list[str], query: str, k: int) -> list[dict]:
    return _rows(index, hybrid_ids(index, tables, query, depth=RRF_DEPTH, limit=k))


def _arm(index: Index, table: str, query: str, k: int) -> list[dict]:
    return [{**index.chunks[r["chunk_id"]], "score": r["score"]}
            for r in index.table(table, query, k)]


def hybrid_plain(query, filters=None, k=10):
    _guard(filters)
    return _hybrid(get_index(), ["bm25_plain", "dense_plain"], query, k)


def hybrid_contextual(query, filters=None, k=10):
    _guard(filters)
    return _hybrid(get_index(), ["bm25_contextual", "dense_contextual"], query, k)


def hybrid_control(query, filters=None, k=10):
    _guard(filters)
    return _hybrid(get_control_index(), ["bm25_contextual", "dense_contextual"], query, k)


def dense_control(query, filters=None, k=10):
    _guard(filters)
    return _arm(get_control_index(), "dense_contextual", query, k)


def bm25_control(query, filters=None, k=10):
    _guard(filters)
    return _arm(get_control_index(), "bm25_contextual", query, k)


# ---- row 4: query understanding and expansion (D86) -----------------------------------------
PARSED_QUERIES = REPO_ROOT / "eval" / "parsed_queries.json"
_parsed_by_text: dict[str, dict] | None = None


def parsed_by_text(path: Path = PARSED_QUERIES) -> dict[str, dict]:
    """The committed query-understanding output, keyed by query text. Row 4 reads this file instead
    of calling the LLM, so the ablation is reproducible without a network (D86)."""
    global _parsed_by_text
    if _parsed_by_text is None:
        import json

        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        _parsed_by_text = {p["text"]: p for p in doc["queries"].values()}
    return _parsed_by_text


def expanded_hybrid(query, filters=None, k=10):
    """Row 4: the request and its sub-queries each go through the row-3 contextual hybrid and the
    per-query rankings are fused with RRF. A query missing from the parsed file raises (no silent
    fallback to the plain row-3 result)."""
    _guard(filters)
    try:
        parsed = parsed_by_text()[query]
    except KeyError:
        raise KeyError("the query is not in eval/parsed_queries.json; run "
                       "scripts/parse_queries.py") from None
    index = get_index()
    return _rows(index, expanded_ids(index, CONTEXTUAL_TABLES, [query, *parsed["sub_queries"]],
                                     depth=RRF_DEPTH, limit=k))
