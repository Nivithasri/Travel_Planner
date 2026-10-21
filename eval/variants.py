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


# ---- rows 5 and 6: metadata filters, then freshness (D85, D87) --------------------------------
# Both act on the 100 best candidates of a base ranking (row 4 expanded, or, as a diagnostic while
# row 4 is provisional, the plain row-3 hybrid) and use the validated parse of the query: its
# states, season, month, place_types and the predicted time_sensitive flag. The gold fields of
# queries.json are never read.
from src.retrieval.filters import load_provenance  # noqa: E402
from src.retrieval.postprocess import PostConfig, postprocess  # noqa: E402

PROVENANCE_FILE = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"
_provenance: dict[str, str] | None = None


def get_provenance() -> dict[str, str]:
    global _provenance
    if _provenance is None:
        _provenance = load_provenance(PROVENANCE_FILE)
    return _provenance


def run_post(query: str, config: PostConfig, k: int = 10, *, base: str = "expanded",
             explicit_filters: dict | None = None):
    """(rows, trace) for one query. base is "expanded" (row 4) or "hybrid" (row 3)."""
    from src.common.schemas import load_reference_date

    parsed = parsed_by_text()
    if query not in parsed:
        raise KeyError("the query is not in eval/parsed_queries.json; run "
                       "scripts/parse_queries.py")
    parse = parsed[query]
    index = get_index()
    queries = [query, *parse["sub_queries"]] if base == "expanded" else [query]
    ranked = expanded_ids(index, CONTEXTUAL_TABLES, queries, depth=RRF_DEPTH, limit=RRF_DEPTH)
    top, trace = postprocess(ranked, index.chunks, get_provenance(), parse, config, k,
                             reference=load_reference_date(), explicit_filters=explicit_filters)
    return _rows(index, top), trace


def _post_variant(config: PostConfig, base: str = "expanded"):
    def retrieve(query, filters=None, k=10):
        rows, _ = run_post(query, config, k, base=base, explicit_filters=filters)
        return rows
    return retrieve


# row 5: filters on top of row 4; 5b filters place_type on every row (rule-based fields used)
row5_filters = _post_variant(PostConfig(use_filters=True))
row5b_filters_all_rows = _post_variant(PostConfig(use_filters=True, rule_based_place_type=True))
# row 6: freshness on top of row 5, one function per weight w of the dev grid
row6_fresh_w10 = _post_variant(PostConfig(use_filters=True, fresh_w=0.1))
row6_fresh_w20 = _post_variant(PostConfig(use_filters=True, fresh_w=0.2))
row6_fresh_w30 = _post_variant(PostConfig(use_filters=True, fresh_w=0.3))
# diagnostics on the plain row-3 hybrid base, because row 4 is provisional until round 2
hybrid_filters = _post_variant(PostConfig(use_filters=True), base="hybrid")
hybrid_filters_fresh_w20 = _post_variant(PostConfig(use_filters=True, fresh_w=0.2), base="hybrid")


# ---- row 8: MMR over the reranked top 30 (D88, D89) -------------------------------------------
# Row 7 (the cross-encoder) is not built yet: run_mmr takes ANY scorer (query, ids) -> logits, so
# MMR is tested and wired now and the real scorer plugs in with row 7. Until then proxy_scorer is
# a STAND-IN (standardised dense cosine) used only to check the wiring; numbers made with it are
# NOT row 8 and are never reported as such.
import numpy as np  # noqa: E402

from src.retrieval import mmr as mmr_mod  # noqa: E402
from src.retrieval import rerank as rerank_mod  # noqa: E402


def proxy_scorer(query: str, ids: list[str]) -> dict[str, float]:
    """STAND-IN reranker scores: dense_contextual cosine to the query, standardised over the
    candidates and scaled to logit size. Wiring checks only."""
    index = get_index()
    matrix, row_of = index.dense_matrix("contextual")
    cos = np.array([float(matrix[row_of[c]] @ index.query_vector(query)) for c in ids])
    z = (cos - cos.mean()) / (cos.std() or 1.0)
    return {c: float(2.0 * v) for c, v in zip(ids, z)}


def run_mmr(query: str, lam: float, scorer, k: int = 10, *, depth: int = rerank_mod.DEPTH):
    """(rows, trace): the row-4 expanded ranking, its top `depth` re-ordered by `scorer`, then by
    MMR at `lam`. Filters and freshness are not applied (row 5 failed its keep rule and row 6 is
    off on dev, LOG 2026-10-21); row 4 itself is provisional until round 2 (D91)."""
    parsed = parsed_by_text()
    if query not in parsed:
        raise KeyError("the query is not in eval/parsed_queries.json; run "
                       "scripts/parse_queries.py")
    parse = parsed[query]
    index = get_index()
    ranked = expanded_ids(index, CONTEXTUAL_TABLES, [query, *parse["sub_queries"]],
                          depth=RRF_DEPTH, limit=RRF_DEPTH)
    logits = scorer(query, [cid for cid, _ in ranked[:depth]])
    reranked = rerank_mod.apply_scores(ranked, logits, depth)
    matrix, row_of = index.dense_matrix("contextual")
    place_term = not parse["places"]  # a place-named query: another place is not diversity
    sim = mmr_mod.make_similarity(matrix, row_of, index.chunks, place_term=place_term)
    final = mmr_mod.apply_mmr(reranked, logits, sim, lam, depth)
    return _rows(index, final[:k]), {"lam": lam, "depth": depth, "place_term": place_term}


def make_mmr_variant(lam: float, scorer):
    def retrieve(query, filters=None, k=10):
        _guard(filters)
        return run_mmr(query, lam, scorer, k)[0]
    return retrieve


# ---- row 7: cross-encoder rerank of the top 30 (D88) ------------------------------------------
RERANK_TEXT_KIND = "contextual"  # D88: row 3 favours contextual on dev (dense 0.829 vs 0.701)
MMR_LAMBDA_FILE = REPO_ROOT / "eval" / "mmr_lambda.json"
_ce: object | None = None


def get_ce_scorer():
    """The bge-reranker-v2-m3 scorer from RERANK_LOCAL_DIR (offline, cached per query and chunk)."""
    global _ce
    if _ce is None:
        from src.common.schemas import load_rerank_local_dir
        from src.retrieval.rerank import CrossEncoderScorer

        model_dir = load_rerank_local_dir()
        if model_dir is None or not Path(model_dir).is_dir():
            raise RuntimeError("RERANK_LOCAL_DIR is not set in .env or is not a folder: download "
                               "BAAI/bge-reranker-v2-m3 into models/ and set it (D88)")
        _ce = CrossEncoderScorer(Path(model_dir), get_index().chunks,
                                 DiskCache(REPO_ROOT / ".cache"), text_kind=RERANK_TEXT_KIND,
                                 log=print)
    return _ce


def run_rerank(query: str, scorer, k: int = 10, *, base: str = "expanded"):
    """The base ranking's top 30 re-ordered by the scorer's logits; ranks 31-100 unchanged."""
    parsed = parsed_by_text()
    if query not in parsed:
        raise KeyError("the query is not in eval/parsed_queries.json; run "
                       "scripts/parse_queries.py")
    index = get_index()
    queries = [query, *parsed[query]["sub_queries"]] if base == "expanded" else [query]
    ranked = expanded_ids(index, CONTEXTUAL_TABLES, queries, depth=RRF_DEPTH, limit=RRF_DEPTH)
    logits = scorer(query, [cid for cid, _ in ranked[:rerank_mod.DEPTH]])
    return _rows(index, rerank_mod.apply_scores(ranked, logits, rerank_mod.DEPTH)[:k])


def row7_rerank(query, filters=None, k=10):
    _guard(filters)
    return run_rerank(query, get_ce_scorer(), k)


def hybrid_rerank(query, filters=None, k=10):
    """Diagnostic: the same rerank on the plain row-3 hybrid base (row 4 is provisional)."""
    _guard(filters)
    return run_rerank(query, get_ce_scorer(), k, base="hybrid")


# ---- row 8 proper: MMR at the lambda chosen on dev by scripts/mmr_curve.py (D89) ----------------
def chosen_lambda() -> float:
    import json

    if not MMR_LAMBDA_FILE.exists():
        raise FileNotFoundError("eval/mmr_lambda.json is missing: run "
                                "python scripts/mmr_curve.py --scorer ce")
    return float(json.loads(MMR_LAMBDA_FILE.read_text(encoding="utf-8"))["lam"])


def row8_mmr(query, filters=None, k=10):
    _guard(filters)
    return run_mmr(query, chosen_lambda(), get_ce_scorer(), k)[0]
