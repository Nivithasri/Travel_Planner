"""retrieve(): the frozen-corpus retrieval function of the project (D84, D85, D96).

    retrieve(query, filters=None, k=10) -> list[dict]      # chunk fields + "score"

FROZEN_CONFIG is the pipeline the Phase 6 keep rule left standing on the dev judgments (D96): RRF of
BM25 and dense search over the CONTEXTUAL text of every chunk (ablation row 3). It needs no LLM,
no reranker and no parse: one query in, a ranked list out. Every other ablated component stays in
this module behind a RetrieveConfig switch, exactly as it was measured, so that Phase 7 can turn
one on knowingly:

    expansion        request + 3-5 LLM sub-queries, each through the hybrid, RRF-fused (row 4)
    use_filters      state / season / place_type filters from the parse (row 5; D69 exemptions)
    fresh_w          aspect- and query-aware freshness weight (row 6)
    rerank           cross-encoder rescoring of the top 30 (row 7)
    mmr_lambda       MMR over the reranked 30 (row 8)
    corrective_tau   one corrective pass when a CRAG trigger fires (row 9)

Only combinations that were ablated are allowed (the constructor refuses the rest). The expensive
parts are injected, never built implicitly: `parser(query) -> parse dict` (query understanding),
`scorer(query, ids) -> logits` and `rewriter(query, triggers) -> rewrite | None`. No gold field of
eval/queries.json is ever read; the parser receives the query TEXT only.

`filters` (D85) is {"state": [..], "season": .., "month": 1-12, "place_type": [..]}; an unknown key
raises. Explicit filters work with any config that does not rerank (reranked pipelines with
explicit filters were not ablated); candidates that fail a filter are moved behind the passing ones,
so a short list is back-filled from the unfiltered ranking. mentioned_places is never read (D50).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from src.common.cache import DiskCache
from src.common.schemas import REPO_ROOT, load_embed_local_dir, load_reference_date
from src.retrieval import grader, mmr
from src.retrieval import rerank as rerank_mod
from src.retrieval.build_index import load_encoder
from src.retrieval.filters import load_provenance
from src.retrieval.hybrid import CONTEXTUAL_TABLES, expanded_ids, hybrid_ids
from src.retrieval.postprocess import PostConfig, postprocess
from src.retrieval.search import Index, model_id_from_manifest

INDEX_DIR = REPO_ROOT / "index"
ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
PROVENANCE = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"
PARSED_QUERIES = REPO_ROOT / "eval" / "parsed_queries.json"
REWRITES = REPO_ROOT / "eval" / "corrective_rewrites.json"
DEPTH = 100  # candidates per hybrid ranking (D72)
MMR_LAMBDA = 0.9  # D89, chosen on dev
CORRECTIVE_TAU = 0.6314  # D90, chosen on dev

Parser = Callable[[str], dict]
Rewriter = Callable[[str, list[str]], "str | None"]


@dataclass(frozen=True)
class RetrieveConfig:
    name: str = "row3_contextual_hybrid"
    expansion: bool = False
    use_filters: bool = False
    rule_based_place_type: bool = False  # diagnostic 5b only
    fresh_w: float = 0.0
    rerank: bool = False
    mmr_lambda: float | None = None
    corrective_tau: float | None = None

    def __post_init__(self) -> None:
        if (self.mmr_lambda is not None or self.corrective_tau is not None) and not (
                self.expansion and self.rerank):
            raise ValueError("MMR and the corrective pass were ablated on top of expansion and "
                             "rerank only (rows 8 and 9)")
        if self.rerank and (self.use_filters or self.fresh_w):
            raise ValueError("filters and freshness were not ablated together with rerank")
        if self.mmr_lambda is not None and not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if self.fresh_w < 0:
            raise ValueError("fresh_w must be >= 0")

    @property
    def needs_parse(self) -> bool:
        return (self.expansion or self.use_filters or self.fresh_w > 0
                or self.mmr_lambda is not None or self.corrective_tau is not None)


# The frozen pipeline (D96): what the D84 keep rule leaves standing on the dev judgments.
FROZEN_CONFIG = RetrieveConfig()
# The full chain as measured (rows 4, 7, 8, 9); NOT frozen, available to Phase 7 (D96).
FULL_CONFIG = RetrieveConfig(name="full_chain", expansion=True, rerank=True,
                             mmr_lambda=MMR_LAMBDA, corrective_tau=CORRECTIVE_TAU)


def default_index() -> Index:
    """The real index with the local bge-m3 encoder (D51); query vectors are disk-cached."""
    model_dir = load_embed_local_dir()
    if model_dir is None:
        raise RuntimeError("EMBED_LOCAL_DIR is not set in .env")
    return Index(INDEX_DIR, ENRICHED, encode=load_encoder(Path(model_dir)),
                 cache=DiskCache(REPO_ROOT / ".cache"), model_id=model_id_from_manifest(INDEX_DIR))


def parser_from_file(path: Path = PARSED_QUERIES) -> Parser:
    """A parser that reads the committed query-understanding output (D86): reproducible, no LLM."""
    import json

    by_text = {p["text"]: p for p in json.loads(Path(path).read_text(
        encoding="utf-8"))["queries"].values()}

    def parse(query: str) -> dict:
        if query not in by_text:
            raise KeyError("the query is not in eval/parsed_queries.json; run "
                           "scripts/parse_queries.py")
        return by_text[query]

    return parse


def rewriter_from_file(path: Path = REWRITES) -> Rewriter:
    """A rewriter that reads the stored corrective rewrites (D90): reproducible, no LLM."""
    store = grader.load_rewrites(path)

    def rewrite(query: str, triggers: list[str]) -> str | None:
        return grader.lookup(store, query, triggers)

    return rewrite


class Retriever:
    def __init__(self, config: RetrieveConfig = FROZEN_CONFIG, *, index: Index | None = None,
                 provenance: dict[str, str] | None = None, parser: Parser | None = None,
                 scorer: rerank_mod.Scorer | None = None, rewriter: Rewriter | None = None,
                 reference: date | None = None):
        self.config = config
        self._index, self._provenance = index, provenance
        self.parser, self.scorer, self.rewriter = parser, scorer, rewriter
        self._reference = reference
        if config.needs_parse and parser is None:
            raise ValueError(f"config {config.name!r} needs a parser (query understanding)")
        if config.rerank and scorer is None:
            raise ValueError(f"config {config.name!r} needs a reranker scorer")
        if config.corrective_tau is not None and rewriter is None:
            raise ValueError(f"config {config.name!r} needs a rewriter")

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = default_index()
        return self._index

    @property
    def provenance(self) -> dict[str, str]:
        if self._provenance is None:
            self._provenance = load_provenance(PROVENANCE)
        return self._provenance

    def _rows(self, ranked: list[tuple[str, float]]) -> list[dict]:
        return [{**self.index.chunks[cid], "score": score} for cid, score in ranked]

    # ---- the reranked pipeline (rows 7 to 9), the same steps as the ablation -----------------
    def _reranked(self, query: str, parse: dict | None, extra: tuple[str, ...] = (),
                  fetch: int = DEPTH):
        cfg, index = self.config, self.index
        if cfg.expansion:
            queries = [query, *parse["sub_queries"], *extra]
            ranked = expanded_ids(index, CONTEXTUAL_TABLES, queries, depth=fetch, limit=fetch)
        else:
            ranked = hybrid_ids(index, CONTEXTUAL_TABLES, query, depth=fetch, limit=fetch)
        logits = self.scorer(query, [cid for cid, _ in ranked[:rerank_mod.DEPTH]])
        out = rerank_mod.apply_scores(ranked, logits, rerank_mod.DEPTH)
        if cfg.mmr_lambda is not None:
            matrix, row_of = index.dense_matrix("contextual")
            sim = mmr.make_similarity(matrix, row_of, index.chunks,
                                      place_term=not parse["places"])
            out = mmr.apply_mmr(out, logits, sim, cfg.mmr_lambda, rerank_mod.DEPTH)
        return out, logits

    def search(self, query: str, filters: dict | None = None, k: int = 10
               ) -> tuple[list[dict], dict]:
        cfg = self.config
        if filters and cfg.rerank:
            raise NotImplementedError("explicit filters with a reranked pipeline were not "
                                      "ablated; use the frozen config or filter the results")
        t0 = time.monotonic()
        parse = self.parser(query) if cfg.needs_parse else None
        trace: dict = {"config": cfg.name, "k": k}
        if cfg.rerank:
            ranked, logits = self._reranked(query, parse)
            trace["passes"] = 0
            if cfg.corrective_tau is not None:
                top = [self.index.chunks[cid] for cid, _ in ranked[:grader.TOP_N]]
                triggers = grader.fired(grader.top_confidence(logits), top, cfg.corrective_tau)
                trace["triggers"] = triggers
                if triggers:  # exactly one corrective pass (plan.md item 9, D90)
                    rewrite = self.rewriter(query, triggers)
                    ranked, logits = self._reranked(query, parse, (rewrite,) if rewrite else (),
                                                    grader.CORRECTION_DEPTH)
                    trace.update(passes=1, rewrite=rewrite)
            top_k = ranked[:k]
        else:
            if cfg.expansion:
                base = expanded_ids(self.index, CONTEXTUAL_TABLES,
                                    [query, *parse["sub_queries"]], depth=DEPTH, limit=DEPTH)
            else:
                base = hybrid_ids(self.index, CONTEXTUAL_TABLES, query, depth=DEPTH, limit=DEPTH)
            needs_post = bool(filters) or cfg.use_filters or cfg.fresh_w > 0
            if needs_post:
                if cfg.fresh_w > 0 and self._reference is None:
                    self._reference = load_reference_date()
                top_k, post = postprocess(
                    base, self.index.chunks, self.provenance, parse or {},
                    PostConfig(use_filters=cfg.use_filters,
                               rule_based_place_type=cfg.rule_based_place_type,
                               fresh_w=cfg.fresh_w),
                    k, reference=self._reference, explicit_filters=filters)
                trace["post"] = post
            else:
                top_k = base[:k]
        trace["seconds"] = round(time.monotonic() - t0, 3)
        return self._rows(top_k), trace


_default: dict[str, Retriever] = {}


def retrieve_with_trace(query: str, filters: dict | None = None, k: int = 10,
                        config: RetrieveConfig = FROZEN_CONFIG, **parts) -> tuple[list[dict], dict]:
    """(results, trace). With the frozen config the real index is built on first use; other
    configs need their parser, scorer and rewriter passed in `parts` (index=, parser=, ...)."""
    if parts or config is not FROZEN_CONFIG:
        return Retriever(config, **parts).search(query, filters, k)
    if config.name not in _default:
        _default[config.name] = Retriever(config)
    return _default[config.name].search(query, filters, k)


def retrieve(query: str, filters: dict | None = None, k: int = 10) -> list[dict]:
    """The project contract (docs/contracts.md): chunk fields plus "score", frozen config."""
    return retrieve_with_trace(query, filters, k)[0]


def config_dict(config: RetrieveConfig = FROZEN_CONFIG) -> dict:
    return asdict(config)
