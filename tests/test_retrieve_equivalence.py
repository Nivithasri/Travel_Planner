"""The engine (src/retrieval/retrieve.py) reproduces the ablation variants on the real index (D96).

Needs the gitignored index/, data/enriched and the local bge-m3 model (EMBED_LOCAL_DIR), so each
test is skipped on a fresh clone. Only rankings are compared (scripts/verify_retrieve.py); no
relevance judgment is read, so the test split is safe to check. Run with the .venv interpreter
(bm25s is installed there).
"""
import os

import pytest

from src.common.schemas import REPO_ROOT, load_embed_local_dir, load_rerank_local_dir

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_embed = load_embed_local_dir()
_ready = (REPO_ROOT / "index" / "ids.json").exists() and (
    REPO_ROOT / "data" / "enriched" / "provenance.jsonl").exists() and bool(
    _embed and os.path.isdir(_embed))
needs_index = pytest.mark.skipif(not _ready, reason="index/, data/enriched or EMBED_LOCAL_DIR "
                                 "missing (gitignored, O12)")
_rr = load_rerank_local_dir()
needs_reranker = pytest.mark.skipif(not (_ready and _rr and os.path.isdir(_rr)),
                                    reason="RERANK_LOCAL_DIR missing")


@needs_index
def test_the_frozen_config_reproduces_its_ablation_row_on_the_test_queries():
    from scripts.verify_retrieve import verify
    assert verify("test", out=lambda *a: None) == 0


@needs_index
def test_non_reranked_rows_are_reproduced_on_the_dev_queries():
    from scripts.verify_retrieve import verify
    names = ("dense_contextual", "hybrid_contextual", "expanded_hybrid", "hybrid_filters",
             "row5_filters", "row6_fresh_w20")
    assert verify("dev", names, with_scorer=False, out=lambda *a: None) == 0


@needs_reranker
def test_reranked_rows_are_reproduced_on_the_dev_queries():
    from scripts.verify_retrieve import verify
    names = ("hybrid_rerank", "row7_rerank", "row8_mmr", "row9_crag")
    assert verify("dev", names, out=lambda *a: None) == 0
