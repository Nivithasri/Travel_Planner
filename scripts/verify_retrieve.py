"""Does retrieve() (src/retrieval/retrieve.py) reproduce the ablated variants exactly? (D96)

    python scripts/verify_retrieve.py          # DEV queries only; offline; exit 1 on any mismatch

For every ablation row that has a RetrieveConfig, runs the engine and the original variant
(eval/variants.py) on the 13 dev queries with the same index, parse file, reranker cache and stored
rewrites, and compares the top 10 chunk ids IN ORDER. Because the engine reproduces them, the dev
numbers measured for the variants are the numbers of the engine, and the frozen config is exactly
row 3. No relevance judgments are read; the test queries are not touched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import variants as V  # noqa: E402
from eval.qrels import load_queries  # noqa: E402
from src.common.schemas import load_reference_date  # noqa: E402
from src.retrieval import retrieve as R  # noqa: E402

C = R.RetrieveConfig
PAIRS = [
    ("hybrid_contextual", V.hybrid_contextual, R.FROZEN_CONFIG),
    ("expanded_hybrid", V.expanded_hybrid, C(name="e", expansion=True)),
    ("hybrid_filters", V.hybrid_filters, C(name="hf", use_filters=True)),
    ("row5_filters", V.row5_filters, C(name="f", expansion=True, use_filters=True)),
    ("row6_fresh_w20", V.row6_fresh_w20,
     C(name="w", expansion=True, use_filters=True, fresh_w=0.2)),
    ("hybrid_rerank", V.hybrid_rerank, C(name="hr", rerank=True)),
    ("row7_rerank", V.row7_rerank, C(name="r", expansion=True, rerank=True)),
    ("row8_mmr", V.row8_mmr, C(name="m", expansion=True, rerank=True, mmr_lambda=R.MMR_LAMBDA)),
    ("row9_crag", V.row9_crag, R.FULL_CONFIG),
]


def main() -> int:
    queries = load_queries("dev")
    index, scorer = V.get_index(), V.get_ce_scorer()
    parts = dict(index=index, provenance=V.get_provenance(), parser=R.parser_from_file(),
                 scorer=scorer, rewriter=R.rewriter_from_file(), reference=load_reference_date())
    bad = 0
    print(f"{'ablation row':<20}{'config':<26}{'same top-10 (ids, order)':>26}")
    for name, variant, cfg in PAIRS:
        engine = R.Retriever(cfg, **parts)
        same = 0
        for q in queries:
            want = [r["chunk_id"] for r in variant(q["text"], None, 10)]
            got = [r["chunk_id"] for r in engine.search(q["text"], None, 10)[0]]
            same += want == got
            if want != got:
                print(f"    MISMATCH {name} {q['id']}: variant {want[:4]}.. engine {got[:4]}..")
        bad += same != len(queries)
        print(f"{name:<20}{cfg.name:<26}{same:>20} of {len(queries)}")
    frozen = R.Retriever(R.FROZEN_CONFIG, **parts)
    q0 = queries[0]["text"]
    a = [(r["chunk_id"], round(r["score"], 12)) for r in frozen.search(q0, None, 10)[0]]
    b = [(r["chunk_id"], round(r["score"], 12)) for r in V.hybrid_contextual(q0, None, 10)]
    print(f"\nfrozen config scores identical to hybrid_contextual on {queries[0]['id']}: {a == b}")
    bad += a != b
    print("VERIFIED: the engine reproduces every ablated row" if not bad
          else f"FAILED: {bad} row(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
