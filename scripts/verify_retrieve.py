"""Does retrieve() (src/retrieval/retrieve.py) reproduce the ablated variants exactly? (D96)

    python scripts/verify_retrieve.py              # DEV queries, every pairing; exit 1 on mismatch
    python scripts/verify_retrieve.py --split test # TEST queries, the frozen pairing only

For every ablation row that has a RetrieveConfig, runs the engine and the original variant
(eval/variants.py, eval/adapters.py) on the queries of the split with the same index, parse file,
reranker cache and stored rewrites, and compares the top 10 chunk ids IN ORDER. Because the engine
reproduces them, the numbers measured for the variants are the numbers of the engine, and the frozen
config is exactly its ablation row. Only rankings are compared: no relevance judgments are read, so
the test split can be checked without looking at any test result.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import adapters  # noqa: E402
from eval import variants as V  # noqa: E402
from eval.qrels import load_queries  # noqa: E402
from src.common.schemas import load_reference_date  # noqa: E402
from src.retrieval import retrieve as R  # noqa: E402

C = R.RetrieveConfig
FROZEN_PAIR = ("dense_contextual", adapters.dense_contextual, R.FROZEN_CONFIG)
PAIRS = [
    FROZEN_PAIR,
    ("hybrid_contextual", V.hybrid_contextual, R.HYBRID_CONFIG),
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


def verify(split: str, names: tuple[str, ...] | None = None, *, with_scorer: bool = True,
           out=print) -> int:
    """Number of failing checks (0 = the engine reproduces every checked row). `names` limits the
    pairings (dev only); with_scorer=False skips the cross-encoder, so reranked rows cannot run."""
    queries = load_queries(split)
    pairs = PAIRS if split == "dev" else [FROZEN_PAIR]
    if names is not None:
        pairs = [p for p in pairs if p[0] in names]
    index = V.get_index()
    parts = dict(index=index, provenance=V.get_provenance(), parser=R.parser_from_file(),
                 scorer=V.get_ce_scorer() if with_scorer and split == "dev" else None,
                 rewriter=R.rewriter_from_file(), reference=load_reference_date())
    bad = 0
    out(f"split {split}, {len(queries)} queries")
    out(f"{'ablation row':<20}{'config':<26}{'same top-10 (ids, order)':>26}")
    for name, variant, cfg in pairs:
        engine = R.Retriever(cfg, **parts)
        same = 0
        for q in queries:
            want = [r["chunk_id"] for r in variant(q["text"], None, 10)]
            got = [r["chunk_id"] for r in engine.search(q["text"], None, 10)[0]]
            same += want == got
            if want != got:
                out(f"    MISMATCH {name} {q['id']}: variant {want[:4]}.. engine {got[:4]}..")
        bad += same != len(queries)
        out(f"{name:<20}{cfg.name:<26}{same:>20} of {len(queries)}")
    q0 = queries[0]["text"]
    for label, cfg, fn in (("frozen", R.FROZEN_CONFIG, adapters.dense_contextual),
                           ("hybrid", R.HYBRID_CONFIG, V.hybrid_contextual)):
        if (split == "test" and label == "hybrid") or (
                names is not None and "hybrid_contextual" not in names and label == "hybrid"):
            continue
        eng = R.Retriever(cfg, **parts)
        a = [(r["chunk_id"], round(r["score"], 12)) for r in eng.search(q0, None, 10)[0]]
        b = [(r["chunk_id"], round(r["score"], 12)) for r in fn(q0, None, 10)]
        out(f"\n{label} config: ids AND scores identical to its ablation row on "
            f"{queries[0]['id']}: {a == b}")
        bad += a != b
    out("VERIFIED: the engine reproduces every checked row" if not bad
        else f"FAILED: {bad} row(s) differ")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    args = ap.parse_args(argv)
    return 1 if verify(args.split) else 0


if __name__ == "__main__":
    sys.exit(main())
