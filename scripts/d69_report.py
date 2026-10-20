"""D69 report: plain vs contextual vs control, on dev (D69, owner point 4).

    python scripts/d69_report.py            # dev; writes eval/ablation/d69_dev.{md,json}

For dense and BM25 it reports nDCG@10 (a) overall, (b) for place-named and place-free queries,
and (c) model-written rows only (the retriever keeps provenance=llm rows and the comparison is
against qrels restricted to llm chunks, i.e. the same retriever on the model-written sub-corpus).
The control arm is "about <place>, <state>" + text (scripts/build_control_index.py), which
separates the place-name effect from the model-written note. Dev only; reads qrels_dev. No network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import adapters, variants  # noqa: E402
from eval.qrels import load_qrels, load_queries  # noqa: E402
from eval.run_eval import ENRICHED, PROVENANCE, evaluate, load_chunk_info, load_excluded  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402

OUT = REPO_ROOT / "eval" / "ablation"
ARMS = [
    ("dense", "plain", adapters.dense_plain), ("dense", "contextual", adapters.dense_contextual),
    ("dense", "control", variants.dense_control), ("bm25", "plain", adapters.bm25_plain),
    ("bm25", "contextual", adapters.bm25_contextual), ("bm25", "control", variants.bm25_control),
]


def llm_only(base, info):
    def wrapped(query, filters=None, k=10):
        deep = base(query, None, max(k * 10, 50))
        return [r for r in deep if info[r["chunk_id"]]["provenance"] == "llm"][:k]
    return wrapped


def ndcg(result, group):
    g = result["groups"].get(group)
    return g["ndcg@10"] if g else None


def fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    reference = load_reference_date()
    queries = load_queries("dev")
    qrels = load_qrels("dev")
    info = load_chunk_info(ENRICHED, PROVENANCE)
    excluded = load_excluded()
    qrels_llm = {qid: {c: g for c, g in q.items() if info[c]["provenance"] == "llm"}
                 for qid, q in qrels.items()}

    full: dict = {}
    mw: dict = {}
    for family, kind, fn in ARMS:
        res = evaluate(fn, queries, qrels, info, reference, excluded)
        full[(family, kind)] = {grp: ndcg(res, grp) for grp in
                                ("overall", "place_scope=named", "place_scope=free")}
        res_mw = evaluate(llm_only(fn, info), queries, qrels_llm, info, reference, excluded)
        mw[(family, kind)] = ndcg(res_mw, "overall")
        print(f"ran {family}_{kind}")

    lines = ["# D69: plain vs contextual vs control (dev, nDCG@10)", "",
             "| family | arm | overall | named | free | model-written only |",
             "|---|---|---|---|---|---|"]
    for family in ("dense", "bm25"):
        for kind in ("plain", "contextual", "control"):
            g = full[(family, kind)]
            lines.append(f"| {family} | {kind} | {fmt(g['overall'])} | "
                         f"{fmt(g['place_scope=named'])} | {fmt(g['place_scope=free'])} | "
                         f"{fmt(mw[(family, kind)])} |")
    for family in ("dense", "bm25"):
        p, c, ctl = (full[(family, k)]["overall"] for k in ("plain", "contextual", "control"))
        lines += ["", f"{family}: contextual - plain = {c - p:+.3f}; control - plain = "
                  f"{ctl - p:+.3f}; contextual - control = {c - ctl:+.3f} (the note beyond the "
                  "place name)"]
    lines += ["", "model-written only (c): the retriever keeps provenance=llm rows and is scored "
              "against qrels restricted to llm chunks, so it is the same arm on the model-written "
              "sub-corpus, not a like-for-like row (D69). 13 dev queries; descriptive (D92)."]
    md = "\n".join(lines)
    print("\n" + md)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d69_dev.md").write_text(md + "\n", encoding="utf-8")
    (OUT / "d69_dev.json").write_text(json.dumps(
        {"full": {f"{a}_{k}": v for (a, k), v in full.items()},
         "model_written_only": {f"{a}_{k}": v for (a, k), v in mw.items()}},
        indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
