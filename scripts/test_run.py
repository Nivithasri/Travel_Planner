"""THE ONE-SHOT TEST RUN of Phase 6 (CLAUDE.md rule 3, D84, D92, D96 as amended).

    python scripts/test_run.py   # runs once; refuses if eval/ablation/test_run_manifest.json exists

PROTOCOL, fixed and committed BEFORE the run:
* Every ablation row is run ONCE on the 27 test queries against the merged judgments
  (eval.qrels.load_qrels_merged: round 1 human + round 2 chat, D95). Nothing is tuned and nothing
  is chosen from the result: the frozen pipeline is dense_contextual whatever the test shows (D96
  as amended). The dev merged numbers are recomputed alongside only to show whether the ordering
  holds; they decide nothing.
* Paired two-sided sign-flip test on per-query nDCG@10 (100,000 seeded Monte Carlo flips, n = 27),
  Holm-Bonferroni over the FAMILY below, which is fixed here: queries with no relevant judged
  chunk are dropped pairwise.
* Also printed: the predicted time_sensitive flag against eval/queries.json for the 27 test
  queries, and a check that the frozen engine returns the same top 10 as its ablation row.
* The working tree must be clean (tracked files), so the run belongs to a commit, and nothing is
  printed or written until every row has been scored. The manifest is written LAST; while it
  exists a second run is refused (delete it only deliberately, and say so in the log).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import ablation as A  # noqa: E402
from eval import adapters  # noqa: E402
from eval import significance as S  # noqa: E402
from eval.qrels import load_qrels_merged, load_queries  # noqa: E402
from eval.run_eval import (  # noqa: E402
    ENRICHED,
    PROVENANCE,
    evaluate,
    load_chunk_info,
    load_excluded,
    load_retriever,
)
from scripts.parse_queries import flag_table  # noqa: E402
from src.common.cache import atomic_write_text  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date  # noqa: E402
from src.retrieval import retrieve as R  # noqa: E402

OUT = REPO_ROOT / "eval" / "ablation"
MANIFEST = OUT / "test_run_manifest.json"
N_MC, SEED = 100_000, 42
FROZEN_ROW = "dense_contextual"

ROWS = [
    ("bm25_plain", "eval.adapters:bm25_plain"),
    ("dense_plain", "eval.adapters:dense_plain"),
    ("bm25_contextual", "eval.adapters:bm25_contextual"),
    ("dense_contextual", "eval.adapters:dense_contextual"),
    ("hybrid_plain", "eval.variants:hybrid_plain"),
    ("hybrid_contextual", "eval.variants:hybrid_contextual"),
    ("rrf4", "eval.adapters:rrf4"),
    ("dense_control", "eval.variants:dense_control"),
    ("bm25_control", "eval.variants:bm25_control"),
    ("hybrid_control", "eval.variants:hybrid_control"),
    ("expanded_hybrid", "eval.variants:expanded_hybrid"),
    ("row5_filters", "eval.variants:row5_filters"),
    ("row6_fresh_w10", "eval.variants:row6_fresh_w10"),
    ("row6_fresh_w20", "eval.variants:row6_fresh_w20"),
    ("row6_fresh_w30", "eval.variants:row6_fresh_w30"),
    ("hybrid_filters", "eval.variants:hybrid_filters"),
    ("row7_rerank", "eval.variants:row7_rerank"),
    ("hybrid_rerank", "eval.variants:hybrid_rerank"),
    ("row8_mmr", "eval.variants:row8_mmr"),
    ("row9_crag", "eval.variants:row9_crag"),
]
# The significance family (11 paired comparisons, first -> second), fixed before the run.
FAMILY = [
    ("bm25_plain", "dense_plain"),                  # the two baselines
    ("dense_plain", "dense_contextual"),            # contextual notes on dense
    ("hybrid_plain", "hybrid_contextual"),          # contextual notes on the hybrid
    ("dense_plain", "hybrid_plain"),                # the hybrid step on plain text
    ("dense_contextual", "hybrid_contextual"),      # the hybrid step on contextual text
    ("bm25_contextual", "dense_contextual"),        # sparse vs dense
    ("dense_contextual", "row9_crag"),              # the frozen row vs the full chain
    ("hybrid_contextual", "expanded_hybrid"),       # row 4
    ("expanded_hybrid", "row7_rerank"),             # row 7
    ("row7_rerank", "row8_mmr"),                    # row 8
    ("row8_mmr", "row9_crag"),                      # row 9
]


def ensure_not_yet_run(manifest: Path = MANIFEST) -> None:
    if manifest.exists():
        raise SystemExit(f"{manifest.name} exists: the one-shot test run was already done. "
                         "Delete it only deliberately, and record why in LOG.md.")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def ensure_clean_tree() -> str:
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT).returncode != 0:
        raise SystemExit("tracked files have uncommitted changes: commit them first, so the "
                         "run belongs to a commit")
    return git("rev-parse", "HEAD")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memo(fn):
    cache: dict = {}

    def wrapped(query, filters=None, k=10):
        if (query, k) not in cache:
            cache[(query, k)] = fn(query, filters, k)
        return cache[(query, k)]

    return wrapped


def comparisons(results: dict) -> list[dict]:
    comps = [S.compare(f"{a} -> {b}", A.ndcg_by_query(results[a]), A.ndcg_by_query(results[b]),
                       n_mc=N_MC, seed=SEED) for a, b in FAMILY]
    out = []
    for comp, holm_p in S.holm_over(comps):
        out.append({"comparison": comp.name, "n": comp.n, "delta": comp.mean_diff,
                    "ci95": [comp.ci_low, comp.ci_high], "p": comp.p, "p_holm": holm_p})
    return out


def f3(v) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ensure_not_yet_run()
    head = ensure_clean_tree()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test_q, dev_q = load_queries("test"), load_queries("dev")
    qrels_test = load_qrels_merged("test", allow_test=True)  # the only place test qrels are read
    qrels_dev = load_qrels_merged("dev")
    info, excluded, ref = load_chunk_info(ENRICHED, PROVENANCE), load_excluded(), \
        load_reference_date()

    test_res: dict[str, dict] = {}
    dev_ndcg: dict[str, float | None] = {}
    for name, spec in ROWS:
        retr = memo(load_retriever(spec))
        test_res[name] = evaluate(retr, test_q, qrels_test, info, ref, excluded)
        dev_ndcg[name] = evaluate(retr, dev_q, qrels_dev, info, ref, excluded)[
            "groups"]["overall"]["ndcg@10"]
        print(f"scored {name}", flush=True)

    comps = comparisons(test_res)
    parsed = json.loads((REPO_ROOT / "eval" / "parsed_queries.json").read_text(
        encoding="utf-8"))["queries"]
    flags = flag_table(test_q, parsed)
    engine = R.Retriever(R.FROZEN_CONFIG)
    same = sum([r["chunk_id"] for r in engine.search(q["text"], None, 10)[0]]
               == [r["chunk_id"] for r in adapters.dense_contextual(q["text"], None, 10)]
               for q in test_q)

    # ---- everything is scored; now print and write
    lines = ["# Phase 6 TEST run (27 queries, one run; round 1 human + round 2 chat judgments)", "",
             f"commit {head}; frozen pipeline: {FROZEN_ROW}. Descriptive: 27 queries detect "
             "differences of about 0.1 nDCG only.", "",
             "| row | nDCG@10 test | dev (merged) | named | free | MRR | P@5 | R@10 | "
             "unjudged@10 | excluded@10 | places@10 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for name, _ in ROWS:
        g = test_res[name]["groups"]
        o = g["overall"]
        table[name] = {"ndcg": o["ndcg@10"], "dev_ndcg": dev_ndcg[name],
                       "named": g["place_scope=named"]["ndcg@10"],
                       "free": g["place_scope=free"]["ndcg@10"], "mrr": o["mrr"], "p5": o["p@5"],
                       "r10": o["recall@10"], "unjudged10": o["unjudged@10"],
                       "excluded10": o["excluded@10"], "places10": o["places@10"],
                       "n_zero_relevant": o["n_zero_relevant"]}
        t = table[name]
        mark = " (FROZEN)" if name == FROZEN_ROW else ""
        lines.append(f"| {name}{mark} | {f3(t['ndcg'])} | {f3(t['dev_ndcg'])} | {f3(t['named'])} "
                     f"| {f3(t['free'])} | {f3(t['mrr'])} | {f3(t['p5'])} | {f3(t['r10'])} | "
                     f"{f3(t['unjudged10'])} | {f3(t['excluded10'])} | {f3(t['places10'])} |")
    lines += ["", "## Pre-registered paired comparisons (nDCG@10; sign-flip, 100,000 flips; Holm "
              "over these 11)", "", "| comparison | n | delta | 95% CI | p | p (Holm) |",
              "|---|---|---|---|---|---|"]
    for c in comps:
        lines.append(f"| {c['comparison']} | {c['n']} | {c['delta']:+.3f} | "
                     f"[{f3(c['ci95'][0])}, {f3(c['ci95'][1])}] | {c['p']:.4f} | "
                     f"{c['p_holm']:.4f} |")
    lines += ["", f"## time_sensitive flag, predicted vs eval/queries.json, TEST ({flags['n']} "
              f"queries)", "",
              f"TP {flags['tp']} FP {flags['fp']} FN {flags['fn']} TN {flags['tn']}, "
              f"accuracy {f3(flags['accuracy'])}; wrong: {flags['wrong'] or 'none'}",
              "", f"frozen engine vs its ablation row (top 10 ids, in order): {same} of "
              f"{len(test_q)} queries identical"]
    md = "\n".join(lines)
    print("\n" + md)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, _ in ROWS:
        atomic_write_text(OUT / f"test_merged_{name}.json",
                          json.dumps(test_res[name], indent=1, sort_keys=True))
    atomic_write_text(OUT / "test_final.md", md + "\n")
    atomic_write_text(OUT / "test_final.json", json.dumps(
        {"table": table, "comparisons": comps, "flags": flags,
         "engine_same_top10": same, "n_queries": len(test_q)}, indent=1, sort_keys=True))
    inputs = {"qrels_test_round1_sha256": sha256_file(REPO_ROOT / "eval" / "qrels_test.json"),
              "qrels_test_r2_sha256": sha256_file(REPO_ROOT / "eval" / "qrels_test_r2.json"),
              "frozen_config_sha256": sha256_file(REPO_ROOT / "eval" / "frozen_config.json"),
              "parsed_queries_sha256": sha256_file(REPO_ROOT / "eval" / "parsed_queries.json"),
              "queries_sha256": sha256_file(REPO_ROOT / "eval" / "queries.json")}
    atomic_write_text(MANIFEST, json.dumps(
        {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"), "git_head": head, "rows": [n for n, _ in ROWS],
         "family": FAMILY, "n_mc": N_MC, "seed": SEED, "frozen_row": FROZEN_ROW,
         "inputs": inputs}, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote eval/ablation/test_final.md, test_final.json, test_merged_<row>.json and "
          f"{MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
