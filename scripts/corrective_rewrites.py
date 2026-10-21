"""The corrective-retrieval rewrites (row 9, D90): what would be needed, and then (only on request)
the LLM calls that produce them.

    python scripts/corrective_rewrites.py                 # DRY RUN: no LLM call, no network
    python scripts/corrective_rewrites.py --execute       # NETWORK: one LLM call per needed rewrite

Dry run: runs the row 8 first pass on the dev queries (reranker scores come from the disk cache),
prints each query's best reranker score (as a sigmoid), its distinct URLs in the top 5 and the
contradiction flag (no metrics), the three tau candidates (the 20th, 40th and 60th percentile of the
dev best-scores, so roughly 3, 5 and 8 of 13 queries are low-confidence), the queries that trigger
under each, and the rewrites that would be generated with an estimate of the tokens. `--execute`
needs the owner's approval. The prompt holds ONLY the query text and the trigger names (owner point
7). Results go to eval/corrective_rewrites.json, so row 9 runs without a network. The quota counters
go to .cache/llm_quota (D93). Exit codes: 0 ok, 2 bad setting, 3 LLM error, 5 quota used up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LLM_QUOTA_STATE", str(Path(__file__).resolve().parents[1] / ".cache"
                                             / "llm_quota" / "quota_state.json"))

import numpy as np  # noqa: E402

from eval import variants as V  # noqa: E402
from eval.qrels import load_queries  # noqa: E402
from src.common.cache import atomic_write_text  # noqa: E402
from src.common.llm import LLMError, OpenAICompatClient, QuotaExhausted  # noqa: E402
from src.common.schemas import ConfigError, load_llm_settings  # noqa: E402
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.retrieval import grader  # noqa: E402
from src.retrieval import query_understanding as qu  # noqa: E402

PERCENTILES = (20, 40, 60)  # the tau candidates, fixed before any metric is computed (D90)
EST_TOKENS_PER_CALL = 900  # conservative: ~500 prompt + the 400-token reply cap


def first_pass(queries: list[dict], scorer, lam: float) -> dict[str, dict]:
    index = V.get_index()
    out = {}
    for q in queries:
        final, logits, _ = V._pipeline(q["text"], V._parse_of(q["text"]), scorer, lam)
        top = [index.chunks[cid] for cid, _ in final[:grader.TOP_N]]
        out[q["text"]] = {"id": q["id"], "confidence": grader.top_confidence(logits), "top": top}
    return out


def tau_candidates(confidences: list[float]) -> list[float]:
    return [round(float(np.percentile(confidences, p)), 4) for p in PERCENTILES]


def taus_for(split: str, confidences: list[float], chosen: float | None) -> list[float]:
    """The tau values whose rewrites are needed. DEV: the three percentile candidates of the dev
    scores (tau is tuned on dev only, rule 3). TEST: only the tau already chosen on dev; the test
    scores never choose or move a threshold."""
    if split == "dev":
        return tau_candidates(confidences)
    if chosen is None:
        raise ValueError("the dev-chosen tau (eval/corrective_tau.json) is needed for test")
    return [chosen]


def needed(signals: dict[str, dict], taus: list[float]) -> dict[str, tuple[str, list[str]]]:
    """{rewrite key: (query, triggers)} over every tau candidate."""
    out = {}
    for text, s in signals.items():
        for tau in taus:
            trig = grader.fired(s["confidence"], s["top"], tau)
            if trig:
                out[grader.rewrite_key(text, trig)] = (text, trig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--execute", action="store_true", help="make the LLM calls (needs approval)")
    ap.add_argument("--out", type=Path, default=V.REWRITES_FILE)
    ap.add_argument("--split", choices=("dev", "test"), default="dev",
                    help="test: rewrites for the queries that trigger at the dev-chosen tau only")
    args = ap.parse_args(argv)

    queries = load_queries(args.split)
    lam = V.chosen_lambda()
    scorer = V.get_ce_scorer()
    signals = first_pass(queries, scorer, lam)
    confs = [s["confidence"] for s in signals.values()]
    chosen = V.chosen_tau() if args.split == "test" else None
    taus = taus_for(args.split, confs, chosen)

    print(f"{args.split} first pass (row 8, lambda {lam}): best reranker score as a sigmoid, "
          "distinct URLs in the top 5, contradiction flag")
    for text, s in sorted(signals.items(), key=lambda t: t[1]["id"]):
        urls = grader.distinct_sources(s["top"])
        print(f"  {s['id']}  conf {s['confidence']:.4f}  urls {urls}  "
              f"contradiction {grader.contradiction(s['top'])}")
    how = (f"candidates (percentiles {PERCENTILES} of the dev scores)" if args.split == "dev"
           else "fixed on dev")
    print(f"\ntau {how}: {taus}")
    for tau in taus:
        rows = {s["id"]: grader.fired(s["confidence"], s["top"], tau) for s in signals.values()}
        hit = {i: t for i, t in rows.items() if t}
        print(f"  tau {tau}: {len(hit)} of {len(rows)} queries trigger: "
              + "; ".join(f"{i} {'+'.join(t)}" for i, t in sorted(hit.items())))
    todo = needed(signals, taus)
    store = grader.load_rewrites(args.out)
    missing = {k: v for k, v in todo.items() if k not in store}
    print(f"\nrewrites needed: {len(todo)} ({len(todo) - len(missing)} already stored, "
          f"{len(missing)} to generate); estimated tokens {len(missing) * EST_TOKENS_PER_CALL:,} "
          "(limit 200,000 a day)")
    if not args.execute:
        print("DRY RUN: no LLM call was made. Re-run with --execute after approval.")
        return 0
    if not missing:
        print("nothing to generate")
        return 0

    try:
        settings = load_llm_settings()
    except ConfigError as exc:
        print(f"settings: {exc}")
        return 2
    holdout = HoldoutFilter.from_registry(qu.load_core_registry())
    client = OpenAICompatClient.from_settings()
    try:
        for key, (text, trig) in missing.items():
            rewrite, why = grader.generate_rewrite(client, settings.generator_model, text, trig,
                                                   holdout)
            store[key] = {"query": text, "triggers": trig, "rewrite": rewrite, "dropped": why}
            print(f"  {key} {'+'.join(trig)}: {'ok' if rewrite else 'unusable (' + str(why) + ')'}")
    except QuotaExhausted as exc:
        print(f"QUOTA: {exc}; the rewrites so far are cached. Rerun to resume.")
        return 5
    except LLMError as exc:
        print(f"LLM error: {type(exc).__name__}: {str(exc)[:200]}")
        return 3
    finally:
        meta = {"prompt": "crag-rewrite-v1", "model": settings.generator_model,
                "options": grader.REWRITE_OPTIONS}
        atomic_write_text(args.out, json.dumps({"meta": meta, "rewrites": store}, indent=1,
                                               sort_keys=True, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out.name} ({len(store)} rewrites); usage {dict(client.usage)}; "
          f"limiter {client.limiter.snapshot() if client.limiter else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
