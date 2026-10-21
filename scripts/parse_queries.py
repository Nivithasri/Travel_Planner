"""Run query understanding (qu-v1, D86) on the eval queries and write eval/parsed_queries.json.

    python scripts/parse_queries.py                 # all 40 queries (NETWORK: one LLM call each)
    python scripts/parse_queries.py --ids Q01,Q03   # a subset, to smoke-test
    python scripts/parse_queries.py --offline       # no call: cache hits only; a miss is an error

Only the QUERY TEXT is sent (owner point 7). The model is GENERATOR_MODEL; the endpoint and key
come from .env through OpenAICompatClient.from_settings and are never printed. The daily quota
counters go to .cache/llm_quota/quota_state.json (D93), not data/enriched. Every validated reply is
cached on disk (rule 4), so a rerun costs nothing. The output has no corpus text.

Printed: the call/cache accounting, the drop notes by count, and the time_sensitive confusion
table against eval/queries.json for the DEV split only. The TEST split's flags are not compared
here (they are checked once, at the end of the phase); its parses are written but not shown.
Exit codes: 0 ok, 2 bad setting, 3 LLM/transport error, 5 quota used up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUOTA_STATE = Path(__file__).resolve().parents[1] / ".cache" / "llm_quota" / "quota_state.json"
os.environ.setdefault("LLM_QUOTA_STATE", str(QUOTA_STATE))  # D93: never data/enriched

from eval.qrels import load_queries  # noqa: E402
from src.common.cache import atomic_write_text  # noqa: E402
from src.common.llm import LLMError, OpenAICompatClient, QuotaExhausted  # noqa: E402
from src.common.schemas import REPO_ROOT, ConfigError, load_llm_settings  # noqa: E402
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.retrieval import query_understanding as qu  # noqa: E402

OUT = REPO_ROOT / "eval" / "parsed_queries.json"


def flag_table(queries: list[dict], parsed: dict[str, dict]) -> dict:
    """Confusion counts of the predicted time_sensitive against the labelled one."""
    c: Counter = Counter()
    wrong = []
    for q in queries:
        pred = parsed[q["id"]]["time_sensitive"]
        gold = q["time_sensitive"]
        c[("tp" if gold else "fp") if pred else ("fn" if gold else "tn")] += 1
        if pred != gold:
            wrong.append(q["id"])
    n = sum(c.values())
    return {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"], "n": n,
            "accuracy": (c["tp"] + c["tn"]) / n if n else None, "wrong": wrong}


def note_counts(parsed: dict[str, dict]) -> Counter:
    kinds: Counter = Counter()
    for p in parsed.values():
        for note in p["notes"]:
            kinds[note.split(" dropped")[0].split(" replaced")[0].split(":")[0][:40]] += 1
    return kinds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ids", help="comma-separated query ids (default: all 40)")
    ap.add_argument("--offline", action="store_true", help="cache hits only; never call the API")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    queries = load_queries()
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in wanted]
        if not queries:
            print("no such query ids")
            return 2
    try:
        settings = load_llm_settings()
    except ConfigError as exc:
        print(f"settings: {exc}")
        return 2
    registry = qu.load_core_registry()
    holdout = HoldoutFilter.from_registry(registry)
    client = OpenAICompatClient.from_settings()
    if args.offline:
        def no_network(*a, **k):
            raise RuntimeError("--offline: a cache miss would call the API")
        client._request = no_network  # a cache hit never reaches _request

    parsed: dict[str, dict] = {}
    try:
        for q in queries:
            parsed[q["id"]] = qu.parse_query(q["text"], client, settings.generator_model,
                                             registry, holdout).to_dict()
            print(f"{q['id']} parsed ({len(parsed[q['id']]['sub_queries'])} sub-queries)")
    except QuotaExhausted as exc:
        print(f"QUOTA: {exc}; the {len(parsed)} parsed so far are cached. Rerun to resume.")
        return 5
    except (LLMError, RuntimeError) as exc:
        print(f"LLM error after {len(parsed)} queries: {type(exc).__name__}: {str(exc)[:200]}")
        return 3

    meta = {"prompt_version": qu.PROMPT_VERSION, "model": settings.generator_model,
            "options": qu.QU_OPTIONS, "n_queries": len(parsed),
            "registry_places_listed": len(qu.known_places(registry))}
    atomic_write_text(args.out, json.dumps({"meta": meta, "queries": parsed}, indent=1,
                                           sort_keys=True, ensure_ascii=False) + "\n")
    shown = args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
    print(f"\nwrote {shown} ({len(parsed)} queries)")
    snap = client.limiter.snapshot() if client.limiter else "-"
    print(f"LLM usage this run: {dict(client.usage)}; limiter: {snap}")
    print(f"fallbacks (< {qu.MIN_SUBQUERIES} sub-queries): "
          f"{sum(p['fallback'] for p in parsed.values())} of {len(parsed)}")
    print("drop/normalise notes by kind:", dict(note_counts(parsed)))
    dev = [q for q in queries if q["split"] == "dev"]
    if dev:
        t = flag_table(dev, parsed)
        print(f"\ntime_sensitive vs eval/queries.json, DEV only ({t['n']} queries): "
              f"TP {t['tp']} FP {t['fp']} FN {t['fn']} TN {t['tn']}, accuracy "
              f"{t['accuracy']:.3f}; wrong: {t['wrong'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
