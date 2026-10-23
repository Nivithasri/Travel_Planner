"""Run the Phase 7 pipeline end to end from the command line.

    .venv/Scripts/python.exe scripts/run_pipeline.py --query "5 days of temples and food" \
        --days 5 --month 1 --start Chennai --budget mid --party couple

Needs index/place_cards.json (scripts/build_place_cards.py), the local bge-m3 model
(EMBED_LOCAL_DIR) and, unless --no-llm-parse, the hosted LLM key. Without a verified live-search
setting (config/live.json) the live step is skipped with a visible note. The answer is printed;
--out saves it (itineraries hold excerpt-derived text: keep them out of git, O12). A timing log
with one line per node goes to .cache/runs/ (no request text is stored, only its hash).

Exit codes: 0 ok, 2 bad input, 3 the critic reported failed_rules, 5 infeasible plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RUNS = ROOT / ".cache" / "runs"


def split(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def overrides_from(args) -> dict:
    o: dict = {"no_llm": args.no_llm_parse}
    for key, val in (("duration_days", args.days), ("budget_tier", args.budget),
                     ("month", args.month), ("start_city", args.start), ("end_city", args.end),
                     ("party", args.party), ("budget_cap_inr", args.cap_inr),
                     ("max_daily_travel_hours", args.max_daily_hours)):
        if val is not None:
            o[key] = val
    if args.interests:
        o["interests"] = split(args.interests)
    if args.constraints:
        o["constraints"] = split(args.constraints)
    if args.named:
        o["named_places"] = split(args.named)
    if args.states:
        o["states"] = [s.upper() for s in split(args.states)]
    if args.time_sensitive:
        o["time_sensitive"] = True
    return o


def write_log(state, text: str, runs: Path | None = None) -> Path:
    runs = runs or RUNS
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs / f"run_{stamp}.jsonl"
    rid = hashlib.sha256(f"{stamp}{text}".encode()).hexdigest()[:10]
    with path.open("w", encoding="utf-8") as f:
        for node, secs in state.timings.items():
            f.write(json.dumps({"run": rid, "node": node, "seconds": secs}) + "\n")
        f.write(json.dumps({
            "run": rid, "summary": True, "request_sha256": hashlib.sha256(text.encode()
                                                                          ).hexdigest()[:16],
            "total_node_seconds": round(sum(state.timings.values()), 3),
            "live_status": state.live_status, "critic": state.critic.status if state.critic
            else None, "tokens": state.tokens, "n_warnings": len(state.warnings)}) + "\n")
    return path


def main(argv: list[str] | None = None, deps_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--query", required=True, help="the request in plain words")
    ap.add_argument("--days", type=int)
    ap.add_argument("--budget", choices=("budget", "mid", "luxury"))
    ap.add_argument("--month", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--party", choices=("solo", "couple", "family", "group"))
    ap.add_argument("--interests", help="comma separated")
    ap.add_argument("--constraints", help="comma separated, e.g. 'avoid hill roads'")
    ap.add_argument("--named", help="comma separated registry places the user asks for")
    ap.add_argument("--states", help="comma separated state codes (TN,KL,KA,AP,TS,PY)")
    ap.add_argument("--cap-inr", type=int)
    ap.add_argument("--max-daily-hours", type=float)
    ap.add_argument("--time-sensitive", action="store_true")
    ap.add_argument("--no-llm-parse", action="store_true",
                    help="skip the query-understanding call; use the flags and registry names")
    ap.add_argument("--no-live", action="store_true", help="do not use the live layer")
    ap.add_argument("--local", action="store_true",
                    help="use the sequential fallback, not LangGraph")
    ap.add_argument("--stream", action="store_true", help="print the draft while it is written")
    ap.add_argument("--out", type=Path, help="save the final text here")
    args = ap.parse_args(argv)

    from src.agents.graph import run_graph, run_local

    pieces = (lambda p: print(p, end="", flush=True)) if args.stream else None
    if deps_factory is None:
        from src.agents.deps import build_default_deps

        deps_factory = build_default_deps
    deps = deps_factory(live=not args.no_live, parse=not args.no_llm_parse, on_piece=pieces)
    run = run_local if args.local else run_graph
    t0 = time.monotonic()
    try:
        state = run(deps, args.query, overrides_from(args))
    except ValueError as exc:  # pydantic validation errors are ValueErrors too
        print(f"bad input: {exc}", file=sys.stderr)
        return 2
    if args.stream:
        print("\n" + "-" * 40)
    print(state.final["text"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(state.final["text"], encoding="utf-8")
    log = write_log(state, args.query)
    print("Timing per node (seconds): " + ", ".join(f"{k} {v}" for k, v in state.timings.items())
          + f"; wall {time.monotonic() - t0:.1f}")
    print(f"live: {state.live_status}; critic: {state.critic.status if state.critic else None}; "
          f"tokens: {state.tokens}; log: {log}")
    if not state.plan.feasible:
        return 5
    return 3 if state.critic and state.critic.status == "failed_rules" else 0


if __name__ == "__main__":
    sys.exit(main())
