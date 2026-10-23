"""One-off live smoke test of the Tavily layer (owner-approved, D108). ONE search per run, at most
3 in total (a counter file survives between runs), cache off, no page fetches. It calls the
lowest-level client directly, so the terms gate of config/live.json is NOT consulted and stays
'not_verified'. Raw replies go to .cache/live_smoke (gitignored) for offline parser work and are
deleted afterwards. Never prints the key, headers or the host.

  .venv/Scripts/python.exe scripts/live_smoke.py --index 1|2|3 [--yes]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.run import make_http  # noqa: E402
from src.common.http import FetchError  # noqa: E402
from src.common.schemas import REPO_ROOT, load_collect_settings, load_live_settings  # noqa: E402
from src.live.config import load_live_config  # noqa: E402
from src.live.search import TavilyClient  # noqa: E402

QUERIES = {1: "Hampi monuments opening hours and ticket price",
           2: "Kochi travel conditions during monsoon",
           3: "Mysuru Palace entry ticket price and timings"}
CAP = 3
SCRATCH = REPO_ROOT / ".cache" / "live_smoke"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, choices=sorted(QUERIES), required=True)
    ap.add_argument("--yes", action="store_true", help="really send the request")
    a = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    counter = SCRATCH / "counter.json"
    state = json.loads(counter.read_text()) if counter.exists() else {"calls": 0, "done": []}
    key = load_live_settings().api_key
    cfg = load_live_config()
    client = TavilyClient(make_http(load_collect_settings()), key or "", cfg)
    body = client.request_body(QUERIES[a.index])
    print("request body:", json.dumps(body, sort_keys=True))
    if not a.yes:
        print("dry run: nothing sent")
        return 0
    if not key:
        print("no LIVE_SEARCH_API_KEY in .env")
        return 4
    if state["calls"] >= CAP or a.index in state["done"]:
        print(f"refused: {state['calls']} of {CAP} calls already counted, done={state['done']}")
        return 2
    state["calls"] += 1  # counted BEFORE the call: a failed call is never retried
    state["done"].append(a.index)
    counter.write_text(json.dumps(state))
    t0 = time.monotonic()
    try:
        doc = client.search_doc(QUERIES[a.index])
    except FetchError as exc:
        print(f"call {state['calls']}/{CAP} FAILED after {time.monotonic() - t0:.2f}s: "
              f"{str(exc)[:160]}")
        return 3
    secs = time.monotonic() - t0
    (SCRATCH / f"q{a.index}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"call {state['calls']}/{CAP} ok in {secs:.2f}s; saved raw reply to scratch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
