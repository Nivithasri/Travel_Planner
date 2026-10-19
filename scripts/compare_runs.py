"""Read-only comparison of two enrichment runs over the same chunks (D58). No LLM call.

    python scripts/compare_runs.py                      # enrich-v1 vs enrich-v2 side by side
    python scripts/compare_runs.py --samples 20         # + 20 samples, >= 6 whose aspect changed

Reads data/enriched/checkpoint_<version>.jsonl for both versions and data/processed/chunks.jsonl
for the text. The sample list prefers chunks whose aspect changed to or from closure/safety.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.cache import read_jsonl  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402
from src.process.enrich import compare_runs, format_comparison  # noqa: E402

ENRICHED = REPO_ROOT / "data" / "enriched"
FAST = {"closure", "safety"}


def load(version: str) -> dict[str, dict]:
    return {r["chunk_id"]: r for r in read_jsonl(ENRICHED / f"checkpoint_{version}.jsonl")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="enrich-v1")
    ap.add_argument("--new", default="enrich-v2")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    old, new = load(args.old), load(args.new)
    cmp = compare_runs(old, new)
    print(format_comparison(cmp, args.old.split("-")[-1], args.new.split("-")[-1]))
    if not args.samples:
        return 0
    chunks = {}
    for line in (REPO_ROOT / "data" / "processed" / "chunks.jsonl").read_text(
            encoding="utf-8").splitlines():
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
    rng = random.Random(args.seed)
    changed = [t for t in cmp["aspect_changed"]]
    fast = [t for t in changed if t[1] in FAST or t[2] in FAST]
    other = [t for t in changed if t not in fast]
    rng.shuffle(fast)
    rng.shuffle(other)
    picked = [t[0] for t in (fast[:10] + other)][: max(6, args.samples // 2)]
    rest = [i for i in cmp["ids"] if i not in picked]
    rng.shuffle(rest)
    picked += rest[: args.samples - len(picked)]
    moved = {t[0]: t for t in changed}
    print(f"\n{len(picked)} samples; {sum(i in moved for i in picked)} with a changed aspect "
          f"({sum(i in {t[0] for t in fast} for i in picked)} involving closure/safety)")
    for n, i in enumerate(picked, 1):
        c, o, w = chunks[i], old[i], new[i]
        p = w["prov"]
        src = "llm" if p["source"] == "llm" else f"fallback:{p['reason']}"
        print("=" * 100)
        print(f"[{n}] {i} | {c['place']} | {c['source_type']} | {src} | calls {p['network_calls']}"
              f" | dropped {p['dropped']}")
        print("TEXT:", c["text"])
        print(f"v1: aspect {o['fields']['aspect']}, place_type {o['fields']['place_type']}, "
              f"is_warning {o['fields']['is_warning']}")
        print(f"v2: aspect {w['fields']['aspect']}, place_type {w['fields']['place_type']}, "
              f"is_warning {w['fields']['is_warning']}, season {w['fields']['season_tags']}, "
              f"mentioned {w['fields']['mentioned_places']}")
        print("v2 NOTE:", w["fields"]["context_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
