"""Read-only comparison of two enrichment checkpoints over the same chunks. No API call.

    python scripts/compare_runs.py                      # the v1 (qwen) checkpoint vs the v3 one
    python scripts/compare_runs.py --samples 20         # + 20 samples, >= 6 whose aspect changed

CONFOUNDED (D59): the v1 checkpoint was produced by a different model (qwen2.5 on Ollama) than
the current one (a hosted gpt-oss model), so any difference mixes the prompt change with the model
change; it says nothing about the prompt alone. Reads data/processed/chunks.jsonl for the text.
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


def load(spec: str) -> dict[str, dict]:
    """A checkpoint file name, a path, or a prompt version (then the single matching file)."""
    path = Path(spec)
    if not path.exists():
        path = ENRICHED / spec
    if not path.exists():
        found = sorted(ENRICHED.glob(f"checkpoint_{spec}*.jsonl"))
        if len(found) != 1:
            sys.exit(f"{spec!r}: expected exactly one checkpoint file, found {len(found)}")
        path = found[0]
    return {r["chunk_id"]: r for r in read_jsonl(path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="checkpoint_enrich-v1.jsonl")
    ap.add_argument("--new", default="enrich-v3")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    old, new = load(args.old), load(args.new)
    cmp = compare_runs(old, new)
    print("CONFOUNDED: the two checkpoints come from different models (D59); a difference mixes "
          "the prompt and the model.")
    print(format_comparison(cmp, "old", "new"))
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
