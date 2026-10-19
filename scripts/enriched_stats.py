"""Distributions of the assembled enriched corpus (read-only, no API call).

    python scripts/enriched_stats.py

Reads data/enriched/chunks.jsonl and provenance.jsonl (written by `scripts/enrich.py --assemble`)
and prints aspect, place_type, season_tags and is_warning for all rows and for the model-written
rows only, plus the provenance counts (llm, fallback by reason, rule_only by reason).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schemas import REPO_ROOT  # noqa: E402
from src.process.enrich import enriched_stats  # noqa: E402


def main() -> int:
    d = REPO_ROOT / "data" / "enriched"
    if not (d / "chunks.jsonl").exists() or not (d / "provenance.jsonl").exists():
        print("data/enriched/chunks.jsonl or provenance.jsonl is missing: run "
              "`python scripts/enrich.py --assemble --eligible-only` after the full run")
        return 2
    rows = [json.loads(line) for line in
            (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    prov = [json.loads(line) for line in
            (d / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    st = enriched_stats(rows, prov)
    print(f"rows {st['rows']}; by provenance {st['by_source']}")
    for part, title in (("all", "ALL ROWS (model-written + fallback + rule_only)"),
                        ("llm_only", "MODEL-WRITTEN ROWS ONLY")):
        s = st[part]
        print(f"\n== {title}" + (f": {s['rows']} rows" if "rows" in s else ""))
        for key in ("aspect", "place_type", "season_tags"):
            print(f"{key}: {s[key]}")
        print(f"is_warning true: {s['is_warning_true']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
