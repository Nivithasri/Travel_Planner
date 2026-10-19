"""ESTIMATE ONLY, read-only, no network: place x source type and place x aspect coverage.

    python scripts/coverage_matrix.py

Aspect columns come from keyword rules (src/collect/size_estimate.py): a planning guide, not labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.orchestrator import CORE_REGISTRY, RAW_ROOT  # noqa: E402
from src.collect.size_estimate import ASPECTS, build_matrix  # noqa: E402
from src.collect.types import SOURCE_TYPE  # noqa: E402
from src.places.registry import Registry  # noqa: E402

TYPES = sorted(set(SOURCE_TYPE.values()))


def cell_text(cells: dict, t: str) -> str:
    if t not in cells:
        return "-"
    return f"{cells[t]['records']} ({cells[t]['chars']})"


def main() -> int:
    records = [json.loads(line) for f in sorted(RAW_ROOT.glob("*/*/records.jsonl"))
               for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    mat = build_matrix(records)
    entries = sorted(Registry.load(CORE_REGISTRY).entries, key=lambda x: x.name)
    print("ESTIMATE ONLY: keyword-rule aspect tags; counts are records with a keyword hit.")
    print("\nA. records (cleaned chars) per place x source type;")
    print("   'not collected' = no raw data, 'hold-out' = never collected (D17)\n")
    print(f"{'place':14}" + "".join(f"{t:>16}" for t in TYPES))
    for e in entries:
        if e.holdout:
            print(f"{e.name:14}hold-out")
        elif e.name not in mat:
            print(f"{e.name:14}not collected")
        else:
            cells = mat[e.name]["by_type"]
            print(f"{e.name:14}" + "".join(f"{cell_text(cells, t):>16}" for t in TYPES))
    print("\nB. aspect coverage: records with a keyword hit, per collected place\n")
    print(f"{'place':14}" + "".join(f"{a:>10}" for a in ASPECTS))
    for e in entries:
        if e.name in mat and not e.holdout:
            asp = mat[e.name]["aspects"]
            print(f"{e.name:14}" + "".join(f"{asp.get(a, 0):>10}" for a in ASPECTS))
    print("\nC. publish dates (publish_date only; wiki revision dates never count, O9)\n")
    print(f"{'place':14}{'records':>9}{'dated':>7}{'share':>8}  newest publish date")
    for e in entries:
        if e.name in mat and not e.holdout:
            m = mat[e.name]
            print(f"{e.name:14}{m['records']:>9}{m['dated']:>7}{m['dated'] / m['records']:>8.0%}  "
                  f"{m['newest'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
