"""Turn a filled judging sheet into qrels (D71, D72).

    python scripts/import_labels.py dev      # eval/sheets/judge_dev.csv  -> eval/qrels_dev.json
    python scripts/import_labels.py test     # eval/sheets/judge_test.csv -> eval/qrels_test.json

Validation (nothing is written unless every check passes): the label of every row is exactly 0,
1 or 2 (no blanks); row_ids are exactly 1..N with no duplicates; every (query, chunk) pair is
unique, belongs to the pool of this split and has its query text unchanged; no pooled pair is
missing. An existing qrels file is never overwritten without --force. The script never reads the
other split's qrels and never fills a label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import pooling  # noqa: E402
from eval.qrels import SPLITS, load_queries, qrels_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "eval" / "pool.json"
SHEETS = ROOT / "eval" / "sheets"
VALID = {"0": 0, "1": 1, "2": 2}


def validate(rows: list[dict], pool: dict, queries: list[dict], split: str
             ) -> tuple[dict[str, dict[str, int]], list[str]]:
    """(qrels, problems). qrels is complete only when problems is empty."""
    problems: list[str] = []
    qtext = {q["id"]: q["text"] for q in queries if q["split"] == split}
    expected = {(qid, cid) for qid in qtext for cid in pool["pairs"].get(qid, {})}
    seen: set[tuple[str, str]] = set()
    qrels: dict[str, dict[str, int]] = {qid: {} for qid in qtext}
    row_ids = []
    for i, r in enumerate(rows, 2):  # sheet line number: header is line 1
        where = f"line {i} (row_id {r.get('row_id')!r})"
        rid = (r.get("row_id") or "").strip()
        if not rid.isdigit():
            problems.append(f"{where}: row_id is not a positive integer")
        else:
            row_ids.append(int(rid))
        key = (r.get("query_id", ""), r.get("chunk_id", ""))
        if key not in expected:
            problems.append(f"{where}: {key[0]}/{key[1]} is not in the {split} pool")
            continue
        if r.get("query_text") != qtext[key[0]]:
            problems.append(f"{where}: query_text was changed")
        if key in seen:
            problems.append(f"{where}: duplicate row for {key[0]}/{key[1]}")
            continue
        seen.add(key)
        raw = (r.get("label") or "").strip()
        if raw == "":
            problems.append(f"{where}: blank label")
        elif raw not in VALID:
            problems.append(f"{where}: label {raw!r} is not 0, 1 or 2")
        else:
            qrels[key[0]][key[1]] = VALID[raw]
    if sorted(row_ids) != list(range(1, len(rows) + 1)):
        problems.append("row_ids are not exactly 1..N without gaps or duplicates")
    for qid, cid in sorted(expected - seen):
        problems.append(f"missing row for {qid}/{cid}")
    return qrels, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split", choices=SPLITS)
    ap.add_argument("--sheet", type=Path,
                    help="filled sheet (default eval/sheets/judge_<split>.csv)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing qrels file")
    args = ap.parse_args(argv)

    out = qrels_path(args.split)
    if out.exists() and not args.force:
        print(f"{out.name} exists; refusing to overwrite it without --force")
        return 2
    sheet = args.sheet or SHEETS / f"judge_{args.split}.csv"
    if not sheet.exists():
        print(f"no sheet at {sheet}")
        return 2
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    qrels, problems = validate(pooling.read_sheet(sheet), pool, load_queries(), args.split)
    if problems:
        print(f"{len(problems)} problem(s) in {sheet.name}; nothing written:")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        return 1
    out.write_text(json.dumps(qrels, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(v) for v in qrels.values())
    dist = {g: sum(1 for v in qrels.values() for x in v.values() if x == g) for g in (0, 1, 2)}
    print(f"wrote {out.name}: {len(qrels)} queries, {n} judgments, labels {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
