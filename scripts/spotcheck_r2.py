"""Blind spot-check of the chat-made round-2 relevance labels (D95): how often do YOU agree?

    python scripts/spotcheck_r2.py sample [--n 30] [--seed 42] [--min-blog 8]
    python scripts/spotcheck_r2.py score  [--sheet <filled csv>]

sample: draws a seeded random set of round-2 pairs that the cross-encoder rows (row7_rerank,
row8_mmr, row9_crag; eval/pool2.json "systems") surfaced, balanced over those three rows and with at
least --min-blog blog chunks, from the dev and test round-2 sheets. Writes the blind sheet
eval/sheets/spotcheck_r2.csv (query, title, place, state, source type, chunk text, EMPTY label and
note) and the hidden key eval/sheets/spotcheck_r2_key.json (row_id -> chunk_id, query id, split,
chat label, sha of the sheet's fixed columns). Both are gitignored (chunk text, O12). The sheet
never shows the chat label, the system, the rank or the URL. Label by docs/labelling_rubric.md and
docs/labelling_rules.md, like round 1.

score: reads the filled sheet and the key and prints exact agreement, unweighted and
linear-weighted Cohen's kappa, the 3x3 table (rows = you, columns = chat) and the rows where the
two differ by two grades. It refuses a blank or non-0/1/2 label, a changed fixed column or a
missing row. Run it only after you have filled the sheet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import pooling  # noqa: E402
from scripts.kappa import confusion, kappa  # noqa: E402

SHEETS = ROOT / "eval" / "sheets"
SHEET = SHEETS / "spotcheck_r2.csv"
KEY = SHEETS / "spotcheck_r2_key.json"
POOL2 = ROOT / "eval" / "pool2.json"
STRATA = ("row7_rerank", "row8_mmr", "row9_crag")
COLUMNS = ("row_id", "query_text", "title", "place", "state", "source_type", "text",
           "label", "note", "rubric")
FIXED = tuple(c for c in COLUMNS if c not in ("label", "note", "rubric"))
SPLITS = ("dev", "test")


def fixed_sha(rows: list[dict]) -> str:
    payload = json.dumps([[r[c] for c in FIXED] for r in rows], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidates(pairs: dict, labelled: dict[str, list[dict]]) -> list[dict]:
    """Chat-labelled round-2 rows that at least one stratum row surfaced, each with its strata."""
    out = []
    for split, rows in labelled.items():
        for r in rows:
            pair = pairs.get(r["query_id"], {}).get(r["chunk_id"])
            if pair is None:
                continue
            strata = [s for s in STRATA if s in pair["systems"]]
            label = pooling.parse_label(r["label"])
            if strata and label is not None:
                out.append({**r, "split": split, "strata": strata, "chat_label": label})
    return out


def draw(cands: list[dict], n: int, seed: int, min_blog: int) -> list[dict]:
    """Seeded draw balanced over the strata with at least `min_blog` blog chunks. A pool too
    small for the constraints is an error, never a silent shortfall."""
    rng = random.Random(seed)
    order = sorted(cands, key=lambda c: (c["split"], c["query_id"], c["chunk_id"]))
    rng.shuffle(order)
    chosen: list[dict] = []
    taken: set[tuple[str, str]] = set()

    def rotate(pool: list[dict], count: int) -> None:
        """Take `count` rows, going round the strata (a row sits in the first free stratum)."""
        queues = {s: [c for c in pool if s in c["strata"]] for s in STRATA}
        while count > 0 and any(queues.values()):
            for s in STRATA:
                while queues[s] and (queues[s][0]["query_id"], queues[s][0]["chunk_id"]) in taken:
                    queues[s].pop(0)
                if queues[s] and count > 0:
                    c = queues[s].pop(0)
                    taken.add((c["query_id"], c["chunk_id"]))
                    chosen.append(c)
                    count -= 1

    rotate([c for c in order if c["source_type"] == "blog"], min_blog)
    if sum(c["source_type"] == "blog" for c in chosen) < min_blog:
        raise SystemExit("not enough blog candidates for --min-blog")
    rotate(order, n - len(chosen))
    if len(chosen) < n:
        raise SystemExit(f"only {len(chosen)} candidates for n={n}")
    rng.shuffle(chosen)
    return chosen


def cmd_sample(args) -> int:
    pairs = json.loads(POOL2.read_text(encoding="utf-8"))["pairs"]
    labelled = {s: pooling.read_sheet(SHEETS / f"judge2_{s}_labelled.csv") for s in SPLITS}
    pick = draw(candidates(pairs, labelled), args.n, args.seed, args.min_blog)
    rows = [{"row_id": str(i), "query_text": c["query_text"], "title": c["title"],
             "place": c["place"], "state": c["state"], "source_type": c["source_type"],
             "text": c["text"], "label": "", "note": "", "rubric": pooling.RUBRIC_NOTE}
            for i, c in enumerate(pick, 1)]
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    key = {"seed": args.seed, "n": args.n, "min_blog": args.min_blog, "strata": list(STRATA),
           "fixed_columns_sha256": fixed_sha(rows),
           "rows": {r["row_id"]: {"chunk_id": c["chunk_id"], "query_id": c["query_id"],
                                  "split": c["split"], "chat_label": c["chat_label"]}
                    for r, c in zip(rows, pick)}}
    KEY.write_text(json.dumps(key, indent=1, sort_keys=True), encoding="utf-8")
    comp = Counter(r["source_type"] for r in rows)
    strata = Counter(s for c in pick for s in c["strata"])
    print(f"sheet: {SHEET}\nrows: {len(rows)}\nsource types: {dict(comp)}\n"
          f"pairs surfaced by (a pair can sit in several): {dict(strata)}\nkey (hidden): {KEY}")
    return 0


def score(sheet_rows: list[dict], key: dict) -> str:
    """The report text; raises ValueError on any problem with the filled sheet."""
    ids = list(key["rows"])
    if sorted(r["row_id"] for r in sheet_rows) != sorted(ids):
        raise ValueError("the sheet's row_ids do not match the key (rows added, removed, renamed)")
    ordered = sorted(sheet_rows, key=lambda r: int(r["row_id"]))
    if fixed_sha(ordered) != key["fixed_columns_sha256"]:
        raise ValueError("a fixed column (query, title, place, state, source type or text) was "
                         "changed; only label and note may be edited")
    mine = {r["row_id"]: pooling.parse_label(r["label"]) for r in sheet_rows}
    bad = [i for i, v in mine.items() if v is None]
    if bad:
        raise ValueError(f"blank or not 0/1/2 label in rows {sorted(bad, key=int)}")
    a = [mine[i] for i in ids]
    b = [int(key["rows"][i]["chat_label"]) for i in ids]
    table = confusion(a, b)
    exact = sum(x == y for x, y in zip(a, b)) / len(a)

    def f(v):
        return "undefined" if v is None else f"{v:.3f}"

    lines = [f"rows: {len(ids)}", f"exact agreement: {exact:.3f}",
             f"kappa (unweighted): {f(kappa(table))}",
             f"kappa (linear-weighted): {f(kappa(table, 'linear'))}", "",
             "confusion (rows = you, columns = chat)", "        chat0  chat1  chat2"]
    for i in range(3):
        lines.append(f"you {i}   " + "  ".join(f"{table[i][j]:>5}" for j in range(3)))
    far = [(i, x, y) for i, x, y in zip(ids, a, b) if abs(x - y) == 2]
    lines += ["", f"rows differing by 2 grades: {len(far)}"]
    for i, x, y in far:
        lines.append(f"  row {i}: you {x}, chat {y} ({key['rows'][i]['query_id']}, "
                     f"{key['rows'][i]['chunk_id']})")
    return "\n".join(lines)


def cmd_score(args) -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    try:
        print(score(pooling.read_sheet(Path(args.sheet)), key))
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--min-blog", type=int, default=8)
    c = sub.add_parser("score")
    c.add_argument("--sheet", default=str(SHEET))
    args = ap.parse_args(argv)
    return cmd_sample(args) if args.cmd == "sample" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
