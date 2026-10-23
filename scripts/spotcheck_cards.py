"""Blind manual spot-check of the place cards (D102): are the cited excerpts about the place,
are the tips useful, and are the shown warnings real warnings?

    python scripts/spotcheck_cards.py sample [--seed 42] [--tips 1] [--warnings 2]
                                             [--limited-tips 1]
    python scripts/spotcheck_cards.py score  [--sheet <filled csv>]

sample: per full-evidence place `--tips` tips and up to `--warnings` warnings, per limited place
`--limited-tips` tips, drawn with a seeded RNG from index/place_cards.json. Writes the blind sheet
eval/sheets/spotcheck_cards.csv (place, state, source type, the card excerpt, the full chunk text,
EMPTY label columns; no chunk id, provenance or item kind) and the hidden key
eval/sheets/spotcheck_cards_key.json. Both are gitignored (chunk text, O12).

Fill two columns with 0 or 1:
  about_place  1 if the text is really about the named place (not merely mentioning it);
  valid        1 if, for a tip, it is useful practical information for a visitor, or, for a warning,
               it is a real warning (a risk, closure, scam, danger or restriction) and not just
               a description.

score: refuses a blank, a value other than 0/1, a changed fixed column or a missing row; prints
counts and precision overall, for tips, for warnings, and for limited-evidence places, and lists
the failed rows. Only these numbers are committed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHEETS = ROOT / "eval" / "sheets"
SHEET = SHEETS / "spotcheck_cards.csv"
KEY = SHEETS / "spotcheck_cards_key.json"
CARDS = ROOT / "index" / "place_cards.json"
ENRICHED = ROOT / "data" / "enriched" / "chunks.jsonl"
COLUMNS = ("row_id", "place", "state", "source_type", "excerpt", "chunk_text", "about_place",
           "valid", "note")
FIXED = ("row_id", "place", "state", "source_type", "excerpt", "chunk_text")


def fixed_sha(rows: list[dict]) -> str:
    payload = json.dumps([[r[c] for c in FIXED] for r in rows], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def draw(cards: dict[str, dict], seed: int, tips: int, warnings: int, limited_tips: int
         ) -> list[dict]:
    """[{place, kind, item, evidence_class}] drawn per place with a seeded RNG."""
    rng = random.Random(seed)
    out = []
    for name in sorted(cards):
        card = cards[name]
        if card["evidence_class"] == "full":
            plan = (("tip", card["tips"], tips), ("warning", card["warnings"], warnings))
        else:
            plan = (("tip", card["tips"], limited_tips),)
        for kind, items, n in plan:
            picked = rng.sample(items, min(n, len(items))) if items else []
            for item in picked:
                out.append({"place": name, "kind": kind, "item": item,
                            "evidence_class": card["evidence_class"], "state": card["state"]})
    rng.shuffle(out)
    return out


def score(sheet_rows: list[dict], key: dict) -> str:
    """The report text; raises ValueError on any problem with the filled sheet."""
    ids = list(key["rows"])
    if sorted(r["row_id"] for r in sheet_rows) != sorted(ids):
        raise ValueError("the sheet's row_ids do not match the key")
    if fixed_sha(sorted(sheet_rows, key=lambda r: int(r["row_id"]))) != key["fixed_sha256"]:
        raise ValueError("a fixed column was changed; only about_place, valid and note may "
                         "be edited")
    marks: dict[str, tuple[int, int]] = {}
    bad = []
    for r in sheet_rows:
        a, v = (r["about_place"] or "").strip(), (r["valid"] or "").strip()
        if a not in ("0", "1") or v not in ("0", "1"):
            bad.append(r["row_id"])
        else:
            marks[r["row_id"]] = (int(a), int(v))
    if bad:
        raise ValueError(f"blank or not 0/1 in rows {sorted(bad, key=int)}")

    def rate(select) -> str:
        rows = [i for i in ids if select(key["rows"][i])]
        if not rows:
            return "n/a (0 rows)"
        about = sum(marks[i][0] for i in rows)
        valid = sum(marks[i][1] for i in rows)
        return f"about_place {about}/{len(rows)} ({about / len(rows):.3f}), valid {valid}/" \
               f"{len(rows)} ({valid / len(rows):.3f})"

    lines = [f"rows: {len(ids)}",
             "all: " + rate(lambda k: True),
             "tips: " + rate(lambda k: k["kind"] == "tip"),
             "warnings: " + rate(lambda k: k["kind"] == "warning"),
             "full-evidence places: " + rate(lambda k: k["evidence_class"] == "full"),
             "limited-evidence places: " + rate(lambda k: k["evidence_class"] == "limited")]
    failed = [i for i in ids if marks[i] != (1, 1)]
    lines.append(f"rows with a 0: {sorted(failed, key=int)}")
    for i in sorted(failed, key=int):
        k = key["rows"][i]
        lines.append(f"  row {i}: {k['place']} {k['kind']} chunk {k['chunk_id']} marks "
                     f"about_place={marks[i][0]} valid={marks[i][1]}")
    return "\n".join(lines)


def cmd_sample(args) -> int:
    cards = json.loads(CARDS.read_text(encoding="utf-8"))["places"]
    chunks = {}
    for line in ENRICHED.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
    picks = draw(cards, args.seed, args.tips, args.warnings, args.limited_tips)
    rows = [{"row_id": str(i), "place": p["place"], "state": p["state"],
             "source_type": p["item"]["source_type"], "excerpt": p["item"]["text"],
             "chunk_text": " ".join(chunks[p["item"]["chunk_id"]]["text"].split()),
             "about_place": "", "valid": "", "note": ""} for i, p in enumerate(picks, 1)]
    SHEETS.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    key = {"seed": args.seed, "fixed_sha256": fixed_sha(rows),
           "rows": {r["row_id"]: {"place": p["place"], "kind": p["kind"],
                                  "chunk_id": p["item"]["chunk_id"],
                                  "provenance": p["item"]["provenance"],
                                  "evidence_class": p["evidence_class"]}
                    for r, p in zip(rows, picks)}}
    KEY.write_text(json.dumps(key, indent=1, sort_keys=True), encoding="utf-8")
    kinds = {k: sum(1 for p in picks if p["kind"] == k) for k in ("tip", "warning")}
    print(f"sheet: {SHEET}\nrows: {len(rows)} ({kinds['tip']} tips, {kinds['warning']} "
          f"warnings)\nkey (hidden): {KEY}")
    return 0


def cmd_score(args) -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    with Path(args.sheet).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    try:
        print(score(rows, key))
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--tips", type=int, default=1)
    s.add_argument("--warnings", type=int, default=2)
    s.add_argument("--limited-tips", type=int, default=1)
    c = sub.add_parser("score")
    c.add_argument("--sheet", default=str(SHEET))
    args = ap.parse_args(argv)
    return cmd_sample(args) if args.cmd == "sample" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
