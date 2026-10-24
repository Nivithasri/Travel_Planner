"""Blind scoring sheets for the Phase 9 itinerary comparison (D124).

A sheet row is one (request, output) pair. The row id is a seeded shuffle, the sheet never names the
system, arm, plan, critic status or token counts. The output is shown with short source markers
([S1].. for corpus passages, [L1].. for live items) numbered by first appearance, and the sources
column lists each marker with its place, type, link and (corpus only) the excerpt the writer saw.
Live items are listed with their link only (the provider snippet is not stored, D109): open it.

The hidden key maps row_id to request id, system and arm, and holds a sha256 of the fixed columns
so an import can prove the scorer changed nothing but the score and note cells.
Sheets and keys hold excerpt text: they stay under eval/sheets (gitignored, O12); only agreement
numbers and aggregated scores are committed.
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
from collections.abc import Iterable
from pathlib import Path

DIMENSIONS = ("feasibility", "groundedness", "freshness", "personalization")
FIXED = ("row_id", "request", "output", "sources")
COLUMNS = (*FIXED, *DIMENSIONS, "note")
MARKER = re.compile(r"\[(chunk|live):([^\]\s]+)\]")


def render(record: dict) -> tuple[str, str]:
    """(output text with short markers, sources column text) for one record."""
    ids: dict[str, str] = {}
    counters = {"S": 0, "L": 0}

    def tag(m: re.Match) -> str:
        kind, cid = m.group(1), m.group(2)
        key = f"{kind}:{cid}"
        if key not in ids:
            letter = "L" if kind == "live" else "S"
            counters[letter] += 1
            ids[key] = f"{letter}{counters[letter]}"
        return f"[{ids[key]}]"

    text = MARKER.sub(tag, record.get("text", ""))
    by_chunk = {c["chunk_id"]: c for c in record.get("contexts", [])}
    meta = {c["chunk_id"]: c for c in record.get("citations", [])}
    lines = []
    for key, alias in ids.items():
        kind, cid = key.split(":", 1)
        c = by_chunk.get(cid) or by_chunk.get(f"live:{cid}") or {}
        m = meta.get(cid) or meta.get(f"live:{cid}") or {}
        place = c.get("place") or m.get("place") or "?"
        stype = c.get("source_type") or m.get("source_type") or "?"
        url = c.get("url") or m.get("url") or ""
        if kind == "live":
            when = m.get("fetch_date") or "today"
            lines.append(f"[{alias}] {place} (live, fetched {when}) {url}  (open the link)")
        else:
            lines.append(f"[{alias}] {place} ({stype}) {url}\n    excerpt: {c.get('text', '')}")
    return text, "\n".join(lines)


def fixed_sha(row: dict) -> str:
    h = hashlib.sha256()
    for col in FIXED[1:]:
        h.update(str(row[col]).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def build_sheet(records: Iterable[dict], requests: dict[str, dict], *, seed: int = 42
                ) -> tuple[list[dict], dict]:
    """(rows, key). Records are shuffled with the seed; row ids are R01, R02, ... in that order."""
    recs = sorted(records, key=lambda r: (r["request_id"], r["arm"]))
    random.Random(seed).shuffle(recs)
    rows, key = [], {}
    width = max(2, len(str(len(recs))))
    for i, rec in enumerate(recs, 1):
        text, sources = render(rec)
        req = requests[rec["request_id"]]
        row = {"row_id": f"R{i:0{width}d}", "request": req["text"], "output": text,
               "sources": sources, **{d: "" for d in DIMENSIONS}, "note": ""}
        rows.append(row)
        key[row["row_id"]] = {"request_id": rec["request_id"], "system": rec["system"],
                              "arm": rec["arm"], "has_live": bool(rec.get("live")),
                              "sha": fixed_sha(row)}
    return rows, key


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_score(value: str) -> int:
    """A whole number 1 to 5; floats only when exactly integral (sheets saved by a spreadsheet)."""
    text = str(value).strip()
    try:
        f = float(text)
    except ValueError as exc:
        raise ValueError(f"not a score: {value!r}") from exc
    if f != int(f) or not 1 <= int(f) <= 5:
        raise ValueError(f"a score must be a whole number 1 to 5, got {value!r}")
    return int(f)


def read_scores(filled: list[dict], key: dict) -> dict[str, dict[str, int]]:
    """{row_id: {dimension: score}}. Refuses a missing row, a changed fixed column, a blank or
    invalid score."""
    by_id = {r["row_id"]: r for r in filled}
    missing = sorted(set(key) - set(by_id))
    if missing:
        raise ValueError(f"rows missing from the filled sheet: {missing[:5]}")
    out: dict[str, dict[str, int]] = {}
    for rid, meta in key.items():
        row = by_id[rid]
        if fixed_sha(row) != meta["sha"]:
            raise ValueError(f"row {rid}: a fixed column was changed")
        scores = {}
        for d in DIMENSIONS:
            try:
                scores[d] = parse_score(row[d])
            except ValueError as exc:
                raise ValueError(f"row {rid}, {d}: {exc}") from exc
        out[rid] = scores
    return out


def by_request_system(scores: dict[str, dict[str, int]], key: dict, *, arm: bool = False
                      ) -> dict[tuple[str, str], dict[str, int]]:
    """{(request_id, system or arm): scores}; arm=True keeps the full system's two arms apart."""
    out = {}
    for rid, s in scores.items():
        meta = key[rid]
        out[(meta["request_id"], meta["arm"] if arm else meta["system"])] = s
    return out


def cross_check_sample(key: dict, n: int, *, seed: int = 42) -> list[str]:
    """Row ids of a seeded sample of CORPUS-ONLY rows for the second scorer (D124): rows whose
    output came from a live run never go in it (D100)."""
    eligible = sorted(rid for rid, m in key.items()
                      if not m["has_live"] and m["arm"] != "full_live")
    return sorted(random.Random(seed).sample(eligible, min(n, len(eligible))))
