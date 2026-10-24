"""Citation precision (D125): extract cited claims from answers, sample them, build the private
claim sheet, read the owner's marks.

A claim is one bullet of the answer's day sections that carries at least one [chunk:<id>] or
[live:<id>] marker; the computed route, season and budget lines carry none, and the Sources and
Notes sections are skipped. Corpus claims are checked against the cited chunk's excerpt; live
claims list only the cited item's place, fetch date and link (the provider snippet is not stored,
D109) and are reported as counts, never as a rate with an interval.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable

MARKER = re.compile(r"\[(chunk|live):([^\]\s]+)\]")
BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
STOP = re.compile(r"^#{1,3}\s*(Sources|Notes and warnings)\b", re.I)
FIXED = ("row_id", "request", "claim", "passages")
COLUMNS = (*FIXED, "supported", "note")


def extract_claims(record: dict) -> list[dict]:
    """The cited claims of one record, in text order."""
    out = []
    for line in record.get("text", "").splitlines():
        if STOP.match(line.strip()):
            break
        m = BULLET.match(line)
        if not m:
            continue
        marks = MARKER.findall(m.group(1))
        if not marks:
            continue
        text = " ".join(MARKER.sub("", m.group(1)).split())
        out.append({"claim_id": f"{record['request_id']}-{len(out) + 1}",
                    "request_id": record["request_id"], "arm": record["arm"], "text": text,
                    "cited": [{"kind": k, "id": cid} for k, cid in marks]})
    return out


def is_live(claim: dict) -> bool:
    return any(c["kind"] == "live" for c in claim["cited"])


def sample_claims(records: Iterable[dict], n: int = 50, *, seed: int = 42,
                  arms: tuple[str, ...] = ("full_corpus",)) -> list[dict]:
    """A seeded sample of `n` claims spread over the requests: one claim at a time from each
    request in turn (requests in seeded order), so no request dominates."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for rec in sorted(records, key=lambda r: (r["request_id"], r["arm"])):
        if rec["arm"] not in arms:
            continue
        claims = [c for c in extract_claims(rec) if not is_live(c)]
        rng.shuffle(claims)
        pools[rec["request_id"]] = claims
    order = sorted(pools)
    rng.shuffle(order)
    chosen: list[dict] = []
    while len(chosen) < n and any(pools.values()):
        for rid in order:
            if pools[rid] and len(chosen) < n:
                chosen.append(pools[rid].pop())
    return chosen


def live_claims(records: Iterable[dict]) -> list[dict]:
    """Every claim of the live arm that cites a live item (counted, listed with links only)."""
    out = []
    for rec in sorted(records, key=lambda r: r["request_id"]):
        if rec["arm"] == "full_live":
            out += [c for c in extract_claims(rec) if is_live(c)]
    return out


def passages(claim: dict, record: dict) -> str:
    by_chunk = {c["chunk_id"]: c for c in record.get("contexts", [])}
    meta = {c["chunk_id"]: c for c in record.get("citations", [])}
    lines = []
    for c in claim["cited"]:
        cid = c["id"]
        ctx = by_chunk.get(cid) or by_chunk.get(f"live:{cid}") or {}
        info = meta.get(cid) or meta.get(f"live:{cid}") or {}
        place = ctx.get("place") or info.get("place") or "?"
        if c["kind"] == "live":
            lines.append(f"[live] {place}, fetched {info.get('fetch_date') or 'today'}: "
                         f"{info.get('url', '')}  (open the link and check)")
        else:
            lines.append(f"[{place}] {ctx.get('text', '(excerpt not stored)')}")
    return "\n".join(lines)


def fixed_sha(row: dict) -> str:
    h = hashlib.sha256()
    for col in FIXED[1:]:
        h.update(str(row[col]).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def build_claim_sheet(claims: list[dict], records: dict[tuple[str, str], dict],
                      requests: dict[str, dict], *, seed: int = 42) -> tuple[list[dict], dict]:
    """(rows, key); rows are shuffled with the seed and named C01, C02, ..."""
    order = list(claims)
    random.Random(seed).shuffle(order)
    rows, key = [], {}
    width = max(2, len(str(len(order))))
    for i, c in enumerate(order, 1):
        rec = records[(c["request_id"], c["arm"])]
        row = {"row_id": f"C{i:0{width}d}", "request": requests[c["request_id"]]["text"],
               "claim": c["text"], "passages": passages(c, rec), "supported": "", "note": ""}
        rows.append(row)
        key[row["row_id"]] = {"claim_id": c["claim_id"], "request_id": c["request_id"],
                              "arm": c["arm"], "live": is_live(c), "sha": fixed_sha(row)}
    return rows, key


def read_marks(filled: list[dict], key: dict) -> dict[str, int]:
    """{row_id: 0 or 1}; refuses missing rows, changed fixed columns and any other value."""
    by_id = {r["row_id"]: r for r in filled}
    missing = sorted(set(key) - set(by_id))
    if missing:
        raise ValueError(f"rows missing from the filled sheet: {missing[:5]}")
    out = {}
    for rid, meta in key.items():
        row = by_id[rid]
        if fixed_sha(row) != meta["sha"]:
            raise ValueError(f"row {rid}: a fixed column was changed")
        v = str(row["supported"]).strip()
        try:
            f = float(v)
        except ValueError as exc:
            raise ValueError(f"row {rid}: supported must be 0 or 1, got {v!r}") from exc
        if f not in (0.0, 1.0):
            raise ValueError(f"row {rid}: supported must be 0 or 1, got {v!r}")
        out[rid] = int(f)
    return out


def precision(marks: dict[str, int], key: dict) -> dict:
    """Citation precision of the corpus claims; live claims are counts only (D125)."""
    corpus = [m for rid, m in marks.items() if not key[rid]["live"]]
    live = [m for rid, m in marks.items() if key[rid]["live"]]
    return {"corpus_n": len(corpus), "corpus_supported": sum(corpus),
            "corpus_precision": sum(corpus) / len(corpus) if corpus else None,
            "live_n": len(live), "live_supported": sum(live)}
