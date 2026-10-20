"""Round-1 pooling and blind judging sheets for the Phase 4 test collection (D72).

Pure functions; `scripts/pool.py` supplies the real search function. Nothing here reads qrels.

Pool: for every query, the union of the top DEPTH results of the five systems (the four indexes
and their RRF fusion), deduplicated per (query, chunk). `pool.json` holds ids only (no text), so it
can be committed. Judging sheets hold chunk text and are gitignored (O12). A sheet row shows the
query, the chunk's place, state, source type, page title, URL, date kind and text, and NEVER the
context_note, the contributing systems, ranks, scores or provenance. Rows keep the query order but
are shuffled within each query (seeded), so the position of a row says nothing about a system's
ranking.
"""
from __future__ import annotations

import csv
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

DEPTH = 10
SYSTEMS = ("bm25_contextual", "bm25_plain", "dense_plain", "dense_contextual", "rrf4")
ROUND = 1
SHEET_COLUMNS = ("row_id", "query_id", "query_text", "chunk_id", "place", "state", "source_type",
                 "title", "url", "date_kind", "text", "label", "note", "rubric")
RUBRIC_NOTE = ("see docs/labelling_rubric.md: 2 directly answers, 1 on topic but partial, 0 not "
               "relevant; excluded-place chunks are judged like any other")

FIXED_COLUMNS = tuple(c for c in SHEET_COLUMNS if c not in ("label", "note", "rubric"))
_LABEL = re.compile(r"[012](\.0+)?")

Search = Callable[[str, str, int], list[tuple[str, float]]]  # (system, query, k) -> [(id, score)]


def build_pool(queries: list[dict], search: Search, info: dict[str, dict],
               excluded: set[str], depth: int = DEPTH) -> dict:
    """{"top": {qid: {system: [ids]}}, "pairs": {qid: {chunk_id: {...}}}}"""
    top: dict[str, dict[str, list[str]]] = {}
    pairs: dict[str, dict[str, dict]] = {}
    for q in queries:
        top[q["id"]] = {}
        pairs[q["id"]] = {}
        for system in SYSTEMS:
            ids = [cid for cid, _ in search(system, q["text"], depth)][:depth]
            top[q["id"]][system] = ids
            for cid in ids:
                pair = pairs[q["id"]].setdefault(cid, {"round": ROUND, "systems": []})
                pair["systems"].append(system)
        for cid, pair in pairs[q["id"]].items():
            pair["provenance"] = info[cid]["provenance"]
            pair["excluded_place"] = info[cid]["place"] in excluded
    return {"top": top, "pairs": pairs}


def pool_size_at_depth(top: dict[str, dict[str, list[str]]], depth: int) -> dict[str, int]:
    """Pool size per query if only the first `depth` results of each system were pooled."""
    return {qid: len({cid for ids in systems.values() for cid in ids[:depth]})
            for qid, systems in top.items()}


def counts_by_split(pool: dict, queries: list[dict]) -> dict[str, int]:
    split = {q["id"]: q["split"] for q in queries}
    c: Counter = Counter()
    for qid, pairs in pool["pairs"].items():
        c[split[qid]] += len(pairs)
    return {"dev": c["dev"], "test": c["test"], "total": c["dev"] + c["test"]}


def report(pool: dict, queries: list[dict], info: dict[str, dict], excluded: set[str]) -> str:
    qs = {q["id"]: q for q in queries}
    lines = []
    cnt = counts_by_split(pool, queries)
    lines.append(f"judgments to label: dev {cnt['dev']}, test {cnt['test']}, total {cnt['total']}")
    sizes = [len(p) for p in pool["pairs"].values()]
    lines.append(f"pool size per query: min {min(sizes)}, median {statistics.median(sizes):g}, "
                 f"max {max(sizes)} (of {DEPTH * len(SYSTEMS)} slots)")
    lines.append("")
    lines.append(f"{'system':<18}{'unique only':>12}{'excluded@10':>13}   top-10 provenance share "
                 "(llm / fallback / rule_only)")
    for system in SYSTEMS:
        only = 0
        exc = []
        prov: Counter = Counter()
        slots = 0
        for qid, pairs in pool["pairs"].items():
            only += sum(1 for p in pairs.values() if p["systems"] == [system])
            ids = pool["top"][qid][system]
            if ids:
                exc.append(sum(1 for c in ids if info[c]["place"] in excluded) / len(ids))
            for c in ids:
                prov[info[c]["provenance"]] += 1
                slots += 1
        share = " / ".join(f"{prov[k] / slots:.2f}" for k in ("llm", "fallback", "rule_only"))
        lines.append(f"{system:<18}{only:>12}{statistics.mean(exc):>13.3f}   {share}")
    pairs_all = [p for pairs in pool["pairs"].values() for p in pairs.values()]
    prov = Counter(p["provenance"] for p in pairs_all)
    lines.append("")
    lines.append("pool provenance: " + ", ".join(f"{k} {prov[k]}" for k in sorted(prov)))
    lines.append(f"pool chunks in excluded places: {sum(p['excluded_place'] for p in pairs_all)} "
                 f"of {len(pairs_all)}")
    by_src = Counter(info[c]["source_type"] for pairs in pool["pairs"].values() for c in pairs)
    lines.append("pool source types: " + ", ".join(f"{k} {v}" for k, v in sorted(by_src.items())))
    lines.append("")
    lines.append("judgments if the depth were reduced (NOT applied; a decision for the owner):")
    for d in (10, 8, 7, 5):
        sz = pool_size_at_depth(pool["top"], d)
        dev = sum(v for qid, v in sz.items() if qs[qid]["split"] == "dev")
        test = sum(v for qid, v in sz.items() if qs[qid]["split"] == "test")
        lines.append(f"  depth {d:>2}: dev {dev}, test {test}, total {dev + test}")
    return "\n".join(lines)


def sheet_rows(pool: dict, queries: list[dict], chunks: dict[str, dict],
               attribution: dict[str, dict], split: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for q in queries:
        if q["split"] != split:
            continue
        ids = sorted(pool["pairs"][q["id"]])  # sorted first: the shuffle is reproducible
        rng.shuffle(ids)
        for cid in ids:
            c = chunks[cid]
            rows.append({
                "row_id": len(rows) + 1, "query_id": q["id"], "query_text": q["text"],
                "chunk_id": cid, "place": c["place"], "state": c["state"],
                "source_type": c["source_type"],
                "title": (attribution.get(c["url"]) or {}).get("title", ""),
                "url": c["url"], "date_kind": c["date_kind"], "text": c["text"],
                "label": "", "note": "", "rubric": RUBRIC_NOTE,
            })
    return rows


def write_sheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_sheet(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_label(raw: str | None) -> int | None:
    """0, 1 or 2 from a sheet cell. Accepts the digit or a float written exactly as 0.0, 1.0 or 2.0
    (a spreadsheet may save it so); everything else (blank, 1.5, 3, "01", "1e0", text) is None."""
    raw = (raw or "").strip()
    return int(raw[0]) if _LABEL.fullmatch(raw) else None


def verify_unchanged(labelled: list[dict], original: list[dict]) -> list[str]:
    """Problems when a labelled sheet differs from the sheet we exported anywhere except the
    label, note and rubric columns: same row_ids, same fixed columns in every row."""
    problems = []
    if len(labelled) != len(original):
        problems.append(f"{len(labelled)} rows, the exported sheet has {len(original)}")
    by_id = {r["row_id"]: r for r in original}
    seen = set()
    for r in labelled:
        rid = r.get("row_id")
        if rid in seen:
            problems.append(f"row_id {rid}: duplicate")
        seen.add(rid)
        o = by_id.get(rid)
        if o is None:
            problems.append(f"row_id {rid}: not in the exported sheet")
            continue
        for c in FIXED_COLUMNS:
            if r.get(c) != o.get(c):
                problems.append(f"row_id {rid}: column {c} was changed")
    for rid in sorted(set(by_id) - seen, key=int):
        problems.append(f"row_id {rid}: missing from the labelled sheet")
    return problems
