"""Cohen's kappa between the owner and a second labeller on a shared subset (O4, D74).

    python scripts/kappa.py select     # choose ~100 judgments, write the partner's blind sheet
    python scripts/kappa.py compute    # kappa from the owner's filled sheets and the partner's CSV

`select` draws, from the round-1 pool, Wikipedia ("reference") and Wikivoyage chunks only (no blog),
split between the dev and test queries in proportion to what the pool offers, spread evenly over the
queries of each split, seeded. It writes eval/kappa_subset.json (ids only, committable) and
eval/sheets/kappa_partner.csv (same columns as the judge sheets, empty label, no owner labels, no
system names; gitignored: it holds chunk text, O12). It refuses to overwrite either file without
--force.

`compute` reads the owner's labels from the filled judging sheets (not from the qrels files) and the
partner's filled CSV. With no partner file it prints "kappa not computed" and nothing else is
claimed. It reports unweighted and linearly weighted kappa, observed agreement and the table.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import pooling  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "eval" / "pool.json"
SUBSET = ROOT / "eval" / "kappa_subset.json"
SHEETS = ROOT / "eval" / "sheets"
PARTNER_SHEET = SHEETS / "kappa_partner.csv"
ELIGIBLE = ("reference", "wikivoyage")  # no blog chunks leave the project (D74)
SEED = 42
N = 100
LABELS = (0, 1, 2)


# ---- kappa ----------------------------------------------------------------------------------

def confusion(a: list[int], b: list[int]) -> list[list[int]]:
    """3x3 counts: rows = first labeller (0,1,2), columns = second labeller."""
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    t = [[0] * 3 for _ in LABELS]
    for x, y in zip(a, b):
        t[x][y] += 1
    return t


def kappa(table: list[list[int]], weights: str = "none") -> float | None:
    """Cohen's kappa for a 3x3 table; weights "none" or "linear" (w = 1 - |i-j|/2). None when the
    expected agreement is 1 (both labellers used one label throughout): kappa is undefined."""
    n = sum(map(sum, table))
    if n == 0:
        return None
    rows = [sum(r) for r in table]
    cols = [sum(table[i][j] for i in range(3)) for j in range(3)]

    def w(i: int, j: int) -> float:
        return float(i == j) if weights == "none" else 1 - abs(i - j) / 2

    po = sum(w(i, j) * table[i][j] for i in range(3) for j in range(3)) / n
    pe = sum(w(i, j) * rows[i] * cols[j] for i in range(3) for j in range(3)) / (n * n)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


# ---- selection ------------------------------------------------------------------------------

def select(pool: dict, queries: list[dict], source_type: dict[str, str], n: int = N,
           seed: int = SEED) -> list[tuple[str, str]]:
    """About n (query_id, chunk_id) pairs, eligible source types only. The quota of each split is
    proportional to its eligible pairs; inside a split the quota is spread evenly over the queries
    (a query with fewer eligible pairs gives all of them, the rest is shared among the others)."""
    rng = random.Random(seed)
    cand: dict[str, dict[str, list[str]]] = {"dev": {}, "test": {}}
    for q in sorted(queries, key=lambda q: q["id"]):
        ids = sorted(c for c in pool["pairs"][q["id"]] if source_type[c] in ELIGIBLE)
        if ids:
            cand[q["split"]][q["id"]] = ids
    total = sum(len(v) for split in cand.values() for v in split.values())
    if total <= n:
        return [(qid, c) for split in cand.values() for qid, ids in split.items() for c in ids]
    quota = {s: round(n * sum(len(v) for v in cand[s].values()) / total) for s in cand}
    quota["test"] = n - quota["dev"]  # exactly n in all
    chosen: list[tuple[str, str]] = []
    for s in ("dev", "test"):
        remaining = {qid: list(ids) for qid, ids in cand[s].items()}
        for ids in remaining.values():
            rng.shuffle(ids)
        left = min(quota[s], sum(len(v) for v in remaining.values()))
        while left > 0:
            for qid in sorted(remaining):  # round-robin: one pair per query per pass
                if left > 0 and remaining[qid]:
                    chosen.append((qid, remaining[qid].pop()))
                    left -= 1
    return chosen


def partner_rows(subset: list[tuple[str, str]], pool: dict, queries: list[dict],
                 chunks: dict[str, dict], attribution: dict[str, dict], seed: int = SEED
                 ) -> list[dict]:
    keep: dict[str, dict] = {}
    for qid, cid in subset:
        keep.setdefault(qid, {})[cid] = pool["pairs"][qid][cid]
    sub = {"pairs": keep}
    rows = []
    for split in ("dev", "test"):
        part = [q for q in queries if q["split"] == split and q["id"] in keep]
        rows += pooling.sheet_rows(sub, part, chunks, attribution, split, seed)
    for i, r in enumerate(rows, 1):
        r["row_id"] = i
    return rows


# ---- compute --------------------------------------------------------------------------------

def labels_from_sheet(rows: list[dict]) -> dict[tuple[str, str], int | None]:
    out: dict[tuple[str, str], int | None] = {}
    for r in rows:
        raw = (r.get("label") or "").strip()
        out[(r["query_id"], r["chunk_id"])] = int(raw) if raw in ("0", "1", "2") else None
    return out


def compute(subset: list[tuple[str, str]], owner: dict, partner: dict) -> tuple[str, dict | None]:
    """(message, result). result is None when kappa cannot be computed."""
    missing_owner = [k for k in subset if owner.get(k) is None]
    if missing_owner:
        return f"kappa not computed: {len(missing_owner)} owner labels missing", None
    missing_partner = [k for k in subset if partner.get(k) is None]
    if missing_partner:
        return f"kappa not computed: {len(missing_partner)} partner labels missing", None
    a = [owner[k] for k in subset]
    b = [partner[k] for k in subset]
    table = confusion(a, b)
    n = len(subset)
    res = {"n": n, "table": table, "agreement": sum(table[i][i] for i in range(3)) / n,
           "kappa": kappa(table), "kappa_linear_weighted": kappa(table, "linear")}
    return "ok", res


def fmt(v: float | None) -> str:
    return "undefined" if v is None else f"{v:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("select", "compute"))
    ap.add_argument("--partner", type=Path, default=PARTNER_SHEET,
                    help="the partner's filled CSV (default eval/sheets/kappa_partner.csv)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    pool = json.loads(POOL.read_text(encoding="utf-8"))

    if args.action == "select":
        from eval.qrels import load_queries
        from eval.run_eval import ENRICHED
        if (SUBSET.exists() or PARTNER_SHEET.exists()) and not args.force:
            print("the kappa subset already exists; the partner may be labelling it. Use --force.")
            return 2
        queries = load_queries()
        chunks = {c["chunk_id"]: c for c in map(json.loads, ENRICHED.read_text(
            encoding="utf-8").splitlines()) if c}
        attribution = json.loads((ROOT / "data" / "processed" / "attribution.json").read_text(
            encoding="utf-8"))
        subset = select(pool, queries, {c: v["source_type"] for c, v in chunks.items()})
        rows = partner_rows(subset, pool, queries, chunks, attribution)
        pooling.write_sheet(PARTNER_SHEET, rows)
        SUBSET.write_text(json.dumps({"seed": SEED, "n": len(subset),
                                      "eligible_source_types": list(ELIGIBLE),
                                      "pairs": [list(p) for p in sorted(subset)]}, indent=1)
                          + "\n", encoding="utf-8")
        by_split = {s: sum(1 for qid, _ in subset if qid in {q["id"] for q in queries
                                                             if q["split"] == s})
                    for s in ("dev", "test")}
        print(f"selected {len(subset)} judgments (dev {by_split['dev']}, test {by_split['test']}); "
              f"wrote {SUBSET.relative_to(ROOT)} and {PARTNER_SHEET.relative_to(ROOT)}")
        return 0

    if not args.partner.exists():
        print("kappa not computed: no partner file")
        return 0
    subset = [tuple(p) for p in json.loads(SUBSET.read_text(encoding="utf-8"))["pairs"]]
    owner: dict = {}
    for split in ("dev", "test"):
        sheet = SHEETS / f"judge_{split}.csv"
        if sheet.exists():
            owner.update(labels_from_sheet(pooling.read_sheet(sheet)))
    partner = labels_from_sheet(pooling.read_sheet(args.partner))
    msg, res = compute(subset, owner, partner)
    if res is None:
        print(msg)
        return 1
    print(f"n = {res['n']}  observed agreement {res['agreement']:.3f}")
    print(f"kappa (unweighted)      {fmt(res['kappa'])}")
    print(f"kappa (linear weighted) {fmt(res['kappa_linear_weighted'])}")
    print("rows = owner, columns = partner (labels 0, 1, 2):")
    for i, row in enumerate(res["table"]):
        print(f"  {i}: " + "  ".join(f"{x:>3}" for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
