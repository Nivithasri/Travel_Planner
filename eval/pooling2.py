"""Round-2 pooling for the Phase 6 variants (D91). Pure functions; scripts/pool2.py supplies the
variants. Nothing here reads relevance judgments of any kind.

Round 1 (Phase 4) pooled the top 10 of five baselines and judged 963 pairs (eval/pool.json). Every
Phase 6 variant retrieves chunks nobody has judged, and an unjudged chunk counts as 0, which biases
the table against the variant. Round 2 pools the top 10 of each new variant for all 40 queries and
keeps only the pairs that are not in eval/pool.json. The judged set is read from that pool file,
which lists exactly the pairs round 1 labelled, so no labels are needed to find the new pairs.

The judging sheets reuse the Phase 4 format (eval.pooling.sheet_rows): blind, shuffled, no
systems, ranks, scores or provenance.
"""
from __future__ import annotations

import statistics
from collections import Counter

DEPTH = 10
ROUND = 2


def judged_pairs(pool1: dict) -> dict[str, set[str]]:
    """{query_id: chunk ids round 1 judged}, from eval/pool.json."""
    return {qid: set(pairs) for qid, pairs in pool1["pairs"].items()}


def build_pool2(queries: list[dict], tops: dict[str, dict[str, list[str] | None]],
                judged: dict[str, set[str]], info: dict[str, dict], excluded: set[str]) -> dict:
    """{"top": tops, "pairs": {qid: {chunk_id: {...}}}, "gaps": {system: [qid, ...]}}.

    tops[qid][system] is that system's top-10 chunk ids, or None when the system could not run for
    that query (e.g. row 9 on a query whose rewrite has not been generated): the gap is reported,
    never filled."""
    pairs: dict[str, dict[str, dict]] = {}
    gaps: dict[str, list[str]] = {}
    for q in queries:
        qid = q["id"]
        pairs[qid] = {}
        for system, ids in tops[qid].items():
            if ids is None:
                gaps.setdefault(system, []).append(qid)
                continue
            for cid in ids[:DEPTH]:
                if cid in judged.get(qid, set()):
                    continue
                pair = pairs[qid].setdefault(cid, {"round": ROUND, "systems": []})
                if system not in pair["systems"]:
                    pair["systems"].append(system)
        for cid, pair in pairs[qid].items():
            pair["provenance"] = info[cid]["provenance"]
            pair["excluded_place"] = info[cid]["place"] in excluded
    return {"top": tops, "pairs": pairs, "gaps": gaps}


def counts_by_split(pool2: dict, queries: list[dict]) -> dict[str, int]:
    split = {q["id"]: q["split"] for q in queries}
    c: Counter = Counter()
    for qid, pairs in pool2["pairs"].items():
        c[split[qid]] += len(pairs)
    return {"dev": c["dev"], "test": c["test"], "total": c["dev"] + c["test"]}


def pairs_at_depth(pool2: dict, depth: int, judged: dict[str, set[str]]) -> dict[str, int]:
    """New pairs per query if only the first `depth` results of each system were pooled."""
    out = {}
    for qid, systems in pool2["top"].items():
        new = {cid for ids in systems.values() if ids for cid in ids[:depth]
               if cid not in judged.get(qid, set())}
        out[qid] = len(new)
    return out


def report(pool2: dict, queries: list[dict], judged: dict[str, set[str]],
           info: dict[str, dict]) -> str:
    qs = {q["id"]: q for q in queries}
    lines = []
    cnt = counts_by_split(pool2, queries)
    lines.append(f"new judgments to label (round 2): dev {cnt['dev']}, test {cnt['test']}, "
                 f"total {cnt['total']}  (round 1 judged 963)")
    sizes = [len(p) for p in pool2["pairs"].values()]
    lines.append(f"new pairs per query: min {min(sizes)}, median {statistics.median(sizes):g}, "
                 f"max {max(sizes)}")
    systems = sorted({s for tops in pool2["top"].values() for s in tops})
    lines += ["", f"{'system':<24}{'slots':>6}{'new':>6}{'new only here':>15}"
              f"{'unjudged@10':>13}"]
    for system in systems:
        slots = new = only = 0
        for qid, tops in pool2["top"].items():
            ids = tops.get(system)
            if not ids:
                continue
            for cid in ids[:DEPTH]:
                slots += 1
                if cid not in judged.get(qid, set()):
                    new += 1
                    only += pool2["pairs"][qid][cid]["systems"] == [system]
        lines.append(f"{system:<24}{slots:>6}{new:>6}{only:>15}"
                     f"{(new / slots if slots else 0.0):>13.3f}")
    pairs_all = [p for pairs in pool2["pairs"].values() for p in pairs.values()]
    prov = Counter(p["provenance"] for p in pairs_all)
    lines += ["", "new pairs by provenance: " + ", ".join(f"{k} {prov[k]}" for k in sorted(prov)),
              f"new pairs in D33-excluded places: {sum(p['excluded_place'] for p in pairs_all)} of "
              f"{len(pairs_all)}"]
    by_src = Counter(info[c]["source_type"] for pairs in pool2["pairs"].values() for c in pairs)
    lines.append("new pairs by source type: " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(by_src.items())))
    if pool2["gaps"]:
        lines += ["", "GAPS (a system that could not run, so the pool is incomplete for it):"]
        for system, qids in sorted(pool2["gaps"].items()):
            lines.append(f"  {system}: {len(qids)} queries ({', '.join(sorted(qids))})")
    lines += ["", "new pairs if the depth were reduced (NOT applied; a decision for the owner):"]
    for d in (10, 8, 7, 5):
        sz = pairs_at_depth(pool2, d, judged)
        dev = sum(v for qid, v in sz.items() if qs[qid]["split"] == "dev")
        test = sum(v for qid, v in sz.items() if qs[qid]["split"] == "test")
        lines.append(f"  depth {d:>2}: dev {dev}, test {test}, total {dev + test}")
    return "\n".join(lines)
