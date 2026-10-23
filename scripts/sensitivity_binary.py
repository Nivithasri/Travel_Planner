"""Sensitivity analysis: collapse grade 2 into 1 and recompute the Phase 6 results (D98).

    python scripts/sensitivity_binary.py   # writes eval/ablation/sensitivity_binary.{md,json}

Why: the chat that labelled round 2 almost never gave grade 2 (D95 spot-check, D98), and nDCG@10
uses the grade as gain, so rows whose top 10 holds many round-2 chunks may be scored lower than a
human would score them. Collapsing 2 -> 1 for ALL judgments (round 1 and round 2) removes that grade
difference. Recall, P@5 and MRR use "relevant = grade >= 1" and are IDENTICAL after the collapse by
construction; they are still recomputed and shown so that is visible.

No retrieval is run. Rankings are the saved top 10 per query and row (eval/pool.json for the five
round-1 systems, eval/pool2.json for the other rows); they are checked against the saved test
per-query rankings (eval/ablation/test_merged_<row>.json) and the recomputed original nDCG@10 is
checked against the saved overall figures. The test judgments are read, as in the one-shot test run,
only to re-score a split that was already run. This is a post-test analysis: it changes no choice.

Significance: the same paired sign-flip test and Holm correction as the final table, over the same
11 pre-registered comparisons (scripts/test_run.py FAMILY): test 100,000 Monte Carlo flips, dev
exact.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import metrics as M  # noqa: E402
from eval import significance as S  # noqa: E402
from eval.qrels import load_qrels_merged, load_queries  # noqa: E402

ABL = ROOT / "eval" / "ablation"
N_MC, SEED = 100_000, 42
FROZEN_ROW = "dense_contextual"
FULL_ROW = "row9_crag"
ROWS = ["bm25_plain", "dense_plain", "bm25_contextual", "dense_contextual", "hybrid_plain",
        "hybrid_contextual", "rrf4", "dense_control", "bm25_control", "hybrid_control",
        "expanded_hybrid", "row5_filters", "row6_fresh_w10", "row6_fresh_w20", "row6_fresh_w30",
        "hybrid_filters", "row7_rerank", "hybrid_rerank", "row8_mmr", "row9_crag"]
# The 11 pre-registered comparisons of the final table (scripts/test_run.py FAMILY).
FAMILY = [
    ("bm25_plain", "dense_plain"), ("dense_plain", "dense_contextual"),
    ("hybrid_plain", "hybrid_contextual"), ("dense_plain", "hybrid_plain"),
    ("dense_contextual", "hybrid_contextual"), ("bm25_contextual", "dense_contextual"),
    ("dense_contextual", "row9_crag"), ("hybrid_contextual", "expanded_hybrid"),
    ("expanded_hybrid", "row7_rerank"), ("row7_rerank", "row8_mmr"), ("row8_mmr", "row9_crag"),
]
HEADING = ("Sensitivity analysis, run after the test split was seen; the frozen retrieve() choice "
           "is unchanged")


def collapse(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Grade 2 becomes 1; 0 stays 0."""
    return {q: {c: min(int(g), 1) for c, g in rels.items()} for q, rels in qrels.items()}


def binary_from_confusion(table: list[list[int]]) -> dict:
    """Relevant-or-not (grade >= 1) agreement and Cohen's kappa from a 3x3 confusion table
    (rows = labeller A, columns = labeller B). Computed from the table, not from the labels."""
    n = sum(map(sum, table))
    # 2x2: [[both 0, A0/B>=1], [A>=1/B0, both >= 1]]
    both0 = table[0][0]
    a0_b1 = sum(table[0][1:])
    a1_b0 = sum(table[i][0] for i in (1, 2))
    both1 = sum(table[i][j] for i in (1, 2) for j in (1, 2))
    po = (both0 + both1) / n
    a_pos, b_pos = (a1_b0 + both1) / n, (a0_b1 + both1) / n
    pe = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    return {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe != 1 else None,
            "table_2x2": [[both0, a0_b1], [a1_b0, both1]]}


def rankings(split_queries: list[dict]) -> dict[str, dict[str, list[str]]]:
    """{row: {qid: top-10 ids}} from the saved pools."""
    top1 = json.loads((ROOT / "eval" / "pool.json").read_text(encoding="utf-8"))["top"]
    top2 = json.loads((ROOT / "eval" / "pool2.json").read_text(encoding="utf-8"))["top"]
    out: dict[str, dict[str, list[str]]] = {r: {} for r in ROWS}
    for q in split_queries:
        merged = {**top1[q["id"]], **top2[q["id"]]}
        for row in ROWS:
            out[row][q["id"]] = list(merged[row])
    return out


def per_query(ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]],
              queries: list[dict]) -> list[dict]:
    rows = []
    for q in queries:
        r, qrel = ranked[q["id"]], qrels.get(q["id"], {})
        rows.append({"id": q["id"], "ndcg@10": M.ndcg_at_k(r, qrel, 10),
                     "mrr": M.reciprocal_rank(r, qrel), "p@5": M.precision_at_k(r, qrel, 5),
                     "recall@10": M.recall_at_k(r, qrel, 10)})
    return rows


def means(rows: list[dict]) -> dict:
    return {k: M.mean(r[k] for r in rows)[0] for k in ("ndcg@10", "mrr", "p@5", "recall@10")}


def chat_share(ranked: dict[str, list[str]], r2: dict[str, dict[str, int]]) -> float:
    """Share of top-10 slots (all queries) whose pair was judged in round 2 (by the chat)."""
    slots = [(q, c) for q, ids in ranked.items() for c in ids[:10]]
    return sum(1 for q, c in slots if c in r2.get(q, {})) / len(slots) if slots else 0.0


def comparisons(by_row: dict[str, list[dict]], n_mc: int) -> list[dict]:
    comps = [S.compare(f"{a} -> {b}",
                       {r["id"]: r["ndcg@10"] for r in by_row[a]},
                       {r["id"]: r["ndcg@10"] for r in by_row[b]}, n_mc=n_mc, seed=SEED)
             for a, b in FAMILY]
    return [{"comparison": c.name, "n": c.n, "delta": c.mean_diff, "p": c.p, "p_holm": h,
             "method": c.method} for c, h in S.holm_over(comps)]


def rank_order(values: dict[str, float | None]) -> list[str]:
    return [r for r, _ in sorted(values.items(), key=lambda kv: (-(kv[1] or 0), kv[0]))]


def significant(comps: list[dict]) -> set[str]:
    return {c["comparison"] for c in comps if c["p_holm"] < 0.05}


def analyse(split: str) -> dict:
    queries = load_queries(split)
    qrels = load_qrels_merged(split, allow_test=True)
    binary = collapse(qrels)
    r2_path = ROOT / "eval" / f"qrels_{split}_r2.json"
    r2 = json.loads(r2_path.read_text(encoding="utf-8"))
    ranked = rankings(queries)
    n_mc = N_MC if split == "test" else S.DEFAULT_MC
    res: dict = {"rows": {}, "split": split, "n_queries": len(queries)}
    pq_o, pq_b = {}, {}
    for row in ROWS:
        pq_o[row] = per_query(ranked[row], qrels, queries)
        pq_b[row] = per_query(ranked[row], binary, queries)
        res["rows"][row] = {"original": means(pq_o[row]), "collapsed": means(pq_b[row]),
                            "chat_labelled_share_of_top10": chat_share(ranked[row], r2)}
    res["comparisons_original"] = comparisons(pq_o, n_mc)
    res["comparisons_collapsed"] = comparisons(pq_b, n_mc)
    for key, name in (("original", "rank_order_original"), ("collapsed", "rank_order_collapsed")):
        res[name] = rank_order({r: res["rows"][r][key]["ndcg@10"] for r in ROWS})
    res["significant_original"] = sorted(significant(res["comparisons_original"]))
    res["significant_collapsed"] = sorted(significant(res["comparisons_collapsed"]))
    res["_ranked"] = ranked
    return res


def checks(res: dict) -> list[str]:
    """Reproduce what was saved: returns the list of mismatches (empty = all consistent)."""
    bad = []
    split = res["split"]
    if split == "test":
        for row in ROWS:
            saved = json.loads((ABL / f"test_merged_{row}.json").read_text(encoding="utf-8"))
            for pq in saved["per_query"]:
                if pq["ranked"][:10] != res["_ranked"][row][pq["id"]]:
                    bad.append(f"test {row} {pq['id']}: saved ranking differs from the pool")
            want = saved["groups"]["overall"]["ndcg@10"]
            if abs(want - res["rows"][row]["original"]["ndcg@10"]) > 1e-9:
                bad.append(f"test {row}: nDCG@10 {res['rows'][row]['original']['ndcg@10']} "
                           f"!= saved {want}")
        final = json.loads((ABL / "test_final.json").read_text(encoding="utf-8"))
        for c_new, c_old in zip(res["comparisons_original"], final["comparisons"]):
            if (c_new["comparison"] != c_old["comparison"]
                    or abs(c_new["p_holm"] - c_old["p_holm"]) > 1e-9):
                bad.append(f"test comparison {c_new['comparison']}: p_holm differs from "
                           "test_final.json")
    else:
        saved = json.loads((ABL / "dev_merged.json").read_text(encoding="utf-8"))["ndcg_merged"]
        for row, want in saved.items():
            if row in res["rows"] and abs(want - res["rows"][row]["original"]["ndcg@10"]) > 1e-9:
                bad.append(f"dev {row}: nDCG@10 differs from dev_merged.json")
    return bad


def round2_counts(frozen_top: dict, full_top: dict) -> dict:
    """Round-2 pairs whose FINAL label is 1 (after the recorded corrections), per split, and how
    many sit in the top 10 of the frozen row and of the full chain. Counts only."""
    out = {}
    for split in ("dev", "test"):
        r2 = json.loads((ROOT / "eval" / f"qrels_{split}_r2.json").read_text(encoding="utf-8"))
        ones = {(q, c) for q, rels in r2.items() for c, g in rels.items() if g == 1}
        labels = Counter(g for rels in r2.values() for g in rels.values())
        inside = lambda top: len({(q, c) for q, ids in top[split].items()  # noqa: E731
                                  for c in ids[:10]} & ones)
        out[split] = {"round2_pairs": sum(labels.values()), "final_label_counts":
                      {str(k): labels[k] for k in (0, 1, 2)}, "label_1": len(ones),
                      "label_1_in_top10_frozen_row": inside(frozen_top),
                      "label_1_in_top10_full_chain": inside(full_top)}
    return out


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def markdown(results: dict[str, dict], counts: dict, spot: dict) -> str:
    L = [f"# {HEADING}", "",
         "Grade 2 is collapsed into 1 for ALL judgments (round 1 human and round 2 chat), then "
         "nDCG@10, MRR, P@5 and Recall@10 are recomputed from the saved top-10 rankings (no "
         "retrieval re-run). MRR, P@5 and Recall use grade >= 1 and are identical by construction; "
         "only nDCG@10 can move. MRR here is computed on the saved top 10, whereas the final "
         "table's MRR comes from the top-20 run, so a row whose first relevant chunk sits at rank "
         "11 to 20 shows a slightly lower MRR here (bm25_plain on test: 0.692 against 0.699); the "
         "original/collapsed pair is identical either way. Recall@20 is not recomputed (only "
         "top-10 rankings are saved); it uses the same threshold, so it is unchanged too. Column "
         "`chat share` is the share of top-10 slots filled by a round-2 (chat-labelled) pair.", ""]
    for split in ("test", "dev"):
        r = results[split]
        L += [f"## {split} ({r['n_queries']} queries)", "",
              "| row | nDCG@10 original | nDCG@10 collapsed | delta | MRR | P@5 | Recall@10 | "
              "chat share |", "|---|---|---|---|---|---|---|---|"]
        for row in ROWS:
            o, c = r["rows"][row]["original"], r["rows"][row]["collapsed"]
            mark = " (FROZEN)" if row == FROZEN_ROW else ""
            L.append(f"| {row}{mark} | {f3(o['ndcg@10'])} | {f3(c['ndcg@10'])} | "
                     f"{c['ndcg@10'] - o['ndcg@10']:+.3f} | {f3(o['mrr'])} / {f3(c['mrr'])} | "
                     f"{f3(o['p@5'])} / {f3(c['p@5'])} | {f3(o['recall@10'])} / "
                     f"{f3(c['recall@10'])} | "
                     f"{r['rows'][row]['chat_labelled_share_of_top10']:.3f} |")
        L += ["", f"### {split}: paired nDCG@10 comparisons, original vs collapsed "
              f"({'100,000 Monte Carlo flips' if split == 'test' else 'exact'}, Holm over 11)", "",
              "| comparison | delta orig | p_holm orig | delta collapsed | p_holm collapsed |",
              "|---|---|---|---|---|"]
        for o, c in zip(r["comparisons_original"], r["comparisons_collapsed"]):
            L.append(f"| {o['comparison']} | {o['delta']:+.3f} | {o['p_holm']:.4f} | "
                     f"{c['delta']:+.3f} | {c['p_holm']:.4f} |")
        L += ["", f"Significant (p_holm < 0.05), original: {r['significant_original'] or 'none'}",
              f"Significant (p_holm < 0.05), collapsed: {r['significant_collapsed'] or 'none'}",
              f"Rank order by nDCG@10, original: {', '.join(r['rank_order_original'])}",
              f"Rank order by nDCG@10, collapsed: {', '.join(r['rank_order_collapsed'])}", ""]
    L += ["## Does any conclusion change?", ""]
    for split in ("test", "dev"):
        r = results[split]
        same_sig = r["significant_original"] == r["significant_collapsed"]
        same_order = r["rank_order_original"] == r["rank_order_collapsed"]
        L.append(f"* {split}: significant pairs {'unchanged' if same_sig else 'CHANGED'}; rank "
                 f"order of all {len(ROWS)} rows {'unchanged' if same_order else 'CHANGED'}.")
        so, sc = set(r["significant_original"]), set(r["significant_collapsed"])
        if not same_sig:
            L.append(f"  * only original: {sorted(so - sc) or 'none'}; "
                     f"only collapsed: {sorted(sc - so) or 'none'}")
        if not same_order:
            pairs = zip(r["rank_order_original"], r["rank_order_collapsed"])
            moved = [(i + 1, a, b) for i, (a, b) in enumerate(pairs) if a != b]
            L.append(f"  * positions that differ (position, original, collapsed): {moved}")
    L += ["", "## Reading (computed from the tables above)", ""]
    for split in ("test", "dev"):
        r = results[split]
        by = {c["comparison"]: c for c in r["comparisons_original"]}
        byc = {c["comparison"]: c for c in r["comparisons_collapsed"]}
        key = f"{FROZEN_ROW} -> {FULL_ROW}"
        lo = [v["collapsed"]["ndcg@10"] - v["original"]["ndcg@10"] for v in r["rows"].values()
              if v["chat_labelled_share_of_top10"] < 0.05]
        hi = [v["collapsed"]["ndcg@10"] - v["original"]["ndcg@10"] for v in r["rows"].values()
              if v["chat_labelled_share_of_top10"] >= 0.15]
        L.append(f"* {split}: frozen row to full chain, delta {by[key]['delta']:+.3f} "
                 f"(p_holm {by[key]['p_holm']:.4f}) originally, {byc[key]['delta']:+.3f} "
                 f"(p_holm {byc[key]['p_holm']:.4f}) collapsed. Mean nDCG@10 gain from collapsing: "
                 f"{sum(lo) / len(lo):+.3f} over the {len(lo)} rows with under 5% chat-labelled "
                 f"slots, {sum(hi) / len(hi):+.3f} over the {len(hi)} rows with 15% or more "
                 "(descriptive; those rows also differ in other ways).")
    L += ["", "## Round-2 label counts (final labels after the recorded corrections; counts only)",
          "", "| split | round-2 pairs | label 0 / 1 / 2 | label 1 in top 10 of frozen row | "
          "label 1 in top 10 of full chain |", "|---|---|---|---|---|"]
    for split in ("dev", "test"):
        c = counts[split]
        fl = c["final_label_counts"]
        L.append(f"| {split} | {c['round2_pairs']} | {fl['0']} / {fl['1']} / {fl['2']} | "
                 f"{c['label_1_in_top10_frozen_row']} | {c['label_1_in_top10_full_chain']} |")
    L += ["", "## Owner vs chat spot-check, binary view (computed from the confusion table)", "",
          f"Confusion (rows owner, columns chat): {spot['confusion']}. Relevant-or-not "
          f"(grade >= 1) "
          f"agreement {spot['binary']['agreement']:.3f} "
          f"({round(spot['binary']['agreement'] * spot['binary']['n'])} of {spot['binary']['n']}), "
          f"kappa {spot['binary']['kappa']:.3f}; 2x2 [[both 0, owner 0 / chat >= 1], [owner >= 1 / "
          f"chat 0, both >= 1]] = {spot['binary']['table_2x2']}.", ""]
    return "\n".join(L)


SPOT_CONFUSION = [[8, 3, 0], [1, 6, 0], [1, 11, 0]]  # rows owner, columns chat (D98)


def main() -> int:
    results = {s: analyse(s) for s in ("test", "dev")}
    bad = [b for s in results for b in checks(results[s])]
    if bad:
        print("CHECKS FAILED:\n" + "\n".join(bad))
        return 1
    counts = round2_counts({s: results[s]["_ranked"][FROZEN_ROW] for s in results},
                           {s: results[s]["_ranked"][FULL_ROW] for s in results})
    spot = {"confusion": SPOT_CONFUSION, "binary": binary_from_confusion(SPOT_CONFUSION)}
    md = markdown(results, counts, spot)
    for r in results.values():
        r.pop("_ranked")
    (ABL / "sensitivity_binary.md").write_text(md + "\n", encoding="utf-8")
    (ABL / "sensitivity_binary.json").write_text(json.dumps(
        {"heading": HEADING, "results": results, "round2_counts": counts, "spot_check": spot},
        indent=1, sort_keys=True), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
