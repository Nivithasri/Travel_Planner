"""Phase 9 tables (eval/phase9/report.py) and the record store and live guards (store.py)."""
import pytest

from eval.phase9 import report as R
from eval.phase9 import store as ST

DIMS = ("feasibility", "groundedness")


def table(a_scores, b_scores, a="full", b="plain_rag"):
    t = {}
    for i, (x, y) in enumerate(zip(a_scores, b_scores, strict=True), 1):
        t[(f"P{i:02d}", a)] = {"feasibility": x, "groundedness": x}
        t[(f"P{i:02d}", b)] = {"feasibility": y, "groundedness": y}
    return t


def test_per_system_and_means():
    by = R.per_system(table([5, 4], [3, 2]))
    assert by["full"]["feasibility"] == {"P01": 5, "P02": 4}
    rows = R.means(by, DIMS)
    assert rows[0] == {"system": "full", "n": 2, "feasibility": 4.5, "groundedness": 4.5}
    assert rows[1]["feasibility"] == 2.5


def test_compare_pairs_gives_mean_difference_exact_p_and_holm():
    by = R.per_system(table([4] * 10, [3] * 10))
    rows = R.compare_pairs(by, [("full", "plain_rag")], ("feasibility",), n_mc=1000)
    r = rows[0]
    assert r["n"] == 10 and r["mean_diff"] == 1.0 and r["ci"] == [1.0, 1.0]
    assert r["p"] == pytest.approx(2 / 1024, abs=1e-4) and r["p_holm"] == r["p"]


def test_holm_is_applied_over_the_whole_family():
    t = table([4] * 10, [3] * 10)
    by = R.per_system(t)
    rows = R.compare_pairs(by, [("full", "plain_rag")], DIMS, n_mc=1000)
    assert len(rows) == 2 and rows[0]["p_holm"] == pytest.approx(min(1, rows[0]["p"] * 2), abs=1e-3)


def test_no_difference_is_not_significant():
    by = R.per_system(table([3, 4, 5, 3], [3, 4, 5, 3]))
    r = R.compare_pairs(by, [("full", "plain_rag")], ("feasibility",))[0]
    assert r["mean_diff"] == 0.0 and r["p"] == 1.0


def live_rec(rid, status="ok", cited=1, date="2026-10-24", **counts):
    live = {"status": status, "searches": 2, "results": 10, "in_scope": 8, "in_plan": 3,
            "dropped_excluded": 1, "merged": 3, "cited": cited, "not_cited": 3 - cited, **counts}
    return {"request_id": rid, "arm": "full_live", "meta": {"utc_date": date}, "live": live,
            "critic": "passed", "timings": {}}


def test_live_summary_counts_status_totals_and_the_one_day_rule():
    s = R.live_summary([live_rec("P01"), live_rec("P02", status="empty", cited=0),
                        {"request_id": "P03", "arm": "full_corpus", "meta": {}}])
    assert s["requests"] == 2 and s["status"] == {"ok": 1, "empty": 1} and s["same_day"]
    assert s["totals"]["searches"] == 4 and s["totals"]["cited"] == 1
    assert s["requests_with_live_cited"] == 1
    assert not R.live_summary([live_rec("P01"), live_rec("P02", date="2026-10-25")])["same_day"]


def test_percentile_is_nearest_rank():
    v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert R.percentile(v, 90) == 9 and R.percentile(v, 50) == 5 and R.percentile(v, 100) == 10
    assert R.percentile([], 90) is None and R.percentile([4], 90) == 4


def test_latency_table_has_nodes_total_and_limiter_wait():
    recs = [{"arm": "latency_corpus", "timings": {"write": w, "parse": 1.0},
             "meta": {"wall_s": w + 1.5, "limiter_wait_s": 0.5 * i}}
            for i, w in enumerate([2.0, 4.0, 6.0], 1)]
    rows = {r["node"]: r for r in R.latency_table(recs)}
    assert rows["write"]["median_s"] == 4.0 and rows["write"]["p90_s"] == 6.0
    assert rows["total"]["median_s"] == 5.5 and rows["limiter_wait"]["median_s"] == 1.0
    assert rows["parse"]["n"] == 3


def test_case_candidates_return_ids_and_reasons_but_never_text():
    corpus = {"request_id": "P01", "arm": "full_corpus", "critic": "failed_rules",
              "meta": {}, "text": "SECRET ANSWER TEXT", "timings": {}}
    live = live_rec("P02", cited=2)
    t = {("P01", "full"): {"feasibility": 1, "groundedness": 1},
         ("P02", "full_live"): {"feasibility": 5}, ("P02", "full_corpus"): {"feasibility": 3}}
    c = R.case_candidates([corpus, live], t)
    assert {x["request_id"] for x in c["failures"]} == {"P01"}
    assert c["live_cases"] == [{"request_id": "P02", "live_cited": 2, "scores_changed": True}]
    assert "SECRET" not in str(c)


def test_md_table_renders_none_as_blank():
    out = R.md_table([{"a": 1, "b": None}], ["a", "b"])
    assert out.splitlines()[0] == "| a | b |" and out.splitlines()[2] == "| 1 |  |"


# ---- store ------------------------------------------------------------------------------------

def test_a_record_is_never_overwritten_without_force(tmp_path):
    rec = {"arm": "llm_only", "request_id": "P01", "text": "one"}
    p = ST.save_record(rec, tmp_path)
    with pytest.raises(ST.RecordExists):
        ST.save_record({**rec, "text": "two"}, tmp_path)
    assert ST.load_arm("llm_only", root=tmp_path)[0]["text"] == "one"
    ST.save_record({**rec, "text": "two"}, tmp_path, force=True)
    assert ST.load_arm("llm_only", root=tmp_path)[0]["text"] == "two" and p.exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_load_arm_is_sorted_and_filterable(tmp_path):
    for rid in ("P03", "P01", "P02"):
        ST.save_record({"arm": "plain_rag", "request_id": rid}, tmp_path)
    assert [r["request_id"] for r in ST.load_arm("plain_rag", root=tmp_path)] == [
        "P01", "P02", "P03"]
    assert [r["request_id"] for r in ST.load_arm("plain_rag", ["P02"], tmp_path)] == ["P02"]
    assert ST.load_arm("full_live", root=tmp_path) == []


def test_line_count(tmp_path):
    p = tmp_path / "log.jsonl"
    assert ST.line_count(p) == 0
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert ST.line_count(p) == 3


def guard(**over):
    kw = {"terms_status": "ok", "key_present": True, "ledger": {"credits": 56},
          "planned_searches": 20, "credits_per_search": 2, "credit_cap": 40,
          "monthly_stop_at": 900, "existing_dates": [], "today": "2026-10-24"}
    kw.update(over)
    return ST.live_guard(**kw)


def test_the_live_guard_allows_a_clean_day_and_names_every_reason_otherwise():
    assert guard() == []
    assert any("not allowed" in p for p in guard(terms_status="not_verified"))
    assert any("no live search key" in p for p in guard(key_present=False))
    assert any("above the cap" in p for p in guard(planned_searches=30))
    assert any("monthly stop" in p for p in guard(ledger={"credits": 880}))
    assert any("another day" in p for p in guard(existing_dates=["2026-10-23"]))
    assert guard(existing_dates=["2026-10-24"]) == []
    assert guard(ledger=None) == []
