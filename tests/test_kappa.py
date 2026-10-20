"""scripts/kappa.py: kappa against a hand-computed 3x3 table, subset selection, compute messages."""
import importlib.util
import json
from pathlib import Path

import pytest

from eval import pooling

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kappa_script", ROOT / "scripts" / "kappa.py")
kp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kp)

# rows = labeller A (0,1,2), columns = labeller B. N = 70.
TABLE = [[20, 5, 0], [4, 15, 6], [1, 3, 16]]
# By hand: po = 51/70, row sums 25 25 20, column sums 25 23 22, pe = 1640/4900 = 82/245,
# kappa = (51/70 - 82/245) / (1 - 82/245) = 193/326 = 0.592025.
# Linear weights w = 1, 1/2, 0: po_w = (51 + 18/2)/70 = 6/7;
# pe_w = (1640 + 2210/2)/4900 = 2745/4900;
# kappa_w = (4200 - 2745) / (4900 - 2745) = 1455/2155 = 0.675174.


def lists_from(table):
    a, b = [], []
    for i in range(3):
        for j in range(3):
            a += [i] * table[i][j]
            b += [j] * table[i][j]
    return a, b


def test_kappa_matches_hand_values():
    assert kp.kappa(TABLE) == pytest.approx(193 / 326)
    assert kp.kappa(TABLE) == pytest.approx(0.592025, abs=1e-6)
    assert kp.kappa(TABLE, "linear") == pytest.approx(1455 / 2155)
    assert kp.kappa(TABLE, "linear") == pytest.approx(0.675174, abs=1e-6)


def test_confusion_from_label_lists():
    a, b = lists_from(TABLE)
    assert kp.confusion(a, b) == TABLE
    with pytest.raises(ValueError):
        kp.confusion([0], [0, 1])


def test_kappa_extremes():
    assert kp.kappa([[5, 0, 0], [0, 5, 0], [0, 0, 5]]) == pytest.approx(1.0)
    assert kp.kappa([[0, 5, 0], [5, 0, 0], [0, 0, 0]]) == pytest.approx(-1.0)
    assert kp.kappa([[10, 0, 0], [0, 0, 0], [0, 0, 0]]) is None  # one label only: undefined
    assert kp.kappa([[0] * 3] * 3) is None


# ---- selection ------------------------------------------------------------------------------

def toy():
    queries = [{"id": f"Q{i}", "split": "dev" if i < 4 else "test", "text": f"q{i}"}
               for i in range(1, 9)]
    pool = {"pairs": {}}
    source = {}
    for q in queries:
        pool["pairs"][q["id"]] = {}
        for j in range(12):
            cid = f"{q['id']}c{j}"
            pool["pairs"][q["id"]][cid] = {"round": 1, "systems": ["bm25_plain"]}
            source[cid] = ["blog", "reference", "wikivoyage"][j % 3]
    return pool, queries, source


def test_select_excludes_blog_and_hits_n():
    pool, queries, source = toy()
    got = kp.select(pool, queries, source, n=20, seed=1)
    assert len(got) == 20 and len(set(got)) == 20
    assert all(source[c] in ("reference", "wikivoyage") for _, c in got)
    assert all(c in pool["pairs"][q] for q, c in got)


def test_select_is_stratified_and_spread():
    pool, queries, source = toy()
    got = kp.select(pool, queries, source, n=20, seed=1)
    split = {q["id"]: q["split"] for q in queries}
    assert sum(1 for q, _ in got if split[q] == "dev") == 8  # 4 of 8 queries, same eligible counts
    per_query = {q["id"]: sum(1 for x, _ in got if x == q["id"]) for q in queries}
    assert max(per_query.values()) - min(per_query.values()) <= 1  # spread evenly


def test_select_is_seeded():
    pool, queries, source = toy()
    assert kp.select(pool, queries, source, 20, 1) == kp.select(pool, queries, source, 20, 1)
    assert kp.select(pool, queries, source, 20, 1) != kp.select(pool, queries, source, 20, 2)


def test_select_small_pool_takes_everything_eligible():
    pool, queries, source = toy()
    got = kp.select(pool, queries, source, n=10_000, seed=1)
    assert len(got) == 8 * 8  # 8 queries x 8 non-blog chunks of 12


def test_partner_sheet_is_blind_and_unlabelled():
    pool, queries, source = toy()
    chunks = {c: {"chunk_id": c, "place": "Hampi", "state": "KA", "source_type": source[c],
                  "url": "https://x/" + c, "date_kind": "revised", "text": "t " + c,
                  "context_note": "SECRET-NOTE"} for c in source}
    subset = kp.select(pool, queries, source, n=20, seed=1)
    rows = kp.partner_rows(subset, pool, queries, chunks, {})
    assert len(rows) == 20 and [r["row_id"] for r in rows] == list(range(1, 21))
    assert set(rows[0]) == set(pooling.SHEET_COLUMNS)
    assert all(r["label"] == "" and r["note"] == "" for r in rows)
    blob = " ".join(str(v) for r in rows for v in r.values())
    assert "SECRET-NOTE" not in blob and "bm25" not in blob
    assert {r["source_type"] for r in rows} <= {"reference", "wikivoyage"}


# ---- compute --------------------------------------------------------------------------------

def test_compute_with_labels():
    a, b = lists_from(TABLE)
    subset = [("Q", str(i)) for i in range(70)]
    owner = {k: v for k, v in zip(subset, a)}
    partner = {k: v for k, v in zip(subset, b)}
    msg, res = kp.compute(subset, owner, partner)
    assert msg == "ok" and res["n"] == 70
    assert res["kappa"] == pytest.approx(193 / 326)
    assert res["kappa_linear_weighted"] == pytest.approx(1455 / 2155)
    assert res["agreement"] == pytest.approx(51 / 70)


def test_compute_refuses_missing_labels():
    subset = [("Q", "1"), ("Q", "2")]
    msg, res = kp.compute(subset, {("Q", "1"): 1, ("Q", "2"): None}, {("Q", "1"): 1})
    assert res is None and "owner labels missing" in msg
    msg, res = kp.compute(subset, {("Q", "1"): 1, ("Q", "2"): 0}, {("Q", "1"): 1})
    assert res is None and "partner labels missing" in msg


def test_labels_from_sheet_reads_only_valid_labels():
    rows = [{"query_id": "Q", "chunk_id": "a", "label": "2"},
            {"query_id": "Q", "chunk_id": "b", "label": ""},
            {"query_id": "Q", "chunk_id": "c", "label": "7"}]
    assert kp.labels_from_sheet(rows) == {("Q", "a"): 2, ("Q", "b"): None, ("Q", "c"): None}


def test_main_without_partner_file_says_not_computed(monkeypatch, tmp_path, capsys):
    (tmp_path / "pool.json").write_text(json.dumps({"pairs": {}}), encoding="utf-8")
    monkeypatch.setattr(kp, "POOL", tmp_path / "pool.json")
    assert kp.main(["compute", "--partner", str(tmp_path / "absent.csv")]) == 0
    assert "kappa not computed" in capsys.readouterr().out


def test_main_compute_end_to_end(monkeypatch, tmp_path, capsys):
    a, b = lists_from(TABLE)
    subset = [("Q1", f"c{i}") for i in range(70)]
    (tmp_path / "pool.json").write_text(json.dumps({"pairs": {}}), encoding="utf-8")
    (tmp_path / "subset.json").write_text(json.dumps({"pairs": [list(p) for p in subset]}),
                                          encoding="utf-8")

    def sheet(labels):
        return [{c: "" for c in pooling.SHEET_COLUMNS} | {"row_id": i + 1, "query_id": q,
                                                          "chunk_id": c, "label": str(v)}
                for i, ((q, c), v) in enumerate(zip(subset, labels))]

    sheets = tmp_path / "sheets"
    pooling.write_sheet(sheets / "judge_dev.csv", sheet(a))
    pooling.write_sheet(sheets / "kappa_partner.csv", sheet(b))
    monkeypatch.setattr(kp, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(kp, "SUBSET", tmp_path / "subset.json")
    monkeypatch.setattr(kp, "SHEETS", sheets)
    assert kp.main(["compute", "--partner", str(sheets / "kappa_partner.csv")]) == 0
    out = capsys.readouterr().out
    assert "0.592" in out and "0.675" in out
