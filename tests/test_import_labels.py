"""scripts/import_labels.py: every failure mode, plus a clean import (toy sheets only)."""
import importlib.util
import json
from pathlib import Path

import pytest

from eval import pooling

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "import_labels", ROOT / "scripts" / "import_labels.py")
il = importlib.util.module_from_spec(spec)
spec.loader.exec_module(il)

QUERIES = [{"id": "Q1", "text": "one", "split": "dev"},
           {"id": "Q2", "text": "two", "split": "test"}]
POOL = {"pairs": {"Q1": {"c1": {}, "c2": {}, "c3": {}}, "Q2": {"c4": {}, "c5": {}}}}


def rows(split="dev", labels=("2", "1", "0")):
    ids = {"dev": ("Q1", "one", ["c1", "c2", "c3"]), "test": ("Q2", "two", ["c4", "c5"])}[split]
    return [{"row_id": str(i), "query_id": ids[0], "query_text": ids[1], "chunk_id": c,
             "label": labels[i - 1]} for i, c in enumerate(ids[2], 1)]


def check(r, split="dev"):
    return il.validate(r, POOL, QUERIES, split)


def test_clean_sheet_gives_qrels():
    qrels, problems = check(rows())
    assert problems == [] and qrels == {"Q1": {"c1": 2, "c2": 1, "c3": 0}}
    qrels, problems = check(rows("test", ("0", "2")), "test")
    assert problems == [] and qrels == {"Q2": {"c4": 0, "c5": 2}}


def test_blank_label():
    _, problems = check(rows(labels=("2", "", "0")))
    assert any("blank label" in p for p in problems)
    _, problems = check(rows(labels=("2", "  ", "0")))
    assert any("blank label" in p for p in problems)


@pytest.mark.parametrize("bad", ["3", "-1", "1.5", "2.1", "two", "x", "01", "1e0", "+1", "1.",
                                 "nan", "0.0.0", "1,0", "２"])
def test_invalid_label(bad):
    _, problems = check(rows(labels=("2", bad, "0")))
    assert any("is not 0, 1 or 2" in p for p in problems), problems


def test_unknown_row_id_and_gaps():
    r = rows()
    r[1]["row_id"] = "9"
    assert any("row_ids are not exactly" in p for p in check(r)[1])
    r = rows()
    r[1]["row_id"] = "abc"
    assert any("not a positive integer" in p for p in check(r)[1])
    r = rows()
    r[2]["row_id"] = "1"  # duplicate row_id
    assert any("row_ids are not exactly" in p for p in check(r)[1])


def test_missing_row():
    _, problems = check(rows()[:2])
    assert any("missing row for Q1/c3" in p for p in problems)
    assert not any("row_ids are not exactly" in p for p in problems)  # ids 1..2 are fine


def test_duplicate_pair():
    r = rows()
    r[2]["chunk_id"] = "c2"
    _, problems = check(r)
    assert any("duplicate row" in p for p in problems)
    assert any("missing row for Q1/c3" in p for p in problems)


def test_unknown_pair_and_wrong_split():
    r = rows()
    r[0]["chunk_id"] = "zzz"
    assert any("not in the dev pool" in p for p in check(r)[1])
    # a test query on the dev sheet
    r = rows()
    r[0]["query_id"] = "Q2"
    assert any("not in the dev pool" in p for p in check(r)[1])


def test_changed_query_text():
    r = rows()
    r[0]["query_text"] = "edited"
    assert any("query_text was changed" in p for p in check(r)[1])


def setup_files(monkeypatch, tmp_path, sheet_rows):
    (tmp_path / "pool.json").write_text(json.dumps(POOL), encoding="utf-8")
    full = [{c: "" for c in pooling.SHEET_COLUMNS} | r for r in sheet_rows]
    sheets = tmp_path / "sheets"
    pooling.write_sheet(sheets / "judge_dev.csv", full)
    monkeypatch.setattr(il, "POOL", tmp_path / "pool.json")
    monkeypatch.setattr(il, "SHEETS", sheets)
    monkeypatch.setattr(il, "qrels_path", lambda split: tmp_path / f"qrels_{split}.json")
    monkeypatch.setattr(il, "load_queries", lambda: QUERIES)
    return tmp_path / "qrels_dev.json"


def test_main_writes_then_refuses_to_overwrite_without_force(monkeypatch, tmp_path, capsys):
    out = setup_files(monkeypatch, tmp_path, rows())
    assert il.main(["dev"]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"Q1": {"c1": 2, "c2": 1, "c3": 0}}
    out.write_text('{"keep": {}}', encoding="utf-8")
    assert il.main(["dev"]) == 2
    assert out.read_text(encoding="utf-8") == '{"keep": {}}'  # untouched
    assert il.main(["dev", "--force"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["Q1"]["c1"] == 2
    assert "refusing" in capsys.readouterr().out


def test_main_writes_nothing_on_a_bad_sheet(monkeypatch, tmp_path):
    out = setup_files(monkeypatch, tmp_path, rows(labels=("2", "", "0")))
    assert il.main(["dev"]) == 1
    assert not out.exists()


def test_main_missing_sheet(monkeypatch, tmp_path):
    setup_files(monkeypatch, tmp_path, rows())
    (tmp_path / "sheets" / "judge_dev.csv").unlink()
    assert il.main(["dev"]) == 2


@pytest.mark.parametrize("raw,want", [("0", 0), ("1", 1), ("2", 2), ("0.0", 0), ("1.0", 1),
                                      ("2.0", 2), (" 2.0 ", 2), ("1.00", 1)])
def test_floats_exactly_0_1_2_are_converted_to_int(raw, want):
    qrels, problems = check(rows(labels=(raw, "1", "0")))
    assert problems == []
    got = qrels["Q1"]["c1"]
    assert got == want and type(got) is int


def test_float_sheet_imports_as_ints():
    qrels, problems = check(rows(labels=("2.0", "1.0", "0.0")))
    assert problems == [] and qrels == {"Q1": {"c1": 2, "c2": 1, "c3": 0}}
    assert all(type(v) is int for v in qrels["Q1"].values())


def test_extra_rubric_column_is_ignored():
    r = rows()
    for x in r:
        x["rubric"] = "anything, even a different text"
        x["note"] = "free text"
    qrels, problems = check(r)
    assert problems == [] and qrels == {"Q1": {"c1": 2, "c2": 1, "c3": 0}}
    for x in r:
        del x["rubric"]  # and its absence is fine too
    assert check(r)[1] == []


def test_parse_label_directly():
    assert pooling.parse_label("2.0") == 2 and pooling.parse_label(None) is None
    assert pooling.parse_label("") is None and pooling.parse_label("2.5") is None


def export(n=3):
    return [{c: f"{c}{i}" for c in pooling.FIXED_COLUMNS} | {"row_id": str(i), "label": "",
            "note": "", "rubric": "r"} for i in range(1, n + 1)]


def test_verify_unchanged_accepts_label_note_rubric_changes():
    orig = export()
    lab = [dict(r) for r in orig]
    for r in lab:
        r["label"], r["note"], r["rubric"] = "1.0", "n", "other"
    assert pooling.verify_unchanged(lab, orig) == []
    lab.reverse()  # order does not matter, row_ids do
    assert pooling.verify_unchanged(lab, orig) == []


@pytest.mark.parametrize("column", list(pooling.FIXED_COLUMNS))
def test_verify_unchanged_flags_every_fixed_column(column):
    orig = export()
    lab = [dict(r) for r in orig]
    if column == "row_id":
        lab[0]["row_id"] = "99"
    else:
        lab[0][column] = "edited"
    assert pooling.verify_unchanged(lab, orig)


def test_verify_unchanged_flags_missing_extra_and_duplicate_rows():
    orig = export()
    assert any("missing" in p for p in pooling.verify_unchanged(orig[:2], orig))
    assert pooling.verify_unchanged(orig + [dict(orig[0])], orig)  # duplicate
    assert any("not in the exported" in p for p in pooling.verify_unchanged(
        orig[:2] + [dict(orig[2], row_id="7")], orig))


def test_main_checks_a_renamed_sheet_against_the_original(monkeypatch, tmp_path):
    out = setup_files(monkeypatch, tmp_path, rows())  # writes judge_dev.csv (the "original")
    orig = pooling.read_sheet(tmp_path / "sheets" / "judge_dev.csv")
    lab = [dict(r) for r in orig]
    for r in lab:
        r["label"] = "1.0"
    pooling.write_sheet(tmp_path / "sheets" / "judge_dev_labelled.csv", lab)
    sheet = str(tmp_path / "sheets" / "judge_dev_labelled.csv")
    assert il.main(["dev", "--sheet", sheet]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"Q1": {"c1": 1, "c2": 1, "c3": 1}}
    lab[0]["text"] = "tampered"
    pooling.write_sheet(tmp_path / "sheets" / "judge_dev_labelled.csv", lab)
    out.unlink()
    assert il.main(["dev", "--sheet", sheet]) == 1
    assert not out.exists()
