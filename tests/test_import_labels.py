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


@pytest.mark.parametrize("bad", ["3", "-1", "1.0", "two", "x", "01"])
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
