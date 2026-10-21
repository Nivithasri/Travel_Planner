"""Round-2 pooling (eval/pooling2.py, scripts/pool2.py), the merged qrels loader and the round-2
label import (D91). Synthetic data only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eval import pooling, pooling2, qrels

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [{"id": "Q1", "split": "dev", "text": "one"},
           {"id": "Q2", "split": "test", "text": "two"}]
INFO = {c: {"provenance": p, "place": pl, "source_type": s} for c, p, pl, s in (
    ("a", "llm", "Hampi", "blog"), ("b", "rule_only", "Munnar", "reference"),
    ("c", "fallback", "Hampi", "wikivoyage"), ("d", "llm", "Kochi", "blog"))}
JUDGED = {"Q1": {"a"}, "Q2": set()}


def pool(tops):
    return pooling2.build_pool2(QUERIES, tops, JUDGED, INFO, {"Munnar"})


def test_only_pairs_round_1_never_judged_are_new():
    p = pool({"Q1": {"s1": ["a", "b"], "s2": ["b", "c"]}, "Q2": {"s1": ["a"], "s2": ["d"]}})
    assert sorted(p["pairs"]["Q1"]) == ["b", "c"]  # a is judged for Q1
    assert sorted(p["pairs"]["Q2"]) == ["a", "d"]  # judged status is per query
    assert p["pairs"]["Q1"]["b"]["systems"] == ["s1", "s2"]


def test_pairs_carry_provenance_and_the_excluded_place_flag_but_no_label():
    pair = pool({"Q1": {"s": ["b"]}, "Q2": {"s": []}})["pairs"]["Q1"]["b"]
    assert pair == {"round": 2, "systems": ["s"], "provenance": "rule_only",
                    "excluded_place": True}


def test_only_the_first_ten_of_a_system_are_pooled():
    ids = [f"x{i}" for i in range(15)]
    info = {i: {"provenance": "llm", "place": "P", "source_type": "blog"} for i in ids}
    p = pooling2.build_pool2(QUERIES[:1], {"Q1": {"s": ids}}, {"Q1": set()}, info, set())
    assert len(p["pairs"]["Q1"]) == pooling2.DEPTH == 10


def test_a_system_that_could_not_run_is_a_reported_gap_not_a_silent_skip():
    p = pool({"Q1": {"s1": ["b"], "row9": None}, "Q2": {"s1": ["d"], "row9": None}})
    assert p["gaps"] == {"row9": ["Q1", "Q2"]}
    assert sorted(p["pairs"]["Q1"]) == ["b"]
    assert "GAPS" in pooling2.report(p, QUERIES, JUDGED, INFO)


def test_counts_and_the_depth_table():
    p = pool({"Q1": {"s": ["b", "c"]}, "Q2": {"s": ["a", "d"]}})
    assert pooling2.counts_by_split(p, QUERIES) == {"dev": 2, "test": 2, "total": 4}
    assert pooling2.pairs_at_depth(p, 1, JUDGED) == {"Q1": 1, "Q2": 1}


def test_report_lists_unjudged_share_and_unique_contributions():
    p = pool({"Q1": {"s1": ["a", "b"], "s2": ["b"]}, "Q2": {"s1": ["d"], "s2": []}})
    text = pooling2.report(p, QUERIES, JUDGED, INFO)
    assert "new judgments to label (round 2): dev 1, test 1, total 2" in text
    s1 = next(ln for ln in text.splitlines() if ln.startswith("s1"))
    # 3 slots (a, b, d), 2 new (b, d), 1 new only here (d; b is also in s2)
    assert s1.split()[1:4] == ["3", "2", "1"]
    assert "depth  5" in text


def test_the_sheets_are_blind_and_deduplicated():
    p = pool({"Q1": {"s1": ["b", "c"], "s2": ["b"]}, "Q2": {"s1": ["d"]}})
    chunks = {c: {"place": INFO[c]["place"], "state": "KL", "source_type": INFO[c]["source_type"],
                  "url": f"https://x/{c}", "date_kind": "none", "text": f"text {c}",
                  "context_note": "SECRET NOTE"} for c in "abcd"}
    rows = pooling.sheet_rows(p, QUERIES, chunks, {}, "dev", 43)
    assert sorted(r["chunk_id"] for r in rows) == ["b", "c"]  # b once although two systems
    flat = json.dumps(rows)
    assert "SECRET NOTE" not in flat and "s1" not in flat and "rule_only" not in flat


# ---- the merged qrels loader --------------------------------------------------------------------

def write(d, name, obj):
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_round_two_never_overrides_round_one(tmp_path):
    write(tmp_path, "qrels_dev.json", {"Q1": {"a": 2, "b": 0}})
    write(tmp_path, "qrels_dev_r2.json", {"Q1": {"b": 2, "c": 1}, "Q9": {"z": 1}})
    merged = qrels.load_qrels_merged("dev", directory=tmp_path)
    assert merged == {"Q1": {"a": 2, "b": 0, "c": 1}, "Q9": {"z": 1}}  # b keeps its round-1 label


def test_without_a_round_two_file_the_merge_is_round_one(tmp_path):
    write(tmp_path, "qrels_dev.json", {"Q1": {"a": 2}})
    assert qrels.load_qrels_merged("dev", directory=tmp_path) == {"Q1": {"a": 2}}


def test_the_merged_loader_still_refuses_the_test_split_without_the_flag(tmp_path):
    write(tmp_path, "qrels_test.json", {"Q2": {"a": 1}})
    with pytest.raises(qrels.TestQrelsRefused):
        qrels.load_qrels_merged("test", directory=tmp_path)
    assert qrels.load_qrels_merged("test", allow_test=True, directory=tmp_path) == {"Q2": {"a": 1}}


def test_a_bad_round_two_label_is_refused(tmp_path):
    write(tmp_path, "qrels_dev.json", {"Q1": {"a": 2}})
    write(tmp_path, "qrels_dev_r2.json", {"Q1": {"b": 5}})
    with pytest.raises(ValueError):
        qrels.load_qrels_merged("dev", directory=tmp_path)


# ---- the round-2 import --------------------------------------------------------------------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_round_two_collisions_are_found():
    imp = load_script("import_labels")
    pool1 = {"pairs": {"Q1": {"a": {}, "b": {}}}}
    assert imp.round2_collisions({"pairs": {"Q1": {"b": {}, "c": {}}}}, pool1) == [("Q1", "b")]
    assert imp.round2_collisions({"pairs": {"Q1": {"c": {}}}}, pool1) == []


def test_the_round_two_pool_validates_like_round_one():
    imp = load_script("import_labels")
    rows = [{"row_id": "1", "query_id": "Q1", "query_text": "one", "chunk_id": "b", "label": "2"},
            {"row_id": "2", "query_id": "Q1", "query_text": "one", "chunk_id": "c", "label": "0.0"}]
    p = {"pairs": {"Q1": {"b": {}, "c": {}}}}
    got, problems = imp.validate(rows, p, QUERIES, "dev")
    assert problems == [] and got == {"Q1": {"b": 2, "c": 0}}
    _, problems = imp.validate(rows[:1], p, QUERIES, "dev")
    assert any("missing row" in x for x in problems)


# ---- the owner's rule: round-2 code never opens the test qrels -----------------------------------

@pytest.mark.parametrize("rel", ["eval/pooling2.py", "scripts/pool2.py"])
def test_pooling_code_never_references_relevance_judgments(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for forbidden in ("qrels_test", "load_qrels", "allow_test", "qrels_dev"):
        assert forbidden not in src, f"{rel} mentions {forbidden}"


def test_a_single_split_can_be_written_without_touching_the_other_sheet(tmp_path, monkeypatch):
    # an owner may already be labelling the dev sheet: `sheets --only test` must leave it alone
    pool2s = load_script("pool2")
    written = []
    monkeypatch.setattr(pool2s, "SHEETS", tmp_path)
    monkeypatch.setattr(pool2s, "ATTRIBUTION", tmp_path / "attr.json")
    (tmp_path / "attr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "chunks.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(pool2s, "ENRICHED", tmp_path / "chunks.jsonl")
    monkeypatch.setattr(pool2s.pooling, "sheet_rows", lambda *a, **k: [{"x": 1}])
    monkeypatch.setattr(pool2s.pooling, "write_sheet",
                        lambda path, rows: written.append(Path(path).name))
    out = pool2s.write_sheets({"gaps": {}}, QUERIES, only="test")
    assert written == ["judge2_test.csv"] and out["dev"].startswith("left as it is")
