"""Batches for a chat-based labeller and the merge of its replies (scripts/label_batches.py).
Synthetic sheets only; no real label or chunk text is involved."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from eval import pooling

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lb = load_script("label_batches")
imp = load_script("import_labels")

# (query id, source type) per row, in sheet order; row_id = position (1-based)
SPEC = [("Q1", "wikivoyage"), ("Q1", "reference"), ("Q1", "wikivoyage"),
        ("Q2", "wikivoyage"), ("Q2", "blog"),
        ("Q3", "reference"), ("Q3", "blog"), ("Q3", "wikivoyage"), ("Q3", "wikivoyage")]
TEXTS = {"Q1": "first query", "Q2": "second query", "Q3": "third query"}


def toy_rows():
    rows = []
    for i, (qid, src) in enumerate(SPEC, 1):
        rows.append({"row_id": str(i), "query_id": qid, "query_text": TEXTS[qid],
                     "chunk_id": f"CHUNK{i}", "place": "Hampi", "state": "KA",
                     "source_type": src, "title": f"Title {i}", "url": f"https://secret.test/{i}",
                     "date_kind": "revised", "text": f"Line one of {i}.\nLine two  of {i}.",
                     "label": "", "note": "", "rubric": "RUBRIC-SENTINEL"})
    return rows


@pytest.fixture
def sheets(tmp_path):
    d = tmp_path / "sheets"
    d.mkdir()
    pooling.write_sheet(d / "judge2_dev.csv", toy_rows())
    return d


def reply(*pairs, unsure="none"):
    return "\n".join(f"{rid}|{label}|" for rid, label in pairs) + f"\nUNSURE: {unsure}\n"


def put_reply(sheets, name, text):
    (sheets / "replies").mkdir(exist_ok=True)
    (sheets / "replies" / name).write_text(text, encoding="utf-8")


def full_replies(sheets, size=5, **kw):
    """Replies for every row of the toy sheet, label = (row_id % 3)."""
    m = lb.export_split("dev", sheets, size, **kw)
    for i, (name, b) in enumerate(m["batches"].items(), 1):
        put_reply(sheets, f"dev_{i}.txt", reply(*[(r, int(r) % 3) for r in b["row_ids"]]))
    return m


def merge(sheets, **kw):
    return lb.merge_split("dev", sheets, sheets / "prov.json", **kw)


# ---- batching ------------------------------------------------------------------------------

def test_a_query_is_never_split_and_every_row_appears_once():
    rows = toy_rows()
    for size in (1, 2, 3, 4, 5, 6, 9, 25):
        batches = lb.make_batches(rows, size)
        flat = [r["row_id"] for b in batches for r in b]
        assert sorted(flat, key=int) == [r["row_id"] for r in rows]
        for q in TEXTS:
            assert sum(1 for b in batches if any(r["query_id"] == q for r in b)) == 1


def test_queries_are_packed_up_to_the_size_and_an_oversize_query_stands_alone():
    batches = lb.make_batches(toy_rows(), 5)
    assert [len(b) for b in batches] == [5, 4]  # Q1+Q2, then Q3
    batches = lb.make_batches(toy_rows(), 3)
    assert [[r["query_id"] for r in b][0] for b in batches] == ["Q1", "Q2", "Q3"]
    assert [len(b) for b in batches] == [3, 2, 4]  # Q3 has 4 rows > 3 and is not cut


def test_a_batch_shows_only_the_judging_columns():
    rows = toy_rows()[:2]
    for r in rows:
        r["label"], r["note"] = "LABEL-SENTINEL", "NOTE-SENTINEL"
    text = lb.format_batch("dev_batch_01", rows)
    for secret in ("CHUNK1", "secret.test", "RUBRIC-SENTINEL", "LABEL-SENTINEL",
                   "NOTE-SENTINEL"):
        assert secret not in text
    assert "[row 1] query Q1" in text and "QUERY: first query" in text
    assert "TEXT: Line one of 1. Line two of 1." in text  # whitespace collapsed to one line
    assert "SOURCE: wikivoyage" in text and "DATE_KIND: revised" in text
    assert "Reply with ONLY the lines row_id|label|note for these 2 rows" in text


def test_export_writes_batches_and_a_manifest_and_removes_stale_batches(sheets):
    m = lb.export_split("dev", sheets, 3)
    d = sheets / "batches"
    assert sorted(p.name for p in d.glob("dev_batch_*.txt")) == [
        "dev_batch_01.txt", "dev_batch_02.txt", "dev_batch_03.txt"]
    assert m["sheet_sha256"] == lb.sha256_file(sheets / "judge2_dev.csv")
    assert json.loads((d / "dev_manifest.json").read_text())["batches"]["dev_batch_03"][
        "row_ids"] == ["6", "7", "8", "9"]
    lb.export_split("dev", sheets, 25)  # a smaller number of batches must not leave old files
    assert [p.name for p in d.glob("dev_batch_*.txt")] == ["dev_batch_01.txt"]


def test_exclude_blog_leaves_blog_rows_out_of_the_batches(sheets):
    m = lb.export_split("dev", sheets, 25, exclude_blog=True)
    assert m["excluded_row_ids"] == ["5", "7"]
    sent = [r for b in m["batches"].values() for r in b["row_ids"]]
    assert "5" not in sent and "7" not in sent and len(sent) == 7
    assert "blog" not in (sheets / "batches" / "dev_batch_01.txt").read_text()


# ---- parsing a reply -----------------------------------------------------------------------

def test_parse_reply_reads_rows_notes_and_unsure():
    text = ("```\nrow_id|label|note\nREADY\n1|2|rule 1: other temple\n2|0|\n3|1\n"
            "UNSURE: 3\n```\n")
    entries, unsure, problems = lb.parse_reply(text)
    assert problems == []
    assert entries == [("1", 2, "rule 1: other temple"), ("2", 0, ""), ("3", 1, "")]
    assert unsure == {"3"}


@pytest.mark.parametrize("label", ["3", "1.5", "two", "", "-1", "01", "1e0"])
def test_bad_labels_are_reported_not_guessed(label):
    entries, _, problems = lb.parse_reply(f"1|{label}|\n")
    assert entries == [] and len(problems) == 1 and "not 0, 1 or 2" in problems[0]


def test_floats_are_accepted_only_as_exactly_0_1_2():
    entries, _, problems = lb.parse_reply("1|2.0|\n2|0.0|\n3|1.0|\n")
    assert [e[1] for e in entries] == [2, 0, 1] and problems == []


def test_unrecognised_lines_and_pipes_in_notes_are_reported():
    _, _, problems = lb.parse_reply("Sure! Here are the labels:\n1|2|a|b\nabc|1|\n")
    assert len(problems) == 3
    assert "not a row line" in problems[0] and "contains a |" in problems[1]


def test_unsure_none_empty_and_bad_tokens():
    assert lb.parse_reply("UNSURE: none\n")[1] == set()
    assert lb.parse_reply("unsure:\n")[1] == set()
    assert lb.parse_reply("UNSURE: 4, 9 12\n")[1] == {"4", "9", "12"}
    assert "not a row id" in lb.parse_reply("UNSURE: x7\n")[2][0]


# ---- merge: the happy path -----------------------------------------------------------------

def test_merge_fills_labels_and_changes_nothing_else(sheets):
    full_replies(sheets)
    rep, problems = merge(sheets)
    assert problems == [] and rep["llm"] == 9 and rep["hand"] == 0
    out = pooling.read_sheet(sheets / "judge2_dev_labelled.csv")
    assert [r["label"] for r in out] == [str(i % 3) for i in range(1, 10)]
    assert pooling.verify_unchanged(out, pooling.read_sheet(sheets / "judge2_dev.csv")) == []
    assert rep["llm_distribution"] == {0: 3, 1: 3, 2: 3}


def test_the_merged_file_passes_the_real_importer_validation(sheets):
    full_replies(sheets)
    merge(sheets)
    rows = pooling.read_sheet(sheets / "judge2_dev_labelled.csv")
    pool = {"pairs": {q: {r["chunk_id"]: {} for r in toy_rows() if r["query_id"] == q}
                      for q in TEXTS}}
    queries = [{"id": q, "text": t, "split": "dev"} for q, t in TEXTS.items()]
    qrels, problems = imp.validate(rows, pool, queries, "dev")
    assert problems == [] and qrels["Q1"] == {"CHUNK1": 1, "CHUNK2": 2, "CHUNK3": 0}


def test_unsure_rows_are_marked_in_the_note(sheets):
    lb.export_split("dev", sheets, 25)
    put_reply(sheets, "dev_all.txt",
              "\n".join(f"{i}|1|{'torn 1/2' if i == 2 else ''}" for i in range(1, 10))
              + "\nUNSURE: 2, 5\n")
    rep, problems = merge(sheets)
    out = {r["row_id"]: r for r in pooling.read_sheet(sheets / "judge2_dev_labelled.csv")}
    assert problems == [] and rep["unsure"] == 2
    assert out["2"]["note"] == "torn 1/2 [unsure]" and out["5"]["note"] == " [unsure]"
    assert out["1"]["note"] == ""


def test_provenance_records_ids_and_labels_but_no_text(sheets):
    full_replies(sheets)
    merge(sheets, assessor="Claude chat, test")
    doc = json.loads((sheets / "prov.json").read_text(encoding="utf-8"))["dev"]
    assert doc["assessor_llm"] == "Claude chat, test" and doc["hand"] == []
    assert doc["llm"] == {str(i): i % 3 for i in range(1, 10)}
    blob = json.dumps(doc)
    assert "Line one" not in blob and "CHUNK" not in blob and "secret" not in blob


# ---- merge: refuses and writes nothing ------------------------------------------------------

def refused(sheets):
    rep, problems = merge(sheets)
    assert rep == {} and problems
    assert not (sheets / "judge2_dev_labelled.csv").exists()
    assert not (sheets / "prov.json").exists()
    return problems


def test_a_missing_answer_is_named_by_batch(sheets):
    m = lb.export_split("dev", sheets, 5)
    first = m["batches"]["dev_batch_01"]["row_ids"]
    put_reply(sheets, "dev_1.txt", reply(*[(r, 1) for r in first]))
    problems = refused(sheets)
    assert any("dev_batch_02: no reply for 4 rows" in p for p in problems)


def test_a_partially_answered_batch_is_missing_exactly_the_unanswered_rows(sheets):
    m = lb.export_split("dev", sheets, 25)
    put_reply(sheets, "dev_1.txt", reply(*[(str(i), 1) for i in range(1, 9)]))  # row 9 missing
    problems = refused(sheets)
    assert any("no reply for 1 rows (9)" in p for p in problems) and m


def test_duplicate_answers_across_files_are_refused(sheets):
    full_replies(sheets, 25)
    put_reply(sheets, "dev_again.txt", reply(("3", 2)))
    assert any("answered twice" in p for p in refused(sheets))


def test_an_unknown_or_excluded_row_is_refused(sheets):
    lb.export_split("dev", sheets, 25, exclude_blog=True)
    put_reply(sheets, "dev_1.txt", reply(*[(str(i), 1) for i in range(1, 10)] + [("77", 1)]))
    problems = refused(sheets)
    assert any("row 77 is not in" in p for p in problems)
    assert any("row 5 was excluded" in p for p in problems)


def test_bad_labels_in_a_reply_stop_the_merge(sheets):
    lb.export_split("dev", sheets, 25)
    put_reply(sheets, "dev_1.txt", reply(*[(str(i), 3 if i == 4 else 1) for i in range(1, 10)]))
    assert any("not 0, 1 or 2" in p for p in refused(sheets))


def test_a_sheet_that_changed_after_the_export_is_refused(sheets):
    full_replies(sheets)
    rows = toy_rows()
    rows[0]["text"] = "edited after the export"
    pooling.write_sheet(sheets / "judge2_dev.csv", rows)
    assert any("changed since the batches were exported" in p for p in refused(sheets))


def test_no_replies_and_no_export_are_clear_errors(sheets):
    rep, problems = merge(sheets)
    assert rep == {} and "export" in problems[0]
    lb.export_split("dev", sheets, 25)
    assert "no reply files" in merge(sheets)[1][0]


def test_unsure_naming_an_unanswered_row_is_refused(sheets):
    lb.export_split("dev", sheets, 25)
    put_reply(sheets, "dev_1.txt",
              reply(*[(str(i), 1) for i in range(1, 9)]) + "UNSURE: 9\n")
    assert any("UNSURE names row 9" in p for p in refused(sheets))


def test_allow_partial_writes_blanks_that_the_importer_then_rejects(sheets):
    lb.export_split("dev", sheets, 25)
    put_reply(sheets, "dev_1.txt", reply(*[(str(i), 1) for i in range(1, 9)]))
    rep, problems = merge(sheets, allow_partial=True)
    assert problems == [] and rep["unanswered"] == ["9"]
    rows = pooling.read_sheet(sheets / "judge2_dev_labelled.csv")
    assert rows[-1]["label"] == ""
    pool = {"pairs": {q: {r["chunk_id"]: {} for r in toy_rows() if r["query_id"] == q}
                      for q in TEXTS}}
    _, probs = imp.validate(rows, pool, [{"id": q, "text": t, "split": "dev"}
                                         for q, t in TEXTS.items()], "dev")
    assert any("blank label" in p for p in probs)


def test_excluded_blog_rows_stay_blank_for_hand_labelling(sheets):
    full_replies(sheets, 25, exclude_blog=True)
    rep, problems = merge(sheets)
    assert problems == [] and rep["llm"] == 7 and rep["blog_to_label_by_hand"] == ["5", "7"]
    blank = [r["row_id"] for r in pooling.read_sheet(sheets / "judge2_dev_labelled.csv")
             if not r["label"]]
    assert blank == ["5", "7"]


# ---- merge: the owner's own labels are never overwritten ------------------------------------

def hand_label(sheets, labels, note="mine"):
    """Write judge2_dev_labelled.csv as the owner would: their labels in some rows."""
    rows = toy_rows()
    for r in rows:
        if r["row_id"] in labels:
            r["label"], r["note"] = str(labels[r["row_id"]]), note
    pooling.write_sheet(sheets / "judge2_dev_labelled.csv", rows)


def test_hand_labels_are_kept_and_compared_with_the_chat(sheets):
    full_replies(sheets, 25)  # chat labels: row_id % 3
    hand_label(sheets, {"1": 1, "2": 0, "3": 0, "4": 1})  # chat: 1, 2, 0, 1
    rep, problems = merge(sheets)
    out = {r["row_id"]: r for r in pooling.read_sheet(sheets / "judge2_dev_labelled.csv")}
    assert problems == [] and rep["hand"] == 4 and rep["llm"] == 5
    assert (out["1"]["label"], out["2"]["label"], out["3"]["label"]) == ("1", "0", "0")
    assert out["2"]["note"] == "mine" and out["5"]["label"] == "2"
    a = rep["agreement"]
    assert (a["n"], a["agree"]) == (4, 3) and a["agreement"] == 0.75
    assert rep["disagreements"] == [("2", "Q1", 0, 2)]


def test_agreement_known_answer():
    a = lb.agreement([(0, 0), (1, 1), (2, 2), (2, 1)])
    assert (a["n"], a["agree"]) == (4, 3)
    assert a["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert lb.agreement([])["agreement"] is None


def test_running_the_merge_twice_does_not_turn_chat_labels_into_hand_labels(sheets):
    full_replies(sheets, 25)
    hand_label(sheets, {"1": 2})
    first, _ = merge(sheets)
    second, problems = merge(sheets)
    assert problems == [] and first["hand"] == second["hand"] == 1
    assert second["llm"] == first["llm"] == 8
    prov = json.loads((sheets / "prov.json").read_text())["dev"]
    assert prov["hand"] == ["1"] and "1" not in prov["llm"]


def test_a_chat_label_the_owner_later_edits_becomes_a_hand_label(sheets):
    full_replies(sheets, 25)
    merge(sheets)
    rows = pooling.read_sheet(sheets / "judge2_dev_labelled.csv")
    rows[3]["label"] = "2"  # row 4: the chat said 1, the owner changes it
    pooling.write_sheet(sheets / "judge2_dev_labelled.csv", rows)
    rep, problems = merge(sheets)
    out = {r["row_id"]: r for r in pooling.read_sheet(sheets / "judge2_dev_labelled.csv")}
    assert problems == [] and rep["hand"] == 1 and out["4"]["label"] == "2"
    assert json.loads((sheets / "prov.json").read_text())["dev"]["hand"] == ["4"]


def test_an_existing_file_whose_other_columns_were_edited_is_refused(sheets):
    full_replies(sheets, 25)
    rows = toy_rows()
    rows[2]["text"] = "tampered"
    pooling.write_sheet(sheets / "judge2_dev_labelled.csv", rows)
    before = (sheets / "judge2_dev_labelled.csv").read_bytes()
    rep, problems = merge(sheets)
    assert rep == {} and any("text" in p or "changed" in p for p in problems)
    assert (sheets / "judge2_dev_labelled.csv").read_bytes() == before  # untouched


def test_an_invalid_label_typed_by_hand_is_reported_not_overwritten(sheets):
    full_replies(sheets, 25)
    rows = toy_rows()
    rows[0]["label"] = "maybe"
    pooling.write_sheet(sheets / "judge2_dev_labelled.csv", rows)
    rep, problems = merge(sheets)
    assert rep == {} and any("row 1 holds 'maybe'" in p for p in problems)


def test_float_hand_labels_from_a_spreadsheet_are_understood(sheets):
    full_replies(sheets, 25)
    hand_label(sheets, {"1": "1.0"})
    rep, problems = merge(sheets)
    assert problems == [] and rep["hand"] == 1


# ---- the owner's rule: this code never opens judgments --------------------------------------

def test_the_script_never_references_the_qrels_files():
    src = (ROOT / "scripts" / "label_batches.py").read_text(encoding="utf-8")
    for forbidden in ("qrels_test", "qrels_dev", "load_qrels", "allow_test"):
        assert forbidden not in src
