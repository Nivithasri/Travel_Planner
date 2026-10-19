"""Gold sample for aspect / is_warning (D58). Offline; toy chunks and a toy CSV."""
from __future__ import annotations

import pytest

from src.process.label import (
    ASPECTS,
    LabelError,
    format_score,
    gold_rows,
    parse_gold,
    read_gold_csv,
    score,
    select_gold,
    write_gold_csv,
)


def toy_chunks():
    out = []
    for st in ("blog", "reference", "wikivoyage"):
        for p in ("A", "B", "C"):
            for k in range(8):
                out.append({"chunk_id": f"{st[:2]}{p}{k}", "source_type": st, "place": p,
                            "text": f"text of {st} {p} {k}"})
    return out


def test_select_gold_is_stratified_seeded_and_avoids_the_pilot():
    chunks = toy_chunks()
    shorts = {c["chunk_id"] for c in chunks if c["chunk_id"].endswith(("6", "7"))}
    pilot = {c["chunk_id"] for c in chunks if c["chunk_id"].endswith("0")}
    ids = select_gold(chunks, pilot, shorts, seed=43)
    assert ids == select_gold(chunks, pilot, shorts, seed=43)  # deterministic
    assert ids != select_gold(chunks, pilot, shorts, seed=44)
    assert len(ids) == len(set(ids)) == 30
    by_id = {c["chunk_id"]: c for c in chunks}
    for st in ("blog", "reference", "wikivoyage"):
        assert sum(by_id[i]["source_type"] == st for i in ids) == 10
    assert sum(i in shorts for i in ids) == 6 and not (set(ids) & pilot)
    assert all(sum(by_id[i]["source_type"] == st and by_id[i]["place"] == p for i in ids) >= 2
               for st in ("blog", "reference", "wikivoyage") for p in "ABC")  # places spread
    assert [by_id[i]["source_type"] for i in ids[:6]] != ["blog"] * 6  # not grouped by type


def test_sheet_shows_text_only_and_never_overwrites_labels(tmp_path):
    chunks = toy_chunks()
    path = tmp_path / "gold.csv"
    rows = gold_rows(chunks, ["blA1", "reB2"])
    write_gold_csv(path, rows)
    back = read_gold_csv(path)
    assert [r["chunk_id"] for r in back] == ["blA1", "reB2"]
    assert set(back[0]) == {"chunk_id", "place", "source_type", "text", "aspect_gold",
                            "is_warning_gold"}
    assert all(r["aspect_gold"] == "" and r["is_warning_gold"] == "" for r in back)  # no labels
    path.write_text(path.read_text(encoding="utf-8-sig").replace(
        "blA1,A,blog,text of blog A 1,,", "blA1,A,blog,text of blog A 1,history,false"),
        encoding="utf-8-sig")
    with pytest.raises(FileExistsError):
        write_gold_csv(path, rows)  # would destroy the owner's labels
    assert "history" in path.read_text(encoding="utf-8-sig")
    write_gold_csv(path, rows, force=True)
    assert read_gold_csv(path)[0]["aspect_gold"] == ""


def test_text_with_commas_quotes_and_newlines_survives_the_csv(tmp_path):
    tricky = 'Line one, with "quotes"\nand a second line; ₹500, 9AM'
    path = tmp_path / "g.csv"
    write_gold_csv(path, gold_rows([{"chunk_id": "x", "source_type": "blog", "place": "A",
                                     "text": tricky}], ["x"]))
    assert read_gold_csv(path)[0]["text"] == tricky


def filled(**kw):
    base = {"chunk_id": "c", "text": "t", "aspect_gold": "", "is_warning_gold": "",
            "place": "A", "source_type": "blog"}
    return {**base, **kw}


def test_parse_gold_skips_blank_rows_and_rejects_bad_labels():
    rows = [filled(chunk_id="a", aspect_gold="History ", is_warning_gold="FALSE"),
            filled(chunk_id="b"),
            filled(chunk_id="c", aspect_gold="safety", is_warning_gold="yes")]
    gold, blank = parse_gold(rows)
    assert blank == 1 and [(g["chunk_id"], g["aspect"], g["is_warning"]) for g in gold] == [
        ("a", "history", False), ("c", "safety", True)]
    with pytest.raises(LabelError, match="weather"):
        parse_gold([filled(aspect_gold="weather", is_warning_gold="false")])
    with pytest.raises(LabelError, match="is_warning_gold"):
        parse_gold([filled(aspect_gold="other", is_warning_gold="maybe")])
    with pytest.raises(LabelError):  # half filled
        parse_gold([filled(aspect_gold="other")])


def test_score_llm_and_rule_based_against_a_toy_gold_set():
    gold = [
        {"chunk_id": "1", "text": "The fort is closed on Mondays.", "aspect": "closure",
         "is_warning": True},
        {"chunk_id": "2", "text": "Entry fee is Rs 50.", "aspect": "price", "is_warning": False},
        {"chunk_id": "3", "text": "A long history of kings and an ancient dynasty.",
         "aspect": "history", "is_warning": False},
        {"chunk_id": "4", "text": "Nothing special here.", "aspect": "scenery",
         "is_warning": False},
        {"chunk_id": "5", "text": "Beware of scams near the bus stand.", "aspect": "safety",
         "is_warning": True},
    ]
    llm = {"1": {"aspect": "closure", "is_warning": True},
           "2": {"aspect": "price", "is_warning": False},
           "3": {"aspect": "scenery", "is_warning": False},   # wrong aspect
           "4": {"aspect": "scenery", "is_warning": True}}    # right aspect, wrong warning
    res = score(gold, llm)  # chunk 5 has no LLM label yet
    assert res["labelled"] == 5 and res["llm_missing"] == 1
    assert res["llm"]["n"] == 4 and res["llm"]["aspect_accuracy"] == 0.75
    assert res["llm"]["is_warning_accuracy"] == 0.75
    cls = res["llm"]["aspect_classes"]
    assert cls["scenery"] == {"gold": 1, "predicted": 2, "correct": 1}
    assert cls["history"] == {"gold": 1, "predicted": 0, "correct": 0}
    m = res["llm"]["aspect_confusion"]
    assert m[ASPECTS.index("history")][ASPECTS.index("scenery")] == 1  # gold -> pred
    assert res["llm"]["is_warning_confusion"] == [[2, 1], [0, 1]]  # rows gold false/true
    rule = res["rule"]  # rule-based labels come from the text itself, for all 5
    assert rule["n"] == 5
    assert rule["aspect_classes"]["closure"]["correct"] == 1
    assert rule["aspect_classes"]["safety"]["correct"] == 1
    assert rule["aspect_classes"]["price"]["correct"] == 1
    assert rule["aspect_classes"]["history"]["correct"] == 1
    assert rule["aspect_accuracy"] == pytest.approx(4 / 5)  # "Nothing special" -> other
    text = format_score(res)
    assert "LLM vs gold, n=4" in text and "RULE-BASED (the fallback) vs gold, n=5" in text
    assert "aspect confusion (rows = gold)" in text and "75.0%" in text


def test_score_with_nothing_labelled_by_the_llm_yet():
    res = score([{"chunk_id": "1", "text": "Entry fee is Rs 50.", "aspect": "price",
                  "is_warning": False}], {})
    assert res["llm"]["n"] == 0 and res["llm"]["aspect_accuracy"] is None
    assert "n/a" in format_score(res) and res["rule"]["aspect_accuracy"] == 1.0
