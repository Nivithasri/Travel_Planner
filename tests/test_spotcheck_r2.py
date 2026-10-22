"""scripts/spotcheck_r2.py on a toy sheet: sampling constraints and the scoring refusals."""
import pytest

from scripts import spotcheck_r2 as sp


def toy_candidates(n=60):
    return [{"split": "dev" if i % 2 else "test", "query_id": f"Q{i % 7:02d}",
             "chunk_id": f"c{i:03d}", "strata": [sp.STRATA[i % 3]],
             "source_type": "blog" if i % 5 == 0 else "reference", "chat_label": i % 3,
             "label": str(i % 3), "text": f"t{i}", "query_text": "q", "title": "T",
             "place": "P", "state": "KA"} for i in range(n)]


def test_draw_is_seeded_balanced_and_has_enough_blog():
    c = toy_candidates()
    a, b = sp.draw(c, 30, 42, 8), sp.draw(c, 30, 42, 8)
    assert [x["chunk_id"] for x in a] == [x["chunk_id"] for x in b]
    assert len({x["chunk_id"] for x in a}) == 30
    assert sum(x["source_type"] == "blog" for x in a) >= 8
    counts = [sum(s in x["strata"] for x in a) for s in sp.STRATA]
    assert max(counts) - min(counts) <= 3
    assert [x["chunk_id"] for x in sp.draw(c, 30, 7, 8)] != [x["chunk_id"] for x in a]


def test_draw_refuses_a_pool_that_cannot_meet_the_constraints():
    with pytest.raises(SystemExit):
        sp.draw(toy_candidates(12), 30, 42, 8)
    no_blog = [x for x in toy_candidates() if x["source_type"] != "blog"]
    with pytest.raises(SystemExit):
        sp.draw(no_blog, 10, 42, 8)


def test_candidates_need_a_stratum_row_and_a_valid_chat_label():
    pairs = {"Q01": {"a": {"systems": ["dense_control", "row7_rerank"]},
                     "b": {"systems": ["dense_control"]},
                     "c": {"systems": ["row9_crag"]}}}
    rows = [{"query_id": "Q01", "chunk_id": k, "label": "1"} for k in "abcd"]
    rows[2]["label"] = ""
    got = sp.candidates(pairs, {"dev": rows})
    assert [c["chunk_id"] for c in got] == ["a"]  # b: no stratum, c: blank label, d: not pooled
    assert got[0]["strata"] == ["row7_rerank"] and got[0]["chat_label"] == 1


def make(mine):
    rows = [{"row_id": str(i), "query_text": "q", "title": "T", "place": "P", "state": "KA",
             "source_type": "blog", "text": f"t{i}", "label": str(v), "note": "", "rubric": ""}
            for i, v in enumerate(mine, 1)]
    chat = [0, 1, 2, 2, 0, 1]
    key = {"fixed_columns_sha256": sp.fixed_sha(rows),
           "rows": {str(i): {"chunk_id": f"c{i}", "query_id": "Q01", "split": "dev",
                             "chat_label": v} for i, v in enumerate(chat, 1)}}
    return rows, key


def test_score_matches_a_hand_computed_table():
    rows, key = make([0, 1, 2, 0, 0, 1])  # chat 0 1 2 2 0 1: five of six agree, row 4 is 0 vs 2
    text = sp.score(rows, key)
    assert "exact agreement: 0.833" in text
    assert "rows differing by 2 grades: 1" in text and "row 4: you 0, chat 2" in text
    # by hand: po = 5/6; your row sums 3 2 1, chat column sums 2 2 2, pe = 12/36 = 1/3,
    # kappa = (5/6 - 1/3) / (2/3) = 0.75
    assert "kappa (unweighted): 0.750" in text


def test_score_refuses_blank_bad_labels_and_changed_rows():
    rows, key = make([0, 1, 2, 2, 0, 1])
    for bad in ("", "3", "1.5", "x"):
        broken = [dict(r) for r in rows]
        broken[2]["label"] = bad
        with pytest.raises(ValueError, match="blank or not 0/1/2"):
            sp.score(broken, key)
    edited = [dict(r) for r in rows]
    edited[0]["text"] = "changed"
    with pytest.raises(ValueError, match="fixed column"):
        sp.score(edited, key)
    with pytest.raises(ValueError, match="row_ids"):
        sp.score(rows[:-1], key)
    floats = [{**r, "label": r["label"] + ".0"} for r in rows]  # a spreadsheet may save 1.0
    assert "exact agreement: 1.000" in sp.score(floats, key)
