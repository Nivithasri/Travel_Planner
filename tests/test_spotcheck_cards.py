"""scripts/spotcheck_cards.py on a toy card set: drawing and the scoring refusals."""
import pytest

from scripts import spotcheck_cards as sc


def item(cid, st="wikivoyage"):
    return {"text": f"text {cid}", "chunk_id": cid, "source_type": st, "aspect": "timing",
            "provenance": "llm", "url": "https://x.test"}


CARDS = {
    "Hampi": {"evidence_class": "full", "state": "KA", "tips": [item("t1"), item("t2")],
              "warnings": [item("w1"), item("w2"), item("w3")]},
    "Chennai": {"evidence_class": "full", "state": "TN", "tips": [item("t3")], "warnings": []},
    "Munnar": {"evidence_class": "limited", "state": "KL", "tips": [item("t4"), item("t5")],
               "warnings": []},
}


def test_draw_is_seeded_and_respects_the_per_place_counts():
    a = sc.draw(CARDS, 42, 1, 2, 1)
    b = sc.draw(CARDS, 42, 1, 2, 1)
    assert [p["item"]["chunk_id"] for p in a] == [p["item"]["chunk_id"] for p in b]
    by = {}
    for p in a:
        by.setdefault((p["place"], p["kind"]), []).append(p)
    assert len(by[("Hampi", "tip")]) == 1 and len(by[("Hampi", "warning")]) == 2
    assert len(by[("Chennai", "tip")]) == 1 and ("Chennai", "warning") not in by
    assert len(by[("Munnar", "tip")]) == 1 and ("Munnar", "warning") not in by
    assert len(a) == 5


def make(marks):
    rows = [{"row_id": str(i), "place": "P", "state": "KA", "source_type": "wikivoyage",
             "excerpt": f"e{i}", "chunk_text": f"c{i}", "about_place": str(a), "valid": str(v),
             "note": ""} for i, (a, v) in enumerate(marks, 1)]
    kinds = ["tip", "tip", "warning", "warning"]
    classes = ["full", "limited", "full", "full"]
    key = {"fixed_sha256": sc.fixed_sha(rows),
           "rows": {str(i): {"place": "P", "kind": kinds[i - 1], "chunk_id": f"c{i}",
                             "provenance": "llm", "evidence_class": classes[i - 1]}
                    for i in range(1, 5)}}
    return rows, key


def test_score_counts_precision_by_kind_and_class():
    rows, key = make([(1, 1), (1, 0), (1, 1), (0, 0)])
    text = sc.score(rows, key)
    assert "all: about_place 3/4 (0.750), valid 2/4 (0.500)" in text
    assert "tips: about_place 2/2 (1.000), valid 1/2 (0.500)" in text
    assert "warnings: about_place 1/2 (0.500), valid 1/2 (0.500)" in text
    assert "limited-evidence places: about_place 1/1 (1.000), valid 0/1 (0.000)" in text
    assert "rows with a 0: ['2', '4']" in text


def test_score_refuses_blank_bad_values_changed_rows_and_missing_rows():
    rows, key = make([(1, 1)] * 4)
    for bad in ("", "2", "yes", "0.5"):
        broken = [dict(r) for r in rows]
        broken[1]["valid"] = bad
        with pytest.raises(ValueError, match="blank or not 0/1"):
            sc.score(broken, key)
    edited = [dict(r) for r in rows]
    edited[0]["excerpt"] = "changed"
    with pytest.raises(ValueError, match="fixed column"):
        sc.score(edited, key)
    with pytest.raises(ValueError, match="row_ids"):
        sc.score(rows[:-1], key)
