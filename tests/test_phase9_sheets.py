"""Blind scoring sheets (eval/phase9/sheets.py, D124): rendering, blindness, key, import."""
import pytest

from eval.phase9 import sheets as S

REQS = {f"P{i:02d}": {"id": f"P{i:02d}", "text": f"Request number {i}."} for i in range(1, 5)}
SYSTEMS = ("llm_only", "plain_rag", "full")


def rec(rid, system, arm, text="## Day 1: Mysuru\n- A fact. [chunk:c1]\n- Another. [chunk:c2]\n",
        live=None):
    return {"request_id": rid, "system": system, "arm": arm, "text": text,
            "contexts": [{"chunk_id": "c1", "place": "Mysuru", "source_type": "wikivoyage",
                          "url": "https://x.test/c1", "text": "Excerpt one."},
                         {"chunk_id": "c2", "place": "Hampi", "source_type": "reference",
                          "url": "https://x.test/c2", "text": "Excerpt two."}],
            "citations": [{"chunk_id": "live:ab", "place": "Mysuru", "url": "https://y.test/p",
                           "fetch_date": "2026-10-24", "live": True}],
            "live": live}


def records():
    arms = {"llm_only": "llm_only", "plain_rag": "plain_rag", "full": "full_corpus"}
    return [rec(rid, s, arms[s]) for rid in REQS for s in SYSTEMS]


def test_render_numbers_markers_by_first_appearance_and_lists_excerpts():
    text, sources = S.render(rec("P01", "full", "full_corpus",
                                 text="- B. [chunk:c2]\n- A. [chunk:c1]\n- B again. [chunk:c2]\n"))
    assert text.count("[S1]") == 2 and text.count("[S2]") == 1 and "[chunk:" not in text
    assert sources.index("[S1] Hampi") < sources.index("[S2] Mysuru")
    assert "excerpt: Excerpt two." in sources


def test_a_live_marker_is_listed_with_its_link_and_never_an_excerpt():
    r = rec("P01", "full", "full_live", text="- Hours today. [live:live:ab]\n")
    r["contexts"].append({"chunk_id": "live:ab", "text": "PROVIDER SNIPPET", "place": "Mysuru"})
    text, sources = S.render(r)
    assert "[L1]" in text and "https://y.test/p" in sources and "fetched 2026-10-24" in sources
    assert "PROVIDER SNIPPET" not in sources


def test_the_sheet_is_seeded_blind_and_keyed():
    rows1, key1 = S.build_sheet(records(), REQS, seed=7)
    rows2, key2 = S.build_sheet(records(), REQS, seed=7)
    assert rows1 == rows2 and key1 == key2
    assert [r["row_id"] for r in rows1] == [f"R{i:02d}" for i in range(1, 13)]
    other, _ = S.build_sheet(records(), REQS, seed=8)
    assert [r["output"] for r in other] != [r["output"] for r in rows1] or \
        [r["request"] for r in other] != [r["request"] for r in rows1]
    blob = " ".join(str(v) for r in rows1 for v in r.values())
    for hidden in ("llm_only", "plain_rag", "full_corpus", "full_live"):
        assert hidden not in blob
    assert {m["system"] for m in key1.values()} == set(SYSTEMS)
    assert all(len(m["sha"]) == 64 for m in key1.values())


def filled(rows, **scores):
    out = []
    for r in rows:
        r = dict(r)
        for d in S.DIMENSIONS:
            r[d] = str(scores.get(d, 4))
        out.append(r)
    return out


def test_read_scores_accepts_whole_numbers_even_as_floats():
    rows, key = S.build_sheet(records(), REQS)
    ok = S.read_scores(filled(rows, feasibility="3.0"), key)
    assert all(v["feasibility"] == 3 and v["groundedness"] == 4 for v in ok.values())


@pytest.mark.parametrize("bad", ["", "0", "6", "3.5", "four"])
def test_read_scores_refuses_blank_or_invalid_scores(bad):
    rows, key = S.build_sheet(records(), REQS)
    with pytest.raises(ValueError):
        S.read_scores(filled(rows, freshness=bad), key)


def test_read_scores_refuses_a_changed_fixed_column_and_a_missing_row():
    rows, key = S.build_sheet(records(), REQS)
    tampered = filled(rows)
    tampered[0]["output"] += " edited"
    with pytest.raises(ValueError, match="fixed column"):
        S.read_scores(tampered, key)
    with pytest.raises(ValueError, match="missing"):
        S.read_scores(filled(rows)[1:], key)


def test_scores_group_by_request_and_system_and_optionally_by_arm():
    recs = records() + [rec("P01", "full", "full_live", live={"status": "ok"})]
    rows, key = S.build_sheet(recs, REQS)
    scores = S.read_scores(filled(rows), key)
    by_sys = S.by_request_system(
        {k: v for k, v in scores.items() if key[k]["arm"] != "full_live"}, key)
    assert ("P01", "full") in by_sys and len(by_sys) == 12
    by_arm = S.by_request_system(scores, key, arm=True)
    assert ("P01", "full_live") in by_arm and ("P01", "full_corpus") in by_arm


def test_the_cross_check_sample_never_holds_a_live_row():
    recs = records() + [rec(rid, "full", "full_live", live={"status": "ok"}) for rid in REQS]
    _, key = S.build_sheet(recs, REQS)
    ids = S.cross_check_sample(key, 20, seed=1)
    assert ids and all(not key[i]["has_live"] and key[i]["arm"] != "full_live" for i in ids)
    assert len(ids) == len([m for m in key.values() if m["arm"] != "full_live"])
    assert S.cross_check_sample(key, 3, seed=1) == S.cross_check_sample(key, 3, seed=1)


def test_csv_round_trip(tmp_path):
    rows, _ = S.build_sheet(records(), REQS)
    p = tmp_path / "sheet.csv"
    S.write_csv(rows, p)
    back = S.read_csv(p)
    assert [r["row_id"] for r in back] == [r["row_id"] for r in rows]
    assert back[0]["output"] == rows[0]["output"] and back[0]["freshness"] == ""
