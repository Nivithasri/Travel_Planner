"""Citation precision tooling (eval/phase9/claims.py, D125)."""
import pytest

from eval.phase9 import claims as C

TEXT = ("# South India itinerary\n\n## Route, season and budget (computed)\n"
        "- Chennai to Mysuru about 6 h (travel_time, provisional)\n\n"
        "## Day 1: Mysuru\n- The palace is lit on Sundays. [chunk:c1]\n"
        "- Plain sentence with no source.\n"
        "- Two sources. [chunk:c2][live:live:ab]\n\n"
        "## Sources\n- [chunk:c1] Mysuru (wikivoyage) https://x.test/c1\n"
        "## Notes and warnings\n- [chunk:zz] a note\n")


def rec(rid, arm="full_corpus", text=TEXT):
    return {"request_id": rid, "arm": arm, "text": text,
            "contexts": [{"chunk_id": "c1", "place": "Mysuru", "text": "Lit on Sundays."},
                         {"chunk_id": "c2", "place": "Hampi", "text": "Ruins."}],
            "citations": [{"chunk_id": "live:ab", "place": "Mysuru", "url": "https://y.test/p",
                           "fetch_date": "2026-10-24"}]}


def test_claims_are_cited_bullets_of_the_day_sections_only():
    cl = C.extract_claims(rec("P01"))
    assert [c["text"] for c in cl] == ["The palace is lit on Sundays.", "Two sources."]
    assert cl[0]["claim_id"] == "P01-1" and cl[1]["cited"][1] == {"kind": "live", "id": "live:ab"}
    assert C.is_live(cl[1]) and not C.is_live(cl[0])


def test_the_sample_is_seeded_spread_over_requests_and_never_live():
    many = ("## Day 1: X\n" + "".join(f"- Fact {i}. [chunk:c{i}]\n" for i in range(6)))
    records = [rec(f"P{i:02d}", text=many) for i in range(1, 5)] + [rec("P05", "full_live", many)]
    a = C.sample_claims(records, 8, seed=3)
    assert a == C.sample_claims(records, 8, seed=3) and len(a) == 8
    assert {c["request_id"] for c in a} == {"P01", "P02", "P03", "P04"}  # no live arm
    assert max(sum(1 for c in a if c["request_id"] == r) for r in {"P01", "P02", "P03", "P04"}) == 2
    assert len(C.sample_claims(records, 500, seed=3)) == 24  # fewer claims than asked for
    assert not any(C.is_live(c) for c in a)


def test_a_live_citing_claim_is_never_in_the_corpus_sample_even_inside_a_sampled_arm():
    # TEXT holds one corpus claim and one claim that also cites a live item
    sampled = C.sample_claims([rec("P01")], 10, seed=1)
    assert [c["text"] for c in sampled] == ["The palace is lit on Sundays."]
    both = C.sample_claims([rec("P01"), rec("P02", "full_live")], 10, seed=1,
                           arms=("full_corpus", "full_live"))
    assert len(both) == 2 and not any(C.is_live(c) for c in both)


def test_live_claims_come_from_the_live_arm_only():
    live = C.live_claims([rec("P01", "full_live"), rec("P02")])
    assert [c["request_id"] for c in live] == ["P01"] and C.is_live(live[0])


def test_passages_show_corpus_excerpts_and_live_links_only():
    r = rec("P01")
    cl = C.extract_claims(r)
    assert "Lit on Sundays." in C.passages(cl[0], r)
    both = C.passages(cl[1], r)
    assert "Ruins." in both and "https://y.test/p" in both and "fetched 2026-10-24" in both


def build(records=None):
    r = rec("P01", text=TEXT)
    chosen = C.extract_claims(r)
    recs = {("P01", "full_corpus"): r}
    reqs = {"P01": {"text": "Trip."}}
    return C.build_claim_sheet(chosen, recs, reqs, seed=1)


def test_the_claim_sheet_is_keyed_and_marks_are_validated():
    rows, key = build()
    assert [r["row_id"] for r in rows] == ["C01", "C02"]
    filled = [{**r, "supported": "1"} for r in rows]
    assert C.read_marks(filled, key) == {"C01": 1, "C02": 1}
    assert C.read_marks([{**r, "supported": "0.0"} for r in rows], key) == {"C01": 0, "C02": 0}
    for bad in ("", "2", "yes"):
        with pytest.raises(ValueError):
            C.read_marks([{**r, "supported": bad} for r in rows], key)
    tampered = [{**r, "supported": "1"} for r in rows]
    tampered[0]["claim"] += " x"
    with pytest.raises(ValueError, match="fixed column"):
        C.read_marks(tampered, key)
    with pytest.raises(ValueError, match="missing"):
        C.read_marks(filled[:1], key)


def test_precision_reports_corpus_rate_and_live_counts_separately():
    _, key = build()
    live_row = next(rid for rid, m in key.items() if m["live"])
    corpus_row = next(rid for rid, m in key.items() if not m["live"])
    res = C.precision({live_row: 1, corpus_row: 0}, key)
    assert res == {"corpus_n": 1, "corpus_supported": 0, "corpus_precision": 0.0,
                   "live_n": 1, "live_supported": 1}
    empty = C.precision({}, key)
    assert empty["corpus_precision"] is None
