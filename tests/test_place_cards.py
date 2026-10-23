"""Place cards (src/tools/place_cards.py, scripts/build_place_cards.py; D69, D102)."""
import json
from types import SimpleNamespace

import pytest

from src.common.schemas import REPO_ROOT
from src.tools import place_cards as pc
from src.tools.routing import UnknownPlaceError


def entry(name, popularity="popular", signal=1000.0, state="KA", holdout=False, aliases=()):
    return SimpleNamespace(name=name, popularity=popularity, popularity_signal=signal,
                           state=SimpleNamespace(value=state), holdout=holdout,
                           aliases=list(aliases))


def chunk(cid, place, text, *, st="wikivoyage", aspect="timing", warn=False, ptype="city"):
    return {"chunk_id": cid, "place": place, "text": text, "source_type": st, "aspect": aspect,
            "is_warning": warn, "place_type": ptype, "url": f"https://x.test/{cid}"}


SENT = "Temples open at six in the morning and close at noon for the heat."
SEASON = {"places": {
    "Hampi": {"confidence": "medium", "hill": False, "nearest_airport": {"iata": "VDY"},
              "months": {str(m): {"verdict": "good" if m in (11, 12, 1) else
                                  ("poor" if m == 5 else "fair"), "access_risk": m == 7}
                         for m in range(1, 13)}},
    "Munnar": {"confidence": "low", "hill": True, "nearest_airport": None,
               "months": {str(m): {"verdict": "fair", "access_risk": False}
                          for m in range(1, 13)}},
    "Gokarna": {"confidence": "medium", "hill": False, "nearest_airport": None,
                "months": {str(m): {"verdict": "good", "access_risk": False}
                           for m in range(1, 13)}}}}


def build(chunks, prov, **kw):
    entries = [entry("Hampi", signal=300.0, aliases=["Vijayanagara"]), entry("Munnar", signal=50.0),
               entry("Gokarna", signal=100.0), entry("Thekkady", holdout=True)]
    return pc.build_cards(chunks, prov, entries, SEASON, {"Munnar"}, **kw)


def test_excerpt_takes_a_full_sentence_and_cuts_at_a_word_boundary():
    assert pc.excerpt("see also. " + SENT + " Another sentence follows here.") == SENT
    long = "Word " * 100
    out = pc.excerpt(long.strip(), 50)
    assert out.endswith("...") and len(out) <= 53 and "  " not in out


def test_only_model_written_warnings_are_shown_as_warnings():
    chunks = [chunk("w1", "Hampi", SENT, aspect="safety", warn=True),
              chunk("w2", "Hampi", SENT + " Extra.", aspect="safety", warn=True),
              chunk("t1", "Hampi", "Carry water because shade is rare at midday.", aspect="timing")]
    prov = {"w1": "llm", "w2": "rule_only", "t1": "llm"}
    card = build(chunks, prov)["Hampi"]
    assert [w["chunk_id"] for w in card["warnings"]] == ["w1"]
    assert card["counts"]["rule_based_warning_flags"] == 1
    assert all(w["provenance"] == "llm" for w in card["warnings"])
    assert {t["chunk_id"] for t in card["tips"]} == {"t1"}  # a warning is never also a tip


def test_blog_chunks_are_excluded_by_default_and_counted():
    chunks = [chunk("b1", "Hampi", "A blog sentence that is long enough to be used as a tip.",
                    st="blog"), chunk("t1", "Hampi", SENT)]
    prov = {"b1": "llm", "t1": "llm"}
    card = build(chunks, prov)["Hampi"]
    assert {t["chunk_id"] for t in card["tips"]} == {"t1"} and card["counts"]["blog_excluded"] == 1
    on = build(chunks, prov, include_blog=True)["Hampi"]
    assert {t["chunk_id"] for t in on["tips"]} == {"b1", "t1"}


def test_limited_places_use_rule_based_rows_but_never_show_warnings():
    chunks = [chunk("r1", "Munnar", SENT, aspect="timing"),
              chunk("r2", "Munnar", "Landslides block the ghat road in heavy rain every year.",
                    aspect="safety", warn=True)]
    prov = {"r1": "rule_only", "r2": "rule_only"}
    card = build(chunks, prov)["Munnar"]
    assert card["evidence_class"] == "limited" and card["visibility_tercile"] is None
    assert card["warnings"] == [] and card["counts"]["rule_based_warning_flags"] == 1
    assert [t["provenance"] for t in card["tips"]] == ["rule_only"]
    assert any("limited evidence" in c for c in card["caveats"])
    assert any("confidence is low" in c for c in card["caveats"])


def test_full_places_ignore_rule_based_rows_for_tips():
    chunks = [chunk("r1", "Hampi", SENT),
              chunk("l1", "Hampi", "Ferries run across the river daily.")]
    card = build(chunks, {"r1": "rule_only", "l1": "llm"})["Hampi"]
    assert {t["chunk_id"] for t in card["tips"]} == {"l1"}


def test_every_item_cites_a_chunk_of_that_place():
    chunks = [chunk("h1", "Hampi", SENT), chunk("g1", "Gokarna", SENT + " Beach."),
              chunk("h2", "Hampi", "The ruins stretch for kilometres along the river.",
                    aspect="scenery", ptype="heritage")]
    cards = build(chunks, {"h1": "llm", "g1": "llm", "h2": "llm"})
    owner = {c["chunk_id"]: c["place"] for c in chunks}
    for name, card in cards.items():
        for item in card["tips"] + card["warnings"] + card["highlights"]:
            assert owner[item["chunk_id"]] == name


def test_holdout_text_is_dropped():
    class Hold:
        def text_match(self, text):
            return ("Thekkady", "x") if "Thekkady" in text else None
    chunks = [chunk("h1", "Hampi", "Visit Thekkady nearby for a boat ride on the lake."),
              chunk("h2", "Hampi", SENT)]
    card = build(chunks, {"h1": "llm", "h2": "llm"}, holdout=Hold())["Hampi"]
    assert {t["chunk_id"] for t in card["tips"]} == {"h2"}


def test_hold_out_places_get_no_card_and_visibility_terciles_cover_full_places():
    cards = build([], {})
    assert set(cards) == {"Hampi", "Munnar", "Gokarna"}
    assert cards["Hampi"]["visibility_tercile"] == 1  # highest signal of the two full places
    assert cards["Gokarna"]["visibility_tercile"] == 2  # two full places: ranks 0 and 1 of 2
    entries = [entry(f"P{i}", signal=float(100 - i)) for i in range(17)]
    t = pc.visibility_terciles(entries, {e.name for e in entries})
    assert sorted(t.values()).count(1) == 6 and sorted(t.values()).count(3) == 5


def test_best_months_and_recency_note():
    card = build([], {})["Hampi"]
    bm = card["best_months"]
    assert bm["good"] == [1, 11, 12] and bm["poor"] == [5] and bm["access_risk"] == [7]
    assert "climate only" in bm["basis"]
    assert "Nothing on this card is evidence of current conditions" in card["recency"]


def test_get_place_card_resolves_aliases_and_rejects_unknown(tmp_path):
    cards = build([], {})
    path = tmp_path / "cards.json"
    path.write_text(json.dumps({"meta": {}, "places": cards}), encoding="utf-8")
    assert pc.get_place_card("hampi", path)["place"] == "Hampi"
    assert pc.get_place_card("Vijayanagara", path)["place"] == "Hampi"
    for bad in ("Thekkady", "MAA", "Atlantis", 42):
        with pytest.raises(UnknownPlaceError):
            pc.get_place_card(bad, path)


# ---- the built file (skipped on a fresh clone: index/ and data/enriched are gitignored) ------

CARDS = REPO_ROOT / "index" / "place_cards.json"
ENRICHED = REPO_ROOT / "data" / "enriched" / "chunks.jsonl"
PROV = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"
needs_data = pytest.mark.skipif(not (CARDS.exists() and ENRICHED.exists() and PROV.exists()),
                                reason="index/place_cards.json or data/enriched missing "
                                       "(gitignored, O12)")


@needs_data
def test_built_cards_cite_real_chunks_of_their_place_and_follow_the_rules():
    doc = json.loads(CARDS.read_text(encoding="utf-8"))
    chunks = {}
    for line in ENRICHED.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
    prov = {}
    for line in PROV.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        prov[r["chunk_id"]] = r["source"]
    assert len(doc["places"]) == 26 and doc["meta"]["include_blog"] is False
    for name, card in doc["places"].items():
        for item in card["tips"] + card["warnings"] + card["highlights"]:
            c = chunks[item["chunk_id"]]
            assert c["place"] == name and c["source_type"] != "blog"
            assert item["provenance"] == prov[item["chunk_id"]]
        for w in card["warnings"]:
            assert w["provenance"] == "llm" and chunks[w["chunk_id"]]["is_warning"] is True
        if card["evidence_class"] == "limited":
            assert card["warnings"] == [] and card["visibility_tercile"] is None
        else:
            assert all(i["provenance"] == "llm" for i in card["tips"] + card["highlights"])
