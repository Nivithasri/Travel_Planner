from __future__ import annotations

import json
from datetime import date

import pytest

from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.places.core_list import (
    CORE_HUB_CAP,
    CoreListError,
    ListedPlace,
    Override,
    OverridesError,
    _district_like,
    _non_place,
    is_hub,
    load_core_list,
    load_overrides,
    resolve_listed,
    to_entry,
)
from src.places.core_list import Candidate as CoreCandidate
from src.places.wikidata import Ancestor, Candidate, Entity, WikidataClient, state_from_ancestors
from src.places.wv_lists import (
    count_see_do,
    parse_markers,
    passes_tourism_filter,
    region_links,
    section_text,
    see_do_words,
)

AS_OF = date(2026, 10, 18)
KL = Ancestor("Q1186", "Kerala", False)
TN = Ancestor("Q1445", "Tamil Nadu", False)
PY = Ancestor("Q66743", "Puducherry", False)
TS = Ancestor("Q677037", "Telangana", False)


# ---- the shipped file -----------------------------------------------------------------------
def test_shipped_core_list_loads_and_respects_the_hub_cap():
    places = load_core_list()
    assert len(places) == 30
    by_state = {s: sum(p.state == s for p in places) for s in ("TN", "KL", "KA", "AP", "TS", "PY")}
    assert by_state == {"TN": 8, "KL": 6, "KA": 7, "AP": 4, "TS": 3, "PY": 2}
    hubs = {p.name for p in places if p.hub}
    assert hubs == {"Chennai", "Madurai", "Kochi", "Bengaluru", "Mysuru", "Hyderabad"}
    assert len(hubs) <= CORE_HUB_CAP
    assert next(p for p in places if p.name == "Kodagu").aliases == ("Coorg",)
    assert "Ananthagiri Hills" not in {p.name for p in places}


def test_shipped_core_list_has_the_four_explicit_holdouts_and_two_regions():
    places = load_core_list()
    assert {p.name for p in places if p.holdout} == {
        "Chikmagalur", "Araku Valley", "Thekkady", "Thanjavur"}
    assert {p.name for p in places if p.place_type == "region"} == {"Kodagu", "Wayanad"}
    assert not any(p.hub and p.holdout for p in places)


def test_core_list_validation(tmp_path):
    p = tmp_path / "l.json"

    def load(places):
        p.write_text(json.dumps({"places": places}))
        return load_core_list(p)

    ok = {"name": "A", "state": "KL", "reason": "r"}
    assert load([ok])[0].name == "A"
    for bad in (
        [{"name": "A", "state": "XX", "reason": "r"}],
        [{"name": "A", "state": "KL"}],
        [{"name": "A", "state": "KL", "reason": "r", "colour": "x"}],
        [ok, {**ok, "name": "a"}],
        [{**ok, "name": f"H{i}", "hub": True} for i in range(CORE_HUB_CAP + 1)],
    ):
        with pytest.raises(CoreListError):
            load(bad)
    p.write_text(json.dumps([ok]))
    with pytest.raises(CoreListError):
        load_core_list(p)


# ---- type classification --------------------------------------------------------------------
def test_district_like_is_not_fooled_by_district_headquarters():
    assert _district_like(["district of India"]) and _district_like(["census district"])
    assert _district_like(["taluk of Kerala"]) and _district_like(["union territory of India"])
    assert _district_like(["division of Karnataka"]) and _district_like(["Taluk of Puducherry"])
    assert not _district_like(["city", "district headquarters"])
    assert not _district_like(["hill station", "town"])


def test_non_place_items_are_constituencies_airports_wards_and_stations():
    assert _non_place(["Lok Sabha constituency"]) and _non_place(["international airport"])
    assert _non_place(["ward"]) and _non_place(["railway station"])
    assert _non_place(["town", "constituency of the Karnataka Legislative Assembly"])
    assert not _non_place(["hill station"]) and not _non_place(["city"])
    assert not _non_place([])


# ---- resolution -----------------------------------------------------------------------------
def ent(qid, label, *, wv=None, wp=None, aliases=(), lat=10.0, lon=77.0, pop=None):
    return Entity(qid, label, list(aliases), lat, lon, wv, wp if wp is not None else label, pop)


class FakeWD:
    """items: qid -> (Entity, ancestors, types). search results come from `hits`."""

    def __init__(self, items, hits=None, titles=None, views=None):
        self.items = items
        self.hits = hits or {}
        self.titles = titles or {}
        self.views = views or {}

    def search(self, text, limit=7):
        return [Candidate(q, self.items[q][0].label, "") for q in self.hits.get(text, [])]

    def entities_by_wikivoyage_titles(self, titles):
        return {t: self.items[self.titles[t]][0] for t in titles if t in self.titles}

    def entities_by_qids(self, qids):
        return {q: self.items[q][0] for q in qids if q in self.items}

    def ancestors_many(self, qids):
        return {q: self.items[q][1] for q in qids}

    def types_many(self, qids):
        return {q: self.items[q][2] for q in qids}

    def pageviews_12m(self, title, as_of):
        return self.views.get(title, 100)  # default: some traffic; pass None to force no data


THREE = "== See ==\n{{see|a}}{{see|b}}{{see|c}}"
WT = {"Munnar": THREE, "Thin": "== See ==\n{{see|a}}", "Pondicherry": THREE, "Auroville": THREE,
      "Kovalam": THREE}


def resolve(wd, place):
    return resolve_listed(wd, place, WT.get, AS_OF)


def test_resolves_the_place_not_the_airport_or_constituency_homonyms():
    wd = FakeWD(
        {"Q1": (ent("Q1", "Munnar", wv="Munnar"), [KL], ["hill station", "town"]),
         "Q2": (ent("Q2", "Munnar constituency"), [KL], ["constituency"]),
         "Q3": (ent("Q3", "Munnar", aliases=["Munnar airport"]), [KL], ["airport"])},
        hits={"Munnar": ["Q1", "Q2", "Q3"]}, titles={"Munnar": "Q1"}, views={"Munnar": 900},
    )
    r = resolve(wd, ListedPlace("Munnar", "KL", "r"))
    c = r.candidate
    assert r.ok and c.qid == "Q1" and c.views == 900 and c.n_see_do == 3 and c.state == "KL"
    assert c.name == "Munnar" and c.source == "list" and c.reason == "r"
    assert r.flags == [] and any("Q3" in a for a in r.alternatives)


def test_registry_name_is_the_list_name_and_label_goes_to_aliases():
    wd = FakeWD({"Q1": (ent("Q1", "Mysore", wv="Mysore"), [Ancestor("Q1185", "Karnataka", False)],
                        ["city"])}, hits={"Mysuru": ["Q1"]}, titles={"Mysore": "Q1"})
    r = resolve(wd, ListedPlace("Mysuru", "KA", "r", aliases=("Mysore",)))
    assert r.candidate.name == "Mysuru" and "Mysore" in r.candidate.aliases


def test_district_item_is_flagged_and_only_used_when_nothing_better():
    wd = FakeWD(
        {"Q1": (ent("Q1", "Wayanad district", aliases=["Wayanad"]), [KL], ["district of India"])},
        hits={"Wayanad": ["Q1"]},
    )
    r = resolve(wd, ListedPlace("Wayanad", "KL", "r"))
    assert r.ok and r.flags[0].startswith("DISTRICT ITEM")
    better = FakeWD(
        {"Q1": (ent("Q1", "Wayanad district", aliases=["Wayanad"]), [KL], ["district of India"]),
         "Q2": (ent("Q2", "Wayanad", wv="Wayanad"), [KL], ["region"])},
        hits={"Wayanad": ["Q1", "Q2"]}, titles={"Wayanad": "Q2"},
    )
    r = resolve(better, ListedPlace("Wayanad", "KL", "r"))
    assert r.candidate.qid == "Q2" and not any("DISTRICT" in f for f in r.flags)


def test_wrong_state_is_flagged_and_homonym_in_the_right_state_wins():
    wd = FakeWD({"Q1": (ent("Q1", "Auroville", wv="Auroville"), [TN], ["city"])},
                hits={"Auroville": ["Q1"]}, titles={"Auroville": "Q1"})
    r = resolve(wd, ListedPlace("Auroville", "PY", "r"))
    assert r.candidate.state == "TN" and "WRONG STATE: list says PY, Wikidata says TN" in r.flags
    two = FakeWD(
        {"Q1": (ent("Q1", "Kovalam", wv="Kovalam"), [KL], ["beach"]),
         "Q2": (ent("Q2", "Kovalam"), [TN], ["village"])},
        hits={"Kovalam": ["Q1", "Q2"]}, titles={"Kovalam": "Q1"},
    )
    r = resolve(two, ListedPlace("Kovalam", "KL", "r"))
    assert r.candidate.qid == "Q1" and r.flags == []  # the one in the listed state wins


def test_item_spanning_two_states_keeps_the_listed_state_and_says_so():
    wd = FakeWD({"Q1": (ent("Q1", "Auroville", wv="Auroville"), [TN, PY], ["city"])},
                hits={"Auroville": ["Q1"]}, titles={"Auroville": "Q1"})
    r = resolve(wd, ListedPlace("Auroville", "PY", "r"))
    assert r.candidate.state == "PY"
    assert r.flags == ["SPANS STATES TN+PY in Wikidata; using PY"]  # no WRONG STATE flag
    r = resolve(wd, ListedPlace("Auroville", "TN", "r"))
    assert r.candidate.state == "TN"


def test_real_homonyms_fail_instead_of_guessing_and_a_pin_fixes_them():
    items = {"Q1": (ent("Q1", "Ananthagiri"), [TS], ["hill"]),
             "Q2": (ent("Q2", "Ananthagiri"), [TS], ["village"])}
    wd = FakeWD(items, hits={"Ananthagiri": ["Q1", "Q2"]})
    r = resolve(wd, ListedPlace("Ananthagiri", "TS", "r"))
    assert not r.ok and r.failure.startswith("HOMONYMS") and len(r.alternatives) == 2
    pinned = resolve(wd, ListedPlace("Ananthagiri", "TS", "r", wikidata_id="Q2"))
    assert pinned.candidate.qid == "Q2"


def test_union_territory_item_loses_to_the_city_for_pondicherry():
    wd = FakeWD(
        {"Q1": (ent("Q1", "Puducherry", aliases=["Pondicherry"]), [PY],
                 ["union territory of India"]),
         "Q2": (ent("Q2", "Pondicherry", wv="Pondicherry"), [PY], ["city"])},
        hits={"Pondicherry": ["Q2"], "Puducherry": ["Q1"]}, titles={"Pondicherry": "Q2"},
    )
    r = resolve(wd, ListedPlace("Pondicherry", "PY", "r", aliases=("Puducherry",)))
    assert r.candidate.qid == "Q2" and r.candidate.state == "PY" and r.flags == []


def test_unlabelled_item_found_through_its_wikivoyage_title_is_exact():
    wd = FakeWD({"Q9": (ent("Q9", "", wv="Chikmagalur", wp="Chikmagalur"), [Ancestor(
        "Q1185", "Karnataka", False)], ["city"])}, titles={"Chikmagalur": "Q9"})
    r = resolve(wd, ListedPlace("Chikmagalur", "KA", "r"))
    assert r.ok and r.candidate.qid == "Q9"


def test_nothing_in_scope_fails_with_a_reason():
    wd = FakeWD({"Q1": (ent("Q1", "Panaji"), [Ancestor("Q1", "Goa", False)], ["city"])},
                hits={"Panaji": ["Q1"]})
    r = resolve(wd, ListedPlace("Panaji", "KA", "r"))
    assert not r.ok and "inside TN/KL/KA/AP/TS/PY" in r.failure


def test_low_listing_missing_article_and_missing_signal_flags():
    wd = FakeWD(
        {"Q1": (ent("Q1", "Thin", wv="Thin"), [KL], ["town"]),
         "Q2": (ent("Q2", "NoWv"), [KL], ["town"]),
         "Q3": (ent("Q3", "NoWp", wv="Munnar", wp=""), [KL], ["town"])},
        hits={"Thin": ["Q1"], "NoWv": ["Q2"], "NoWp": ["Q3"]},
        titles={"Thin": "Q1"}, views={"Thin": 5, "NoWv": 5},
    )
    flags = {n: resolve(wd, ListedPlace(n, "KL", "r")).flags for n in ("Thin", "NoWv", "NoWp")}
    assert any(f.startswith("<3 LISTINGS (1;") for f in flags["Thin"])
    assert any(f.startswith("NO WIKIVOYAGE ARTICLE") for f in flags["NoWv"])
    assert any(f.startswith("NO ENGLISH WIKIPEDIA") for f in flags["NoWp"])
    nodata = FakeWD({"Q1": (ent("Q1", "Quiet", wv="Thin"), [KL], ["town"])},
                    hits={"Quiet": ["Q1"]}, titles={"Quiet": "Q1"}, views={"Quiet": None})
    assert any(f.startswith("NO PAGEVIEW DATA") for f in resolve(
        nodata, ListedPlace("Quiet", "KL", "r")).flags)


def test_hub_flag_comes_from_the_list_and_mismatch_with_the_auto_rule_is_reported():
    wd = FakeWD({"Q1": (ent("Q1", "Big", wv="Munnar", pop=2_000_000), [KL], ["city"])},
                hits={"Big": ["Q1"]}, titles={"Big": "Q1"})
    r = resolve(wd, ListedPlace("Big", "KL", "r", hub=False))
    assert r.candidate.hub is False and any(f.startswith("hub: list says False") for f in r.flags)
    r = resolve(wd, ListedPlace("Big", "KL", "r", hub=True))
    assert r.candidate.hub is True and r.flags == []


# ---- explicit hold-outs and place_type ------------------------------------------------------
def test_holdout_and_place_type_flow_from_the_list_to_the_candidate_and_registry_entry():
    wd = FakeWD({"Q1": (ent("Q1", "Kodagu district", aliases=["Kodagu"]),
                        [Ancestor("Q1185", "Karnataka", False)], ["district of India"])},
                hits={"Kodagu": ["Q1"]})
    r = resolve(wd, ListedPlace("Kodagu", "KA", "r", holdout=True, place_type="region"))
    assert r.candidate.holdout is True and r.candidate.place_type == "region"
    e = to_entry(r.candidate, "popular")
    assert (e.holdout, e.place_type, e.core_source) == (True, "region", "list")
    plain = resolve(wd, ListedPlace("Kodagu", "KA", "r"))
    assert plain.candidate.holdout is False
    assert to_entry(plain.candidate, "popular").holdout is None


def test_loader_reads_holdout_and_place_type(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"places": [
        {"name": "A", "state": "KL", "reason": "r", "holdout": True, "place_type": "region"},
        {"name": "B", "state": "KL", "reason": "r"}]}))
    a, b = load_core_list(p)
    assert (a.holdout, a.place_type, b.holdout, b.place_type) == (True, "region", False, None)


# ---- real response shapes (regression for the bugs the fakes hid) ---------------------------
def wd_with(routes, tmp_path):
    def transport(method, url, headers, timeout):
        for frag, body in routes.items():
            if frag in url:
                return TransportResponse(200, {}, json.dumps(body))
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path), "T/1 (t@example.com)", transport=transport,
                      sleep=lambda s: None, min_delay_s=0)
    return WikidataClient(http)


def test_district_flag_comes_from_an_optional_row_not_an_exists_expression(tmp_path):
    wd = wd_with({"sparql": {"results": {"bindings": [
        {"item": {"value": "http://www.wikidata.org/entity/Q1"},
         "anc": {"value": "http://www.wikidata.org/entity/Q1"}, "ancLabel": {"value": "Munnar"}},
        {"item": {"value": "http://www.wikidata.org/entity/Q1"},
         "anc": {"value": "http://www.wikidata.org/entity/Q2"},
         "ancLabel": {"value": "Idukki district"}, "dist": {"value": "yes"}},
        {"item": {"value": "http://www.wikidata.org/entity/Q1"},
         "anc": {"value": "http://www.wikidata.org/entity/Q2"},
         "ancLabel": {"value": "Idukki district"}},  # a second row for the same ancestor
        {"item": {"value": "http://www.wikidata.org/entity/Q1"},
         "anc": {"value": "http://www.wikidata.org/entity/Q1186"},
         "ancLabel": {"value": "Kerala"}}]}}}, tmp_path)
    anc = wd.ancestors_many(["Q1"])["Q1"]
    assert [(a.qid, a.is_district) for a in anc] == [("Q1", False), ("Q2", True), ("Q1186", False)]
    assert state_from_ancestors(anc) == "KL"


def test_entity_label_falls_back_to_the_language_neutral_mul_label(tmp_path):
    wd = wd_with({"wbgetentities": {"entities": {"Q980917": {
        "labels": {"mul": {"language": "mul", "value": "Chikmagalur"}},
        "aliases": {"mul": [{"value": "Chikkamagaluru"}]},
        "sitelinks": {"enwikivoyage": {"title": "Chikmagalur"}}}}}}, tmp_path)
    e = wd.entities_by_qids(["Q980917"])["Q980917"]
    assert e.label == "Chikmagalur" and e.aliases == ["Chikkamagaluru"]



def ov(name, reason="r"):
    return Override(name, reason)

STATE = """intro
== Regions ==
===North===
{{Regionlist
| region1name=[[Malabar]]
| region1color=#c9bd79
|region2name = [[Chola Nadu]]
}}
== Cities ==
* {{marker|type=city|zoom=13
| name=[[Thiruvananthapuram]]
|lat=8.4 |long=76.9
|wikidata=Q167715}} (Trivandrum) &mdash; capital
* {{marker|zoom=13
| name=[[Bekal|Bekal Fort]]
| lat=12.39 |long=75.03
}} &mdash; a fort
* {{marker|name=Eravikulam National Park|lat=10.1|long=77.0}} &mdash; park
== Other destinations ==
* {{marker|name=[[Athirappilly Falls]]|wikidata=Q1}} &mdash; falls
== Understand ==
* {{marker|name=[[Not Included]]}}
"""

CITY = """== See ==
{{see|name=A}}
{{See | name=B}}
=== Temples ===
{{listing|type=see|name=C}}
== Do ==
{{do|name=D}}
== Eat ==
{{eat|name=not counted}}
"""
PROSE = "== See ==\n" + ("Beautiful beaches and old temples line the coast. " * 20)
THIN = "== See ==\nA small shrine."

def cand(qid, state, views, hub=False, name=None):
    return CoreCandidate(qid=qid, name=name or qid, wp_title=qid, wv_title=qid, state=state,
                         lat=10, lon=76, types=["town"], aliases=[], district=None,
                         population=None, views=views, hub=hub)


def test_marker_parsing_links_display_names_coordinates_and_unlinked():
    cities = parse_markers(section_text(STATE, "Cities"), "Kerala", "Cities")
    assert [(m.title, m.display, m.wikidata, m.lat) for m in cities] == [
        ("Thiruvananthapuram", "Thiruvananthapuram", "Q167715", 8.4),
        ("Bekal", "Bekal Fort", None, 12.39),
        (None, "Eravikulam National Park", None, 10.1),
    ]
    other = parse_markers(section_text(STATE, "Other destinations"), "Kerala", "Other")
    assert [m.title for m in other] == ["Athirappilly Falls"]  # "Understand" is not read


def test_region_links_include_nested_regionlists():
    assert region_links(STATE) == ["Malabar", "Chola Nadu"]


def test_count_see_do_counts_only_see_and_do_listings_including_subsections():
    assert count_see_do(CITY) == 4
    assert count_see_do("== Understand ==\ntext") == 0


def test_count_see_do_handles_combined_see_and_do_heading():
    text = "== See and do ==\n=== Temples ===\n{{see|name=A}}\n{{Do|name=B}}\n{{see|name=C}}"
    assert count_see_do(text) == 3


def test_filter_passes_on_listings_or_on_prose_words():
    assert see_do_words(PROSE) == 160
    assert passes_tourism_filter(PROSE) == (True, 0, 160)  # prose only, no listing templates
    assert passes_tourism_filter("== See ==\n{{see|a}}{{see|b}}{{see|c}}")[0] is True
    assert passes_tourism_filter(THIN)[0] is False
    assert passes_tourism_filter("== Eat ==\n" + "word " * 500)[0] is False  # other sections
    just_under = "== Do ==\n" + "word " * 119
    assert passes_tourism_filter(just_under)[0] is False
    assert passes_tourism_filter("== Do ==\n" + "word " * 120)[0] is True


def test_see_do_words_ignores_markup_comments_refs_images_and_templates():
    text = (
        "== See and do ==\n=== Temples ===\n<!-- hidden note -->\n"
        "[[File:x.jpg|thumb|a caption that is long]]\n"
        "{{see|name=Fort|content=ignored words here}}\n"
        "Visit the [[Meenakshi Temple|great temple]] today.<ref>citation text</ref>\n"
    )
    assert see_do_words(text) == 5  # Visit the great temple today


def test_override_variants_and_matching():
    o = ov("Kodagu (Coorg)")
    assert o.variants == ["Kodagu (Coorg)", "Kodagu", "Coorg"]
    assert o.keys == {"kodagu coorg", "kodagu", "coorg"}


def test_overrides_file_is_validated(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"include": [{"name": "A", "reason": "x"}],
                             "exclude": [{"name": "B", "reason": "y"}]}))
    got = load_overrides(p)
    assert [o.name for o in got.include] == ["A"] and [o.name for o in got.exclude] == ["B"]
    bad_files = (
        {"include": [{"name": "A"}]},
        {"include": "A"},
        {"other": []},
        {"include": [{"name": "A", "reason": "x"}], "exclude": [{"name": "A", "reason": "y"}]},
    )
    for bad in bad_files:
        p.write_text(json.dumps(bad))
        with pytest.raises(OverridesError):
            load_overrides(p)
    assert load_overrides(tmp_path / "missing.json").include == []


def test_shipped_overrides_file_loads_with_the_agreed_lists():
    o = load_overrides()
    assert len(o.include) == 16 and "Mahabalipuram" in {x.name for x in o.include}
    assert {x.name for x in o.exclude} == {"Palakkad", "Mancherial", "Arikamedu", "Khammam"}


def test_hub_flag_ignores_districts_and_small_places():
    assert is_hub(["big city"], None) and is_hub(["city"], 2_000_000)
    assert not is_hub(["district of India"], 2_000_000)
    assert not is_hub(["town"], 50_000)


def test_to_entry_carries_provenance_and_flags():
    c = cand("Q7", "PY", 50, hub=True)
    c.holdout, c.n_see_do, c.see_do_words, c.source = True, 12, 300, "include"
    e = to_entry(c, "hidden_gem")
    assert (e.state.value, e.hub, e.holdout, e.see_do_listings) == ("PY", True, True, 12)
    assert (e.see_do_words, e.core_source) == (300, "include")
    assert e.popularity_source == "wikipedia_pageviews_12m" and e.popularity_signal == 50.0
    assert to_entry(cand("Q8", "KL", 5), "popular").holdout is None
