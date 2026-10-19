"""Tests for src/places: registry, Wikidata lookups, resolver (D47)."""
from __future__ import annotations

import json
from datetime import date

from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry, State
from src.places.places import (
    Ancestor,
    Candidate,
    Entity,
    Registry,
    WikidataClient,
    normalize,
    pageview_window,
    resolve_place,
    slugify,
    state_from_ancestors,
)

# ==== from test_places_resolve.py


AS_OF = date(2026, 10, 18)


def entry(name, state="KL", aliases=(), qid=None, **kw):
    return PlaceRegistryEntry(
        name=name, aliases=list(aliases), state=state, region_cluster=f"{state}-x",
        popularity="popular", lat=10.0, lon=76.0, wikidata_id=qid, **kw,
    )


class FakeWD:
    """Duck-typed WikidataClient: items = {qid: (label, desc, Entity, [Ancestor])}."""

    def __init__(self, items, views=None):
        self.items = items
        self.views = views or {}
        self.calls = 0

    def search(self, text, limit=7):
        self.calls += 1
        k = normalize(text)
        return [Candidate(q, v[0], v[1]) for q, v in self.items.items()
                if k in normalize(v[0])]

    def entity(self, qid):
        self.calls += 1
        return self.items[qid][2]

    def ancestors(self, qid):
        self.calls += 1
        return self.items[qid][3]

    def pageviews_12m(self, title, as_of):
        self.calls += 1
        return self.views.get(title)


def ent(qid, label, lat=10.0, lon=77.0, title=None, aliases=()):
    return Entity(qid=qid, label=label, aliases=list(aliases), lat=lat, lon=lon,
                  wikivoyage_title=title, wikipedia_title=title)


KERALA = Ancestor("Q1186", "Kerala", False)


TN = Ancestor("Q1445", "Tamil Nadu", False)


IDUKKI = Ancestor("Q2", "Idukki", True)


def test_normalize_and_slug():
    assert normalize("  Pondichéry! ") == "pondichery"
    assert slugify("Fort Kochi") == "fort_kochi"


def test_registry_lookup_by_alias_and_roundtrip(tmp_path):
    reg = Registry([entry("Udhagamandalam", "TN", ["Ooty", "Ootacamund"], "Q1")], {"a": 1})
    assert reg.find("ooty")[0].name == "Udhagamandalam"
    assert reg.get("Ooty") is None  # get() is canonical-name only
    reg.save(tmp_path / "r.json")
    again = Registry.load(tmp_path / "r.json")
    assert again.find("OOTACAMUND")[0].wikidata_id == "Q1" and again.meta == {"a": 1}
    assert Registry.load(tmp_path / "missing.json").entries == []


def test_puducherry_is_its_own_state_including_mahe_yanam_karaikal():
    assert state_from_ancestors([KERALA]) == "KL"
    pudu = Ancestor("Q66743", "Puducherry", False)
    assert state_from_ancestors([pudu]) == "PY"
    for district in ("Mahe district", "Yanam district", "Karaikal district"):
        assert state_from_ancestors([pudu, Ancestor("Q9", district, True)]) == "PY"
    assert state_from_ancestors([Ancestor("Q1", "Goa", False)]) is None
    assert "PY" in {s.value for s in State}


def test_puducherry_place_resolves_with_state_py():
    items = {"Q40": ("Pondicherry", "city in Puducherry", ent("Q40", "Pondicherry"),
                     [Ancestor("Q66743", "Puducherry", False)])}
    res = resolve_place("Pondicherry", [Registry()], FakeWD(items), AS_OF)
    assert res.status == "resolved" and res.entry.state.value == "PY"


def test_pageview_window():
    assert pageview_window(date(2026, 10, 18)) == ("2025100100", "2026093000")
    assert pageview_window(date(2026, 1, 5)) == ("2025010100", "2025123100")


def test_registry_hit_makes_no_wikidata_call():
    wd = FakeWD({})
    reg = Registry([entry("Munnar", aliases=["Moonar"])])
    res = resolve_place("moonar", [reg], wd, AS_OF)
    assert res.status == "resolved" and res.entry.name == "Munnar" and wd.calls == 0
    assert not res.network and not res.new


def test_new_place_is_built_from_wikidata_with_provenance():
    items = {"Q5": ("Gokarna", "town in Karnataka",
                    ent("Q5", "Gokarna", 14.5, 74.3, "Gokarna", ["Gokarn"]),
                    [Ancestor("Q1185", "Karnataka", False),
                     Ancestor("Q3", "Uttara Kannada", True)])}
    wd = FakeWD(items, views={"Gokarna": 900})
    res = resolve_place("Gokarna", [Registry()], wd, AS_OF, popular_cutoff=500)
    e = res.entry
    assert res.status == "resolved" and res.new and res.network
    assert (e.state.value, e.region_cluster, e.popularity) == ("KA", "KA-Uttara Kannada", "popular")
    assert e.coord_source == "wikidata:P625" and e.popularity_signal == 900.0
    assert "Gokarn" in e.aliases


def test_no_pageview_signal_is_recorded_as_unavailable():
    items = {"Q5": ("Tiny Village", "village in Kerala", ent("Q5", "Tiny Village"),
                    [KERALA])}
    res = resolve_place("Tiny Village", [Registry()], FakeWD(items), AS_OF, popular_cutoff=500)
    assert res.entry.popularity == "hidden_gem"
    assert res.entry.popularity_source == "unavailable" and res.entry.popularity_signal is None


def test_out_of_scope_place_is_rejected():
    items = {"Q7": ("Panaji", "city in Goa", ent("Q7", "Panaji"), [Ancestor("Q1", "Goa", False)])}
    res = resolve_place("Panaji", [Registry()], FakeWD(items), AS_OF)
    assert res.status == "rejected" and "Goa" in res.reason


def test_unlocated_things_are_not_found():
    items = {"Q9": ("Munnar", "2009 film", ent("Q9", "Munnar", lat=None, lon=None), [])}
    res = resolve_place("Munnar", [Registry()], FakeWD(items), AS_OF)
    assert res.status == "not_found"


def test_ambiguous_text_returns_candidates_and_picks_nothing():
    items = {
        "Q10": ("Kovalam", "beach in Kerala", ent("Q10", "Kovalam"), [KERALA]),
        "Q11": ("Kovalam", "village in Tamil Nadu", ent("Q11", "Kovalam"), [TN]),
    }
    res = resolve_place("Kovalam", [Registry()], FakeWD(items), AS_OF)
    assert res.status == "ambiguous" and res.entry is None
    assert {c["state"] for c in res.candidates} == {"KL", "TN"}


def test_exact_label_beats_partial_matches():
    items = {
        "Q20": ("Munnar", "town in Kerala", ent("Q20", "Munnar"), [KERALA, IDUKKI]),
        "Q21": ("Munnar taluk", "taluk in Kerala", ent("Q21", "Munnar taluk"), [KERALA]),
    }
    res = resolve_place("Munnar", [Registry()], FakeWD(items), AS_OF)
    assert res.status == "resolved" and res.entry.name == "Munnar"


def test_known_wikidata_id_reuses_existing_entry():
    known = entry("Udhagamandalam", "TN", ["Ooty"], "Q30")
    items = {"Q30": ("Ooty hill station", "town in Tamil Nadu", ent("Q30", "Ooty"), [TN])}
    wd = FakeWD(items)
    res = resolve_place("Ooty hill", [Registry([known])], wd, AS_OF)
    assert res.status == "resolved" and res.entry is known and not res.new
    assert wd.calls == 1  # search only: no entity/ancestor/pageview lookups for a known id


def test_name_collision_gets_qualified():
    reg = Registry([entry("Kovalam", "KL", qid="Q10")])
    items = {"Q11": ("Kovalam beach", "village in Tamil Nadu",
                     ent("Q11", "Kovalam", title=None), [TN])}
    res = resolve_place("Kovalam beach", [reg], FakeWD(items), AS_OF)
    assert res.entry.name.startswith("Kovalam (") and res.entry.state.value == "TN"


def make_wd(tmp_path, routes):
    def transport(method, url, headers, timeout):
        for frag, body in routes.items():
            if frag in url:
                return TransportResponse(200, {}, json.dumps(body))
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path), "T/1 (t@example.com)", transport=transport,
                      sleep=lambda s: None, min_delay_s=0)
    return WikidataClient(http), http


def test_wikidata_client_parses_entity_search_and_ancestors(tmp_path):
    routes = {
        "wbsearchentities": {"search": [{"id": "Q1", "label": "Hampi", "description": "village"}]},
        "wbgetentities": {"entities": {"Q1": {
            "labels": {"en": {"value": "Hampi"}},
            "aliases": {"en": [{"value": "Vijayanagara"}]},
            "claims": {"P625": [{"mainsnak": {"datavalue": {"value": {
                "latitude": 15.3, "longitude": 76.46}}}}]},
            "sitelinks": {"enwikivoyage": {"title": "Hampi"}},
        }}},
        "sparql": {"results": {"bindings": [
            {"item": {"value": "http://www.wikidata.org/entity/Q1"},
             "anc": {"value": "http://www.wikidata.org/entity/Q1185"},
             "ancLabel": {"value": "Karnataka"}},
            {"item": {"value": "http://www.wikidata.org/entity/Q1"},
             "anc": {"value": "http://www.wikidata.org/entity/Q9"},
             "ancLabel": {"value": "Vijayanagara district"}, "dist": {"value": "true"}}]}},
        "pageviews": {"items": [{"views": 100}, {"views": 50}]},
    }
    wd, http = make_wd(tmp_path, routes)
    assert wd.search("Hampi")[0].qid == "Q1"
    e = wd.entity("Q1")
    assert (e.lat, e.lon, e.wikivoyage_title, e.aliases) == (15.3, 76.46, "Hampi", ["Vijayanagara"])
    anc = wd.ancestors("Q1")
    assert state_from_ancestors(anc) == "KA"
    assert wd.pageviews_12m("Hampi", AS_OF) == 150
    n = http.requests_made
    wd.search("Hampi"), wd.entity("Q1"), wd.ancestors("Q1"), wd.pageviews_12m("Hampi", AS_OF)
    assert http.requests_made == n  # everything cached: a repeat resolution is free


def test_wikidata_error_body_raises_and_is_not_cached(tmp_path):
    from src.common.http import FetchError

    def transport(method, url, headers, timeout):
        return TransportResponse(200, {}, json.dumps({"error": {"code": "params-illegal"}}))

    http = HttpClient(DiskCache(tmp_path), "T/1 (t@example.com)", transport=transport,
                      sleep=lambda s: None, min_delay_s=0)
    import pytest as _pytest

    with _pytest.raises(FetchError):
        WikidataClient(http).search("x")
