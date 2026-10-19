"""The explicit core place list (config/core_places_list.json, D17) and its resolution.

Every listed place is resolved to a Wikidata item, then checked rather than trusted:
- the item must have coordinates and sit inside TN/KL/KA/AP/TS/PY (P131 chain);
- a district-like item (district, taluk, mandal, division, ...) is accepted only when nothing better
  exists, and is flagged;
- an item whose state differs from the listed state is flagged;
- two equally good items with the same name (homonyms) are never guessed between: the place fails
  and the candidates are shown. A `wikidata_id` in the list file pins the item.
Labels come from the pool threshold computed elsewhere (bootstrap.popularity_threshold) and are
never set by hand. Hub status is the list's, reported next to the automatic hub rule.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src.common.config import REPO_ROOT
from src.common.schemas import PlaceRegistryEntry, State
from src.places.registry import normalize
from src.places.wikidata import Entity, WikidataClient, district_of, states_from_ancestors
from src.places.wv_lists import passes_tourism_filter

DEFAULT_PATH = REPO_ROOT / "config" / "core_places_list.json"

# ---- moved from the removed src/places/bootstrap.py (automatic core-list experiment): only the
# helpers this module, scripts/historical/build_core_list.py and the tests still use.
POPULAR_SHARE = 0.2

HUB_POPULATION = 1_000_000

HUB_TYPES = {"big city", "megacity", "metropolis", "largest city", "state capital"}

SIGNAL = "wikipedia_pageviews_12m"


@dataclass
class Candidate:
    qid: str
    name: str
    wp_title: str | None
    wv_title: str | None
    state: str
    lat: float
    lon: float
    types: list[str]
    aliases: list[str]
    district: str | None
    population: float | None
    listed_in: str = ""
    n_see_do: int | None = None  # None = not countable (no Wikivoyage article)
    see_do_words: int | None = None
    views: int | None = None
    hub: bool = False
    holdout: bool = False
    source: str = "auto"  # "auto", "include" or "list"
    reason: str = ""
    place_type: str | None = None
    include_name: str | None = None


def is_hub(types: list[str], population: float | None) -> bool:
    lowered = {t.casefold() for t in types}
    if lowered & HUB_TYPES:
        return True
    is_district = any("district" in t for t in lowered)
    return not is_district and (population or 0) >= HUB_POPULATION


def label_for(views: int | None, threshold: float) -> str:
    return "popular" if (views or 0) >= threshold else "hidden_gem"


def to_entry(c: Candidate, popularity: str) -> PlaceRegistryEntry:
    cluster = f"{c.state}-{c.district}" if c.district else f"{c.state}-unclustered"
    return PlaceRegistryEntry(
        name=c.name,
        aliases=sorted({a for a in [*c.aliases, c.wp_title or "", c.wv_title or ""]
                        if a and a != c.name}),
        state=c.state,
        region_cluster=cluster,
        popularity=popularity,
        lat=c.lat,
        lon=c.lon,
        hub=c.hub,
        place_type=c.place_type,
        holdout=True if c.holdout else None,
        wikidata_id=c.qid,
        wikipedia_title=c.wp_title,
        wikivoyage_title=c.wv_title,
        see_do_listings=c.n_see_do,
        see_do_words=c.see_do_words,
        core_source=c.source if c.source in ("auto", "include", "list") else None,
        core_reason=c.reason or None,
        coord_source="wikidata:P625",
        region_cluster_source="wikidata:P131 district" if c.district else "state only",
        popularity_source=SIGNAL,
        popularity_signal=float(c.views) if c.views is not None else None,
    )
CORE_HUB_CAP = 6
SEARCH_LIMIT = 10
# Administrative regions: "district of India", "census district", "taluk of Kerala", "division of
# Karnataka", "union territory of India", ... but NOT "district headquarters" (a real city).
_ADMIN_TYPE = re.compile(
    r"^((revenue |census )?district|taluk|mandal|tehsil|(revenue )?division|subdivision|"
    r"union territory)( of .*)?$|( district| taluk| mandal| division)$"
)
# Items that are not places to visit: constituencies, airports, wards, railway stations.
_NON_PLACE_TYPE = re.compile(
    r"constituency|airport|aerodrome|electoral|^ward( of .*)?$|railway station"
)
LOW_LISTINGS = 3


class CoreListError(ValueError):
    pass


@dataclass(frozen=True)
class ListedPlace:
    name: str
    state: str
    reason: str
    hub: bool = False
    aliases: tuple[str, ...] = ()
    wikidata_id: str | None = None
    holdout: bool = False  # explicit, chosen by the project owner (D17)
    place_type: str | None = None  # e.g. "region" for district-level items (D21)

    @property
    def names(self) -> list[str]:
        return [self.name, *self.aliases]


def load_core_list(path: str | Path = DEFAULT_PATH) -> list[ListedPlace]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"places"} or not isinstance(doc["places"], list):
        raise CoreListError('file must be {"places": [...]}')
    out: list[ListedPlace] = []
    for i, it in enumerate(doc["places"]):
        if not isinstance(it, dict):
            raise CoreListError(f"places[{i}] must be an object")
        extra = set(it) - {"name", "state", "reason", "hub", "aliases", "wikidata_id", "holdout",
                           "place_type"}
        if extra:
            raise CoreListError(f"places[{i}] has unknown keys {sorted(extra)}")
        name, state, reason = (str(it.get(k, "")).strip() for k in ("name", "state", "reason"))
        if not name or not reason:
            raise CoreListError(f"places[{i}] needs a name and a reason")
        if state not in {s.value for s in State}:
            raise CoreListError(f"{name}: unknown state {state!r}")
        aliases = tuple(str(a).strip() for a in it.get("aliases", []) if str(a).strip())
        place_type = str(it["place_type"]).strip() if it.get("place_type") else None
        out.append(ListedPlace(name, state, reason, bool(it.get("hub", False)), aliases,
                               it.get("wikidata_id"), bool(it.get("holdout", False)), place_type))
    names = [normalize(p.name) for p in out]
    if len(set(names)) != len(names):
        raise CoreListError("duplicate names in the core list")
    hubs = sum(p.hub for p in out)
    if hubs > CORE_HUB_CAP:
        raise CoreListError(f"{hubs} hubs listed, the cap is {CORE_HUB_CAP}")
    return out


@dataclass
class Resolved:
    place: ListedPlace
    candidate: Candidate | None = None
    flags: list[str] = field(default_factory=list)
    failure: str | None = None
    alternatives: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.candidate is not None


def _district_like(types: list[str]) -> bool:
    return any(_ADMIN_TYPE.search(t.casefold()) for t in types)


def _non_place(types: list[str]) -> bool:
    return bool(types) and all(_NON_PLACE_TYPE.search(t.casefold()) for t in types) or any(
        "constituency" in t.casefold() for t in types
    )


@dataclass
class _Opt:
    ent: Entity
    state: str
    types: list[str]
    exact: bool
    ancestors: list
    states: list[str]

    @property
    def desc(self) -> str:
        return f"{self.ent.qid} {self.ent.label} [{self.state}; {', '.join(self.types[:3]) or '?'}]"

    @property
    def rank(self) -> tuple[int, int]:
        return (1 if self.ent.wikivoyage_title else 0, 1 if self.ent.wikipedia_title else 0)


def resolve_listed(
    wd: WikidataClient,
    p: ListedPlace,
    wikitext_of: Callable[[str], str | None],
    as_of: date,
) -> Resolved:
    qids: list[str] = [p.wikidata_id] if p.wikidata_id else []
    by_title: dict[str, Entity] = {}
    if not p.wikidata_id:
        for n in p.names:
            qids += [c.qid for c in wd.search(n, limit=SEARCH_LIMIT)]
        by_title = wd.entities_by_wikivoyage_titles(list(p.names))
        qids += [e.qid for e in by_title.values()]
    qids = list(dict.fromkeys(qids))
    ents = wd.entities_by_qids(qids) if qids else {}
    located = [q for q in qids if q in ents and ents[q].lat is not None]
    ancestors = wd.ancestors_many(located) if located else {}
    types = wd.types_many(located) if located else {}
    want = {normalize(n) for n in p.names}
    titled = {e.qid for e in by_title.values()} if not p.wikidata_id else set()
    opts: list[_Opt] = []
    for q in located:
        e = ents[q]
        states = states_from_ancestors(ancestors.get(q, []))
        if not states:
            continue
        state = p.state if p.state in states else states[0]  # a spanning item keeps the listed one
        names = {normalize(e.label), *(normalize(a) for a in e.aliases),
                 normalize(e.wikivoyage_title or ""), normalize(e.wikipedia_title or "")}
        exact = bool(want & names) or q == p.wikidata_id or q in titled
        opts.append(_Opt(e, state, types.get(q, []), exact, ancestors.get(q, []), states))
    res = Resolved(p)
    exact = [o for o in opts if o.exact and not _non_place(o.types)]
    tiers = [
        ("", [o for o in exact if o.state == p.state and not _district_like(o.types)]),
        ("DISTRICT ITEM", [o for o in exact if o.state == p.state and _district_like(o.types)]),
        ("WRONG STATE", [o for o in exact if o.state != p.state and not _district_like(o.types)]),
        ("WRONG STATE + DISTRICT ITEM",
         [o for o in exact if o.state != p.state and _district_like(o.types)]),
    ]
    chosen: _Opt | None = None
    flag = ""
    for flag, tier in tiers:
        if not tier:
            continue
        best = max(o.rank for o in tier)
        top = [o for o in tier if o.rank == best]
        if len(top) == 1:
            chosen = top[0]
        else:
            res.failure = "HOMONYMS: " + "; ".join(o.desc for o in top)
            res.alternatives = [o.desc for o in top]
            return res
        break
    if chosen is None:
        why = "no Wikidata item with this name inside TN/KL/KA/AP/TS/PY"
        res.failure = why
        res.alternatives = [o.desc for o in opts[:5]]
        return res
    if flag:
        res.flags.append(flag + (f": list says {p.state}, Wikidata says {chosen.state}"
                                 if "WRONG STATE" in flag else ""))
    res.alternatives = [o.desc for o in opts if o.exact and o is not chosen]
    if len(chosen.states) > 1:
        spans = "+".join(chosen.states)
        res.flags.append(f"SPANS STATES {spans} in Wikidata; using {chosen.state}")
    ent = chosen.ent
    counts = None
    if ent.wikivoyage_title:
        text = wikitext_of(ent.wikivoyage_title)
        counts = passes_tourism_filter(text) if text is not None else None
    district = district_of(chosen.ancestors)
    views = wd.pageviews_12m(ent.wikipedia_title, as_of) if ent.wikipedia_title else None
    cand = Candidate(
        qid=ent.qid, name=p.name, wp_title=ent.wikipedia_title,
        wv_title=ent.wikivoyage_title, state=chosen.state, lat=ent.lat, lon=ent.lon,
        types=chosen.types,
        aliases=sorted((set(ent.aliases) | set(p.aliases) | {ent.label}) - {p.name} - {""}),
        district=district, population=ent.population, listed_in="config/core_places_list.json",
        n_see_do=counts[1] if counts else None, see_do_words=counts[2] if counts else None,
        views=views, hub=p.hub, source="list", reason=p.reason, holdout=p.holdout,
        place_type=p.place_type,
    )
    res.candidate = cand
    if not ent.wikipedia_title:
        res.flags.append("NO ENGLISH WIKIPEDIA ARTICLE (no signal)")
    elif views is None:
        res.flags.append("NO PAGEVIEW DATA (no signal)")
    if counts is None:
        res.flags.append("NO WIKIVOYAGE ARTICLE (listings n/a)")
    elif counts[1] < LOW_LISTINGS:
        res.flags.append(f"<{LOW_LISTINGS} LISTINGS ({counts[1]}; {counts[2]} words)")
    if p.hub != is_hub(chosen.types, ent.population):
        res.flags.append(f"hub: list says {p.hub}, automatic rule says {not p.hub}")
    return res


# ---- merged from src/places/overrides.py: config/core_overrides.json, the project owner's
# includes and excludes for the core list (D19).

OVERRIDES_PATH = REPO_ROOT / "config" / "core_overrides.json"


class OverridesError(ValueError):
    pass


@dataclass(frozen=True)
class Override:
    name: str
    reason: str

    @property
    def variants(self) -> list[str]:
        """'Kodagu (Coorg)' -> ['Kodagu (Coorg)', 'Kodagu', 'Coorg'] (de-duplicated, in order)."""
        out = [self.name]
        base = re.sub(r"\s*\(.*?\)", "", self.name).strip()
        inner = re.findall(r"\((.*?)\)", self.name)
        for v in [base, *inner]:
            if v and v not in out:
                out.append(v)
        return out

    @property
    def keys(self) -> set[str]:
        return {normalize(v) for v in self.variants if normalize(v)}


@dataclass(frozen=True)
class Overrides:
    include: list[Override]
    exclude: list[Override]


def _entries(doc: dict, key: str) -> list[Override]:
    items = doc.get(key, [])
    if not isinstance(items, list):
        raise OverridesError(f"{key!r} must be a list")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not str(it.get("name", "")).strip() or not str(
            it.get("reason", "")
        ).strip():
            raise OverridesError(f"{key}[{i}] needs a non-empty name and reason")
        out.append(Override(it["name"].strip(), it["reason"].strip()))
    return out


def load_overrides(path: str | Path = OVERRIDES_PATH) -> Overrides:
    p = Path(path)
    if not p.exists():
        return Overrides([], [])
    doc = json.loads(p.read_text(encoding="utf-8"))
    unknown = set(doc) - {"include", "exclude"}
    if unknown:
        raise OverridesError(f"unknown keys: {sorted(unknown)}")
    inc, exc = _entries(doc, "include"), _entries(doc, "exclude")
    clash = {o.name for o in inc} & {o.name for o in exc}
    if clash:
        raise OverridesError(f"in both include and exclude: {sorted(clash)}")
    return Overrides(inc, exc)
