"""Places: the registry (name/alias lookup), Wikidata lookups and the free-text resolver."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from src.common.cache import atomic_write_text
from src.common.http import HttpClient
from src.common.schemas import PlaceRegistryEntry

# ==== registry ====================================================================
# Place registry: name/alias lookup over PlaceRegistryEntry records, stored as JSON.
#
# Two registries exist and never mix on disk (D17): the committed snapshot registry
# (data/registry/core_places.json, written only by the bootstrap script) and the live registry
# (data/live/registry.json, grows as places are resolved on demand).

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Casefold, strip accents and punctuation: 'Pondichéry!' -> 'pondichery'."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", ascii_only.casefold()).strip()


def slugify(text: str) -> str:
    return normalize(text).replace(" ", "_")


class Registry:
    def __init__(self, entries: list[PlaceRegistryEntry] | None = None, meta: dict | None = None):
        self.meta: dict = dict(meta or {})
        self._entries: list[PlaceRegistryEntry] = []
        self._index: dict[str, list[int]] = {}
        for e in entries or []:
            self.add(e)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[PlaceRegistryEntry]:
        return list(self._entries)

    def add(self, entry: PlaceRegistryEntry) -> None:
        if self.get(entry.name) is not None:
            raise ValueError(f"duplicate registry name: {entry.name!r}")
        i = len(self._entries)
        self._entries.append(entry)
        for key in {normalize(entry.name), *(normalize(a) for a in entry.aliases)}:
            if key:
                self._index.setdefault(key, []).append(i)

    def get(self, name: str) -> PlaceRegistryEntry | None:
        """Exact canonical-name match (normalized)."""
        key = normalize(name)
        for i in self._index.get(key, []):
            if normalize(self._entries[i].name) == key:
                return self._entries[i]
        return None

    def find(self, text: str) -> list[PlaceRegistryEntry]:
        """All entries whose name or alias matches the text. More than one means ambiguous."""
        seen: list[PlaceRegistryEntry] = []
        for i in self._index.get(normalize(text), []):
            if self._entries[i] not in seen:
                seen.append(self._entries[i])
        return seen

    def by_wikidata_id(self, qid: str) -> PlaceRegistryEntry | None:
        return next((e for e in self._entries if e.wikidata_id == qid), None)

    def to_json(self) -> str:
        doc = {
            "meta": self.meta,
            "places": [e.model_dump(mode="json", exclude_none=True) for e in self._entries],
        }
        return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> None:
        atomic_write_text(Path(path), self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Registry:
        p = Path(path)
        if not p.exists():
            return cls()
        doc = json.loads(p.read_text(encoding="utf-8"))
        return cls([PlaceRegistryEntry(**e) for e in doc.get("places", [])], doc.get("meta"))


# ==== wikidata ====================================================================
# Wikidata / Wikimedia lookups used to resolve places and bootstrap the core list.
#
# All requests go through HttpClient (cached, polite). API endpoints are not subject to robots.txt,
# so
# check_robots is off for them; the User-Agent (built from CollectSettings only) carries a contact
# address as Wikimedia requires.

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"
PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/user/{title}/monthly/{start}/{end}"
)

# Scope: the five states plus the Union Territory of Puducherry (D18), which includes Karaikal,
# Mahe and Yanam. Anything whose administrative chain reaches one of these is in scope.
STATE_QIDS = {
    "Q1445": "TN",
    "Q1186": "KL",
    "Q1185": "KA",
    "Q1159": "AP",
    "Q677037": "TS",
    "Q66743": "PY",
}
DISTRICT_OF_INDIA_QID = "Q1149652"
SPARQL_BATCH = 40
PAGEVIEWS_PROJECT = "en.wikipedia"


@dataclass(frozen=True)
class Candidate:
    qid: str
    label: str
    description: str


@dataclass
class Entity:
    qid: str
    label: str
    aliases: list[str] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    wikivoyage_title: str | None = None
    wikipedia_title: str | None = None
    population: float | None = None


@dataclass(frozen=True)
class Ancestor:
    qid: str
    label: str
    is_district: bool


def states_from_ancestors(ancestors: list[Ancestor]) -> list[str]:
    """Every in-scope state the administrative chain reaches (an item can span two, e.g.
    Auroville in Tamil Nadu and Puducherry)."""
    qids = {a.qid for a in ancestors}
    return [code for qid, code in STATE_QIDS.items() if qid in qids]


def state_from_ancestors(ancestors: list[Ancestor]) -> str | None:
    """Map an administrative ancestor chain to TN/KL/KA/AP/TS/PY, or None if out of scope.
    Where the chain spans several states the first in STATE_QIDS order is returned."""
    states = states_from_ancestors(ancestors)
    return states[0] if states else None


def district_of(ancestors: list[Ancestor]) -> str | None:
    names = sorted(a.label for a in ancestors if a.is_district and a.label)
    return names[0] if names else None


def pageview_window(as_of: date) -> tuple[str, str]:
    """The 12 full calendar months before `as_of`, as YYYYMMDD00 strings for the REST API."""
    first_of_month = as_of.replace(day=1)
    end = first_of_month - timedelta(days=1)  # last day of the previous month
    start_year, start_month = end.year - 1, end.month + 1
    if start_month == 13:
        start_year, start_month = end.year, 1
    start = date(start_year, start_month, 1)
    return start.strftime("%Y%m%d00"), end.strftime("%Y%m%d00")


def _parse_entity(qid: str, ent: dict) -> Entity | None:
    if not ent or "missing" in ent:
        return None
    labels = ent.get("labels", {})  # many items carry only a language-neutral "mul" label
    label = (labels.get("en") or labels.get("mul") or {}).get("value", "")
    out = Entity(qid=qid, label=label)
    out.aliases = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
    out.aliases += [a["value"] for a in ent.get("aliases", {}).get("mul", [])]
    out.aliases = list(dict.fromkeys(out.aliases))
    claims = ent.get("claims", {})
    for c in claims.get("P625", [])[:1]:
        v = (c.get("mainsnak", {}).get("datavalue") or {}).get("value") or {}
        if "latitude" in v and "longitude" in v:
            out.lat, out.lon = float(v["latitude"]), float(v["longitude"])
    for c in claims.get("P1082", [])[:1]:
        v = (c.get("mainsnak", {}).get("datavalue") or {}).get("value") or {}
        if "amount" in v:
            out.population = float(v["amount"])
    sitelinks = ent.get("sitelinks", {})
    out.wikivoyage_title = (sitelinks.get("enwikivoyage") or {}).get("title")
    out.wikipedia_title = (sitelinks.get("enwiki") or {}).get("title")
    return out


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[1]


def _json_without_error(text: str) -> bool:
    """Wikimedia APIs answer errors with HTTP 200 and an \"error\" object: never cache those."""
    try:
        body = json.loads(text)
    except ValueError:
        return False
    return isinstance(body, dict) and "error" not in body


class WikidataClient:
    def __init__(self, http: HttpClient):
        self.http = http

    def _api(self, params: dict) -> dict:
        return self.http.fetch(
            WIKIDATA_API, {**params, "format": "json"}, check_robots=False,
            validate=_json_without_error,
        ).json()

    def sparql(self, query: str) -> list[dict]:
        res = self.http.fetch(
            SPARQL_URL, {"query": query, "format": "json"}, check_robots=False,
            validate=_json_without_error,
        )
        return res.json()["results"]["bindings"]

    def search(self, text: str, limit: int = 7) -> list[Candidate]:
        body = self._api({
            "action": "wbsearchentities", "search": text, "language": "en", "uselang": "en",
            "type": "item", "limit": limit,
        })
        return [
            Candidate(r["id"], r.get("label", ""), r.get("description", ""))
            for r in body.get("search", [])
        ]

    def entity(self, qid: str) -> Entity | None:
        body = self._api({
            "action": "wbgetentities", "ids": qid, "languages": "en|mul",
            "props": "labels|aliases|claims|sitelinks", "sitefilter": "enwikivoyage|enwiki",
        })
        return _parse_entity(qid, (body.get("entities") or {}).get(qid) or {})


    def ancestors(self, qid: str) -> list[Ancestor]:
        return self.ancestors_many([qid]).get(qid, [])

    def ancestors_many(self, qids: list[str]) -> dict[str, list[Ancestor]]:
        out: dict[str, list[Ancestor]] = {q: [] for q in qids}
        for i in range(0, len(qids), SPARQL_BATCH):
            values = " ".join(f"wd:{q}" for q in qids[i:i + SPARQL_BATCH])
            # The endpoint does not return an EXISTS expression in SELECT, so the district flag
            # is an OPTIONAL pattern: a row only has ?dist when the ancestor is (a subclass of)
            # "district of India".
            rows = self.sparql(
                "SELECT ?item ?anc ?ancLabel ?dist WHERE { "
                f"VALUES ?item {{ {values} }} ?item wdt:P131* ?anc. "
                f"OPTIONAL {{ ?anc wdt:P31/wdt:P279* wd:{DISTRICT_OF_INDIA_QID}. "
                'BIND("yes" AS ?dist) } '
                'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }'
            )
            merged: dict[tuple[str, str], Ancestor] = {}
            for r in rows:
                item, anc = _qid(r["item"]["value"]), _qid(r["anc"]["value"])
                is_dist = "dist" in r
                prev = merged.get((item, anc))
                merged[(item, anc)] = Ancestor(
                    qid=anc, label=r.get("ancLabel", {}).get("value", ""),
                    is_district=is_dist or bool(prev and prev.is_district),
                )
            for (item, _), anc_obj in merged.items():
                out[item].append(anc_obj)
        return out


    def pageviews_12m(
        self, title: str, as_of: date, project: str = PAGEVIEWS_PROJECT
    ) -> int | None:
        """Sum of monthly user views over the 12 full months before `as_of` (English Wikipedia
        by default, D19). None if the article has no data (404)."""
        start, end = pageview_window(as_of)
        url = PAGEVIEWS_URL.format(
            project=project, title=quote(title.replace(" ", "_"), safe=""), start=start, end=end
        )
        res = self.http.fetch(url, check_robots=False)  # 5xx raises; only a 404 means "no data"
        if res.status != 200:
            return None
        return int(sum(i["views"] for i in res.json().get("items", [])))


# ==== resolve =====================================================================
# Free text -> canonical place. Registry and aliases first (no network), then Wikidata.
#
# Outcomes: resolved (an entry, existing or newly built), ambiguous (candidates, nothing is picked),
# rejected (a real place outside TN/KL/KA/AP/TS/PY), not_found. A new entry is only returned here;
# the
# caller (ensure_place) decides to store it in the live registry.

Status = Literal["resolved", "ambiguous", "rejected", "not_found"]
MAX_WIKIDATA_CANDIDATES = 7
POPULARITY_SIGNAL = "wikipedia_pageviews_12m"


@dataclass
class Resolution:
    status: Status
    entry: PlaceRegistryEntry | None = None
    candidates: list[dict] = field(default_factory=list)
    reason: str = ""
    new: bool = False  # True when `entry` was just built from Wikidata (not yet in any registry)
    network: bool = False  # True when the resolution consulted Wikidata (may still be cached)


def _candidate_dict(e: PlaceRegistryEntry) -> dict:
    return {"name": e.name, "state": e.state.value, "wikidata_id": e.wikidata_id}


def resolve_place(
    text: str,
    registries: list[Registry],
    wd: WikidataClient,
    as_of: date,
    popular_cutoff: float | None = None,
) -> Resolution:
    text = text.strip()
    if not normalize(text):
        return Resolution("not_found", reason="empty place text")

    # 1. registries: no network
    hits: list[PlaceRegistryEntry] = []
    for reg in registries:
        for e in reg.find(text):
            if e not in hits:
                hits.append(e)
    if len(hits) == 1:
        return Resolution("resolved", hits[0])
    if len(hits) > 1:
        return Resolution(
            "ambiguous", candidates=[_candidate_dict(e) for e in hits],
            reason=f"{text!r} matches several known places",
        )

    # 2. Wikidata
    in_scope: list[tuple] = []
    out_of_scope: list[str] = []
    for cand in wd.search(text, limit=MAX_WIKIDATA_CANDIDATES):
        known = next((r.by_wikidata_id(cand.qid) for r in registries if r.by_wikidata_id(cand.qid)),
                     None)
        if known:
            in_scope.append((cand, None, None, known))
            continue
        ent = wd.entity(cand.qid)
        if ent is None or ent.lat is None or ent.lon is None:
            continue  # not a located thing (films, people, ...)
        ancestors = wd.ancestors(cand.qid)
        if not ancestors:
            continue
        state = state_from_ancestors(ancestors)
        if state is None:
            out_of_scope.append(f"{ent.label} ({cand.description})")
            continue
        in_scope.append((cand, ent, (state, district_of(ancestors)), None))

    if not in_scope:
        if out_of_scope:
            return Resolution(
                "rejected", network=True,
                reason="outside TN/KL/KA/AP/TS/PY: " + "; ".join(out_of_scope[:3]),
            )
        return Resolution("not_found", network=True, reason=f"no located place found for {text!r}")

    key = normalize(text)
    exact = [
        t for t in in_scope
        if normalize(t[0].label) == key
        or any(normalize(a) == key for a in (t[1].aliases if t[1] else []))
    ]
    pick = exact or in_scope
    if len(pick) > 1:
        cands = []
        for cand, ent, geo, known in pick:
            cands.append(_candidate_dict(known) if known else {
                "name": cand.label, "state": geo[0], "wikidata_id": cand.qid,
                "description": cand.description,
            })
        return Resolution(
            "ambiguous", candidates=cands, network=True,
            reason=f"{text!r} could be several places; nothing was picked",
        )

    cand, ent, geo, known = pick[0]
    if known:
        return Resolution("resolved", known, network=True)
    assert ent is not None and geo is not None
    state, district = geo
    views = wd.pageviews_12m(ent.wikipedia_title, as_of) if ent.wikipedia_title else None
    if views is not None and popular_cutoff is not None:
        popularity = "popular" if views >= popular_cutoff else "hidden_gem"
        pop_source = POPULARITY_SIGNAL
    else:
        popularity, pop_source = "hidden_gem", "unavailable"
    name = ent.label or cand.label
    if any(r.get(name) for r in registries):  # name taken by a different place
        name = f"{name} ({district or state})"
    aliases = sorted({a for a in [*ent.aliases, text, ent.wikipedia_title or ""]
                      if a and normalize(a) != normalize(name)})
    entry = PlaceRegistryEntry(
        name=name, aliases=aliases, state=state,
        region_cluster=f"{state}-{district}" if district else f"{state}-unclustered",
        popularity=popularity, lat=ent.lat, lon=ent.lon, hub=None,
        wikidata_id=ent.qid, wikipedia_title=ent.wikipedia_title,
        wikivoyage_title=ent.wikivoyage_title,
        coord_source="wikidata:P625",
        region_cluster_source="wikidata:P131 district" if district else "state only",
        popularity_source=pop_source,
        popularity_signal=float(views) if views is not None else None,
    )
    return Resolution("resolved", entry, new=True, network=True)
