"""Wikidata / Wikimedia lookups used to resolve places and bootstrap the core list.

All requests go through HttpClient (cached, polite). API endpoints are not subject to robots.txt, so
check_robots is off for them; the User-Agent (built from CollectSettings only) carries a contact
address as Wikimedia requires.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.parse import quote

from src.common.http import HttpClient

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

_POINT = re.compile(r"Point\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")


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


def parse_point(wkt: str) -> tuple[float, float] | None:
    """'Point(lon lat)' -> (lat, lon)."""
    m = _POINT.match(wkt.strip())
    if not m:
        return None
    lon, lat = float(m.group(1)), float(m.group(2))
    return lat, lon


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


def normalize_title(title: str) -> str:
    t = title.replace("_", " ").strip()
    return t[:1].upper() + t[1:]


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

    def entities_by_qids(self, qids: list[str]) -> dict[str, Entity]:
        out: dict[str, Entity] = {}
        for i in range(0, len(qids), 50):
            chunk = qids[i:i + 50]
            body = self._api({
                "action": "wbgetentities", "ids": "|".join(chunk), "languages": "en|mul",
                "props": "labels|aliases|claims|sitelinks", "sitefilter": "enwikivoyage|enwiki",
            })
            for qid, raw in (body.get("entities") or {}).items():
                ent = _parse_entity(qid, raw)
                if ent:
                    out[qid] = ent
        return out

    def items_in(self, root_qid: str) -> list[dict]:
        """Items located (P131+) in `root_qid` with coordinates and an English Wikipedia article.
        Rows: qid, label, wp_title, wv_title (or None), lat, lon, types, population."""
        rows = self.sparql(
            "SELECT ?item ?itemLabel ?wp ?wv ?coord ?pop "
            '(GROUP_CONCAT(DISTINCT ?tl; separator="|") AS ?types) WHERE { '
            f"?item wdt:P131+ wd:{root_qid}; wdt:P625 ?coord. "
            "?a schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>; schema:name ?wp. "
            "OPTIONAL { ?v schema:about ?item; schema:isPartOf <https://en.wikivoyage.org/>; "
            "schema:name ?wv } "
            "OPTIONAL { ?item wdt:P1082 ?pop } "
            'OPTIONAL { ?item wdt:P31 ?t. ?t rdfs:label ?tl. FILTER(lang(?tl)="en") } '
            'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } '
            "} GROUP BY ?item ?itemLabel ?wp ?wv ?coord ?pop"
        )
        out = []
        for r in rows:
            point = parse_point(r["coord"]["value"])
            if point is None:
                continue
            pop = r.get("pop", {}).get("value")
            out.append({
                "qid": _qid(r["item"]["value"]), "label": r["itemLabel"]["value"],
                "wp_title": r["wp"]["value"], "wv_title": r.get("wv", {}).get("value"),
                "lat": point[0], "lon": point[1], "population": float(pop) if pop else None,
                "types": sorted({t for t in r.get("types", {}).get("value", "").split("|") if t}),
            })
        return sorted(out, key=lambda d: d["qid"])

    def entities_by_wikivoyage_titles(self, titles: list[str]) -> dict[str, Entity]:
        """Resolve English Wikivoyage article titles to Wikidata entities through the sitelink.
        Returns {requested title: Entity}; titles with no linked item are absent."""
        out: dict[str, Entity] = {}
        for i in range(0, len(titles), 50):
            chunk = titles[i:i + 50]
            body = self._api({
                "action": "wbgetentities", "sites": "enwikivoyage",
                "titles": "|".join(normalize_title(t) for t in chunk), "languages": "en|mul",
                "props": "labels|aliases|claims|sitelinks", "sitefilter": "enwikivoyage|enwiki",
            })
            by_title = {}
            for qid, raw in (body.get("entities") or {}).items():
                ent = _parse_entity(qid, raw)
                if ent and ent.wikivoyage_title:
                    by_title[ent.wikivoyage_title] = ent
            for t in chunk:
                ent = by_title.get(normalize_title(t))
                if ent:
                    out[t] = ent
        return out

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

    def types_many(self, qids: list[str]) -> dict[str, list[str]]:
        """English labels of P31 (instance of) per item."""
        out: dict[str, list[str]] = {q: [] for q in qids}
        for i in range(0, len(qids), SPARQL_BATCH):
            values = " ".join(f"wd:{q}" for q in qids[i:i + SPARQL_BATCH])
            rows = self.sparql(
                f"SELECT ?item ?tl WHERE {{ VALUES ?item {{ {values} }} ?item wdt:P31 ?t. "
                '?t rdfs:label ?tl. FILTER(lang(?tl)="en") }'
            )
            for r in rows:
                out[_qid(r["item"]["value"])].append(r["tl"]["value"])
        return {q: sorted(set(v)) for q, v in out.items()}

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
