"""Wikidata entity data (CC0) through Special:EntityData, one request per place (D37).

The raw envelope holds the whole entity. `select_statement` and `extract_basic` are pure helpers
for the Phase 5 place_facts build; the fetcher does not build a table.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.collect.types import check_block
from src.common.http import FetchError, HttpClient
from src.common.schemas import PlaceRegistryEntry
from src.facts.envelope import envelope_path, make_envelope, place_dir, write_envelope
from src.facts.hosts import ApiHost, require_host

SOURCE = "wikidata"
URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
PROPS = {"P625": "coordinates", "P2044": "elevation", "P131": "district", "P1435": "heritage",
         "P1082": "population", "P31": "instance_of"}


def entity_url(qid: str) -> str:
    return URL.format(qid=qid)


def _is_entity(qid: str):
    def check(text: str) -> bool:
        try:
            doc = json.loads(text)
        except ValueError:
            return False
        return isinstance(doc, dict) and qid in (doc.get("entities") or {})
    return check


def point_in_time(statement: dict) -> str | None:
    """The P585 qualifier as 'YYYY-MM-DD' (month or day may be 00), or None."""
    for q in (statement.get("qualifiers") or {}).get("P585", []):
        value = (q.get("datavalue") or {}).get("value")
        if isinstance(value, dict) and value.get("time"):
            return value["time"].lstrip("+").split("T")[0]
    return None


def select_statement(statements: list[dict]) -> tuple[dict | None, bool]:
    """Pick the statement to use for a multi-valued property. Returns (statement, ambiguous).

    Deprecated rank is ignored; preferred rank wins over normal; among the candidates the latest
    P585 wins; if none has a P585 the first listed is taken and the choice is marked ambiguous.
    """
    live = [s for s in statements if s.get("rank", "normal") != "deprecated"]
    preferred = [s for s in live if s.get("rank") == "preferred"]
    pool = preferred or live
    if not pool:
        return None, False
    dated = [(point_in_time(s), i, s) for i, s in enumerate(pool) if point_in_time(s)]
    if dated:
        best = max(dated, key=lambda t: (t[0], -t[1]))
        return best[2], False
    return pool[0], len(pool) > 1


def _value(prop: str, statement: dict):
    dv = (statement.get("mainsnak") or {}).get("datavalue") or {}
    v = dv.get("value")
    if prop == "P625" and isinstance(v, dict):
        return {"lat": v.get("latitude"), "lon": v.get("longitude")}
    if prop in ("P2044", "P1082") and isinstance(v, dict):
        try:
            return {"amount": float(v.get("amount")), "unit": v.get("unit")}
        except (TypeError, ValueError):
            return None
    if isinstance(v, dict) and "id" in v:
        return v["id"]
    return v


def source_revision(entity: dict) -> dict:
    return {"lastrevid": entity.get("lastrevid"), "modified": entity.get("modified")}


def extract_basic(entity: dict) -> dict[str, dict]:
    """property name -> {value, as_of, ambiguous}. as_of is the P585 qualifier or None."""
    out: dict[str, dict] = {}
    claims = entity.get("claims") or {}
    for prop, name in PROPS.items():
        chosen, ambiguous = select_statement(claims.get(prop, []))
        if chosen is None:
            continue
        out[name] = {"value": _value(prop, chosen), "as_of": point_in_time(chosen),
                     "ambiguous": ambiguous}
    return out


def fetch_place(http: HttpClient, host: ApiHost, entry: PlaceRegistryEntry, raw_root: Path,
                deadline: float | None = None, manifest: Path | None = None) -> dict:
    """Fetch and store one entity. Returns a small report dict."""
    qid = entry.wikidata_id
    if not qid:
        return {"place": entry.name, "error": "no wikidata_id in the registry"}
    url = entity_url(qid)
    require_host({host.host: host}, url)
    try:
        res = http.fetch(url, check_robots=not host.robots_exempt, validate=_is_entity(qid),
                         deadline=deadline)
    except FetchError as exc:
        check_block(exc, "wikidata")
        return {"place": entry.name, "error": str(exc)[:200]}
    if not res.ok:
        return {"place": entry.name, "error": f"HTTP {res.status}"}
    entity = res.json()["entities"][qid]
    env = make_envelope(SOURCE, entry.name, url, None, res, host.licence, "none required (CC0)",
                        source_revision(entity))
    path = envelope_path(raw_root, SOURCE, place_dir(entry.name), "entity.json")
    wrote = write_envelope(path, env, manifest)
    return {"place": entry.name, "qid": qid, "wrote": wrote, "from_cache": res.from_cache,
            "properties": sorted(extract_basic(entity))}
