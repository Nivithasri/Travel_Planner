"""OSM features through the public Overpass instance (ODbL, D37). Not run until approved.

Airports only: ONE query over the south bounding box. The per-place station queries were removed
(D40: not in the plan, ODbL share-alike cost, O12). Hold-outs are dropped when derived lists are
made (HoldoutFilter), not from the raw file.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.collect.types import BlockDetected, check_block
from src.common.http import FetchError, HttpClient
from src.facts.envelope import envelope_path, make_envelope, write_envelope
from src.facts.holdout import HoldoutFilter, haversine_km
from src.facts.hosts import ApiHost, require_host

SOURCE = "overpass"
URL = "https://overpass-api.de/api/interpreter"
SOUTH_BBOX = (8.0, 74.0, 20.0, 85.0)  # south, west, north, east
ATTRIBUTION = "Contains information from OpenStreetMap, made available under the ODbL"
BLOCK_STATUS = (403, 406, 429)


def airports_query() -> str:
    s, w, n, e = SOUTH_BBOX
    return (f'[out:json][timeout:60];nwr["aeroway"="aerodrome"]["iata"]({s},{w},{n},{e});'
            "out center tags;")


def validate_response(text: str) -> bool:
    try:
        doc = json.loads(text)
    except ValueError:
        return False
    if not isinstance(doc, dict) or not isinstance(doc.get("elements"), list):
        return False
    remark = str(doc.get("remark", "")).lower()
    return not ("runtime error" in remark or "timed out" in remark)


def point(el: dict) -> tuple[float, float] | None:
    p = el.get("center") or el
    if p.get("lat") is None or p.get("lon") is None:
        return None
    return p["lat"], p["lon"]


def nearest(elements: list[dict], lat: float, lon: float,
            max_km: float | None = None) -> tuple[dict, float] | None:
    best: tuple[dict, float] | None = None
    for el in elements:
        pt = point(el)
        if pt is None:
            continue
        km = haversine_km(lat, lon, pt[0], pt[1])
        if max_km is not None and km > max_km:
            continue
        if best is None or km < best[1]:
            best = (el, km)
    return best


def nearest_airport(elements: list[dict], lat: float, lon: float,
                    flt: HoldoutFilter) -> tuple[dict, float] | None:
    kept, _ = flt.filter(elements)
    return nearest(kept, lat, lon)


def _fetch(http: HttpClient, host: ApiHost, query: str, deadline: float | None):
    url = URL
    require_host({host.host: host}, url)
    try:
        res = http.fetch(url, {"data": query}, check_robots=not host.robots_exempt,
                         validate=validate_response, deadline=deadline)
    except FetchError as exc:
        check_block(exc, "overpass")
        if exc.status in BLOCK_STATUS:  # 406 is treated like 429 (usage policy)
            raise BlockDetected(f"overpass: HTTP {exc.status}") from exc
        raise
    return url, res


def fetch_airports(http: HttpClient, host: ApiHost, raw_root: Path,
                   deadline: float | None = None, manifest: Path | None = None) -> dict:
    query = airports_query()
    try:
        url, res = _fetch(http, host, query, deadline)
    except FetchError as exc:  # a block already raised BlockDetected; this is a plain failure
        return {"place": "_south", "error": str(exc)[:200]}
    env = make_envelope(SOURCE, "_south", url, {"data": query}, res, host.licence, ATTRIBUTION)
    path = envelope_path(raw_root, SOURCE, "_south", "airports.json")
    wrote = write_envelope(path, env, manifest)
    return {"place": "_south", "wrote": wrote, "elements": len(res.json()["elements"])}
