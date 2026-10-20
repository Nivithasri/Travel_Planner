"""Routing reference data (Phase 5; D76 to D81): airports, nodes, travel-time model, matrix.

Everything here is offline and deterministic. Inputs are the registry, the saved Overpass airports,
the saved Wikidata entities and config/routing_params.json; nothing calls the network.
"""
from __future__ import annotations

import difflib
import json
from collections.abc import Iterable
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.facts.facts import haversine_km

AIRPORTS_ALLOWLIST_PATH = REPO_ROOT / "config" / "airports_allowlist.json"
AIRPORTS_RAW_PATH = REPO_ROOT / "data" / "raw" / "overpass" / "_south" / "airports.json"
_TYPE_PRIORITY = {"relation": 0, "way": 1, "node": 2}  # lower wins (O13)


class UnknownPlaceError(ValueError):
    """A place name that is not in the table the function reads (D81). Carries close names."""

    def __init__(self, name: str, known: Iterable[str] = ()):
        self.name = name
        self.suggestions = difflib.get_close_matches(str(name), list(known), n=3, cutoff=0.6)
        hint = f"; did you mean {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(f"unknown place {name!r}{hint}")


# ==== airports (O13) ==============================================================

def load_airports_allowlist(path: Path = AIRPORTS_ALLOWLIST_PATH) -> list[str]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(doc["in_region"])


def element_point(element: dict) -> tuple[float, float] | None:
    """(lat, lon) of an Overpass element: its own point, else its `center` (ways, relations)."""
    point = element if "lat" in element and "lon" in element else element.get("center") or {}
    if "lat" in point and "lon" in point:
        return float(point["lat"]), float(point["lon"])
    return None


def is_military(tags: dict) -> bool:
    """D79: military=* or an aerodrome:type containing "military" marks a military airport."""
    return bool(tags.get("military")) or "military" in str(tags.get("aerodrome:type", "")).lower()


def dedupe_airports(elements: Iterable[dict], allowed: Iterable[str]) -> dict[str, dict]:
    """IATA -> airport record, only for allowed codes. Same IATA: relation > way > node, then the
    lowest OSM id (O13). Elements without a usable point are skipped."""
    allow = set(allowed)
    best: dict[str, tuple[tuple[int, int], dict]] = {}
    for el in elements:
        tags = el.get("tags") or {}
        iata = tags.get("iata")
        point = element_point(el)
        if iata not in allow or point is None:
            continue
        key = (_TYPE_PRIORITY.get(el.get("type"), 9), int(el["id"]))
        if iata not in best or key < best[iata][0]:
            name = tags.get("name") or tags.get("alt_name") or iata
            best[iata] = (key, {"iata": iata, "name": name, "lat": point[0], "lon": point[1],
                                "osm_type": el.get("type"), "osm_id": int(el["id"]),
                                "military": is_military(tags)})
    return {k: v[1] for k, v in sorted(best.items())}


def load_airports(raw_path: Path = AIRPORTS_RAW_PATH,
                  allowlist_path: Path = AIRPORTS_ALLOWLIST_PATH) -> dict[str, dict]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    return dedupe_airports(raw["response"]["elements"], load_airports_allowlist(allowlist_path))


def nearest_airport(lat: float, lon: float, airports: dict[str, dict]) -> dict:
    """The closest allowlist airport (never an out-of-region one: `airports` is already filtered).
    Ties go to the lower IATA so the result is deterministic."""
    civil = {k: v for k, v in airports.items() if not v.get("military")}
    if not civil:
        raise ValueError("no airports")
    iata = min(civil, key=lambda k: (haversine_km(lat, lon, civil[k]["lat"], civil[k]["lon"]), k))
    a = civil[iata]
    return {"iata": iata, "name": a["name"],
            "km": round(haversine_km(lat, lon, a["lat"], a["lon"]), 1)}
