"""Hold-out filter for facts (D17, D37, docs/facts-layer.md 3.2).

A feature is dropped when any of its name tags matches a hold-out name or alias (word match),
or when "Periyar" stands next to Thekkady, Kumily, Periyar Lake, Tiger Reserve, National Park or
Wildlife Sanctuary (O8), or when it lies within `radius_km` of a hold-out centre.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable

DROP_NAMES = (
    "Thanjavur", "Tanjore", "Thanjai", "Chikmagalur", "Chikkamagaluru", "Araku Valley", "Araku",
    "Thekkady", "Kumily", "Brihadisvara", "Brihadeeswarar",
)
NAME_TAGS = ("name", "name:en", "alt_name", "old_name", "official_name", "short_name")
_PERIYAR = re.compile(
    r"Periyar\s+(?:Lake|Tiger\s+Reserve|National\s+Park|Wildlife\s+Sanctuary)"
    r"|(?:National\s+Park|Wildlife\s+Sanctuary)\W+Periyar", re.IGNORECASE)


def _word_re(terms: Iterable[str]) -> re.Pattern:
    alts = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in alts) + r")(?!\w)",
                      re.IGNORECASE)


class HoldoutFilter:
    def __init__(self, centres: Iterable[tuple[float, float]] = (),
                 extra_terms: Iterable[str] = (), radius_km: float = 10.0):
        self.centres = list(centres)
        self.radius_km = radius_km
        self._names = _word_re(list(DROP_NAMES) + list(extra_terms))

    @classmethod
    def from_registry(cls, registry, radius_km: float = 10.0) -> HoldoutFilter:
        held = [e for e in registry.entries if e.holdout]
        terms = [t for e in held for t in [e.name, *e.aliases] if len(t) > 3]
        return cls([(e.lat, e.lon) for e in held], terms, radius_km)

    def name_matches(self, text: str) -> bool:
        return bool(self._names.search(text) or _PERIYAR.search(text))

    def near_holdout(self, lat: float, lon: float) -> bool:
        return any(haversine_km(lat, lon, c[0], c[1]) < self.radius_km for c in self.centres)

    def reason(self, element: dict) -> str | None:
        tags = element.get("tags") or {}
        for key in NAME_TAGS:
            value = tags.get(key)
            if value and self.name_matches(value):
                return f"name ({key})"
        point = element.get("center") or element
        lat, lon = point.get("lat"), point.get("lon")
        if lat is not None and lon is not None and self.near_holdout(lat, lon):
            return f"within {self.radius_km:g} km of a hold-out centre"
        return None

    def filter(self, elements: Iterable[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
        kept, dropped = [], []
        for el in elements:
            why = self.reason(el)
            (dropped.append((el, why)) if why else kept.append(el))
        return kept, dropped


# ---- merged from src/facts/geo.py: Great-circle distance.

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))
