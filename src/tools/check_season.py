"""check_season(place, month) -> dict (Phase 5; contract in docs/contracts.md, D80, D81).

A lookup in index/season_data.json, loaded once. Only the destinations with climate data are known
(hubs and hold-outs raise UnknownPlaceError). The verdict comes from ERA5 monthly normals; tag and
warning evidence is attached with counts split by provenance, never used to decide the verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.tools.routing import UnknownPlaceError

SEASON_PATH = REPO_ROOT / "index" / "season_data.json"
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_cache: dict[Path, SeasonTable] = {}


def parse_month(month) -> int:
    """1..12, or an English month name or three-letter abbreviation. Anything else is ValueError."""
    if isinstance(month, bool):
        raise ValueError(f"invalid month {month!r}")
    if isinstance(month, int):
        if 1 <= month <= 12:
            return month
        raise ValueError(f"month must be 1 to 12, got {month}")
    if isinstance(month, str):
        key = month.strip().casefold()
        for i, name in enumerate(_MONTHS, start=1):
            if key in (name, name[:3]):
                return i
    raise ValueError(f"invalid month {month!r}")


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


class SeasonTable:
    def __init__(self, doc: dict):
        self.meta = doc["meta"]
        self.places = doc["places"]
        self.names: dict[str, str] = {}
        for name in self.places:
            for label in [name, *(doc["meta"].get("aliases", {}).get(name, []))]:
                self.names.setdefault(_norm(label), name)

    def resolve(self, place: str) -> str:
        if not isinstance(place, str) or _norm(place) not in self.names:
            raise UnknownPlaceError(place, list(self.places))
        return self.names[_norm(place)]

    def check(self, place: str, month) -> dict:
        name = self.resolve(place)
        m = parse_month(month)
        entry = self.places[name]
        row = entry["months"].get(str(m))
        if row is None:
            raise ValueError(f"no climate data for {name} in month {m}")
        ev = row["evidence"]
        caveats = ["verdict from ERA5 monthly normals 2015-2024 (0.25 degree reanalysis), "
                   "thresholds are provisional starting values (D80)"]
        if entry["confidence"] == "low":
            caveats.append("low confidence: district or hill-station grid cell (D21, D42)")
        if ev["model_written"]["chunks"] == 0:
            caveats.append("no model-written chunks for this place: tags are rule-based (D69)")
        return {"place": name, "month": m, "verdict": row["verdict"], "reasons": row["reasons"],
                "access_risk": row["access_risk"], "climate": row["climate"], "evidence": ev,
                "warning_chunk_ids": row["warning_chunk_ids"],
                "warning_sources": row["warning_sources"], "confidence": entry["confidence"],
                "nearest_airport": entry.get("nearest_airport"), "caveats": caveats}


def load_season(path: Path | str | None = None) -> SeasonTable:
    p = Path(path) if path else SEASON_PATH
    if p not in _cache:
        _cache[p] = SeasonTable(json.loads(p.read_text(encoding="utf-8")))
    return _cache[p]


def check_season(place: str, month) -> dict:
    """Suitability of `place` in `month` (1..12 or a month name): verdict good | fair | poor,
    reasons, access_risk for hill places in heavy rain, climate normals, evidence counts by
    provenance, warning chunk ids and a confidence label."""
    return load_season().check(place, month)
