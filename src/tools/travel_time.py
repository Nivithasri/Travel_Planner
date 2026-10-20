"""travel_time(place_a, place_b) -> hours (Phase 5; contract in docs/contracts.md, D81).

A lookup in index/travel_matrix.json, loaded once. Names resolve by node name, registry alias, IATA
code (hubs) or node id, ignoring case. Unknown names, including hold-outs, raise UnknownPlaceError.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import REPO_ROOT
from src.tools.routing import UnknownPlaceError
from src.tools.travel_model import pair_key

MATRIX_PATH = REPO_ROOT / "index" / "travel_matrix.json"
_cache: dict[Path, Matrix] = {}


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


class Matrix:
    def __init__(self, doc: dict):
        self.meta = doc["meta"]
        self.nodes = {n["id"]: n for n in doc["nodes"]}
        self.hours = doc["hours"]
        self.km = doc["km"]
        self.names: dict[str, str] = {}
        for n in doc["nodes"]:
            for label in [n["id"], n["name"], n.get("iata"), *n.get("aliases", [])]:
                if label:
                    self.names.setdefault(_norm(label), n["id"])

    def resolve(self, name: str) -> str:
        if not isinstance(name, str) or _norm(name) not in self.names:
            raise UnknownPlaceError(name, [n["name"] for n in self.nodes.values()])
        return self.names[_norm(name)]

    def travel_time(self, a: str, b: str) -> float:
        ia, ib = self.resolve(a), self.resolve(b)
        return 0.0 if ia == ib else float(self.hours[pair_key(ia, ib)])

    def road_km(self, a: str, b: str) -> float:
        ia, ib = self.resolve(a), self.resolve(b)
        return 0.0 if ia == ib else float(self.km[pair_key(ia, ib)])


def load_matrix(path: Path | str | None = None) -> Matrix:
    p = Path(path) if path else MATRIX_PATH
    if p not in _cache:
        _cache[p] = Matrix(json.loads(p.read_text(encoding="utf-8")))
    return _cache[p]


def travel_time(place_a: str, place_b: str) -> float:
    """Estimated road travel time in hours between two destinations or hubs (symmetric).

    Provisional: parameters were fitted on unverified reference routes (D77)."""
    return load_matrix().travel_time(place_a, place_b)


def travel_km(place_a: str, place_b: str) -> float:
    """Estimated road distance in km, same lookup as travel_time."""
    return load_matrix().road_km(place_a, place_b)
