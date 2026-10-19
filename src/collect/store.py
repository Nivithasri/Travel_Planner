"""Live store (D17): data/live/places/<slug>.json per place and data/live/registry.json.

Separate from the frozen snapshot (data/processed, data/registry/core_places.json); nothing here
ever writes there. All writes are atomic.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.cache import atomic_write_text
from src.places.registry import Registry, slugify


class PlaceStore:
    def __init__(self, live_root: str | Path):
        self.root = Path(live_root)

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    @property
    def raw_root(self) -> Path:
        return self.root / "raw"

    def _path(self, place: str) -> Path:
        return self.root / "places" / f"{slugify(place)}.json"

    def load(self, place: str) -> dict | None:
        try:
            doc = json.loads(self._path(place).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def save(self, place: str, doc: dict) -> None:
        atomic_write_text(self._path(place),
                          json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

    def load_registry(self) -> Registry:
        return Registry.load(self.registry_path)

    def save_registry(self, reg: Registry) -> None:
        reg.save(self.registry_path)
