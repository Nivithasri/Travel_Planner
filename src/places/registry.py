"""Place registry: name/alias lookup over PlaceRegistryEntry records, stored as JSON.

Two registries exist and never mix on disk (D17): the committed snapshot registry
(data/registry/core_places.json, written only by the bootstrap script) and the live registry
(data/live/registry.json, grows as places are resolved on demand).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.common.cache import atomic_write_text
from src.common.schemas import PlaceRegistryEntry

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


# ---- merged from src/places/aliases.py: Alias hygiene for registry entries.

MIN_LATIN_ALIAS_LEN = 4  # aliases of 3 characters or fewer are dropped


@dataclass(frozen=True)
class DroppedAlias:
    place: str
    alias: str
    reason: str  # "short_latin" | "shared_with:<other place>"


def _is_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(c, "") for c in letters)


def clean_aliases(
    entries: list[PlaceRegistryEntry],
) -> tuple[list[PlaceRegistryEntry], list[DroppedAlias]]:
    dropped: list[DroppedAlias] = []
    kept: dict[str, list[str]] = {}
    for e in entries:
        keep = []
        for a in e.aliases:
            if _is_latin(a) and len(a.strip()) < MIN_LATIN_ALIAS_LEN:
                dropped.append(DroppedAlias(e.name, a, "short_latin"))
            elif a.strip():
                keep.append(a)
        kept[e.name] = keep

    owners: dict[str, set[str]] = {}
    for e in entries:
        for key in {normalize(e.name), *(normalize(a) for a in kept[e.name])} - {""}:
            owners.setdefault(key, set()).add(e.name)
    out: list[PlaceRegistryEntry] = []
    for e in entries:
        keep = []
        for a in kept[e.name]:
            others = sorted(owners.get(normalize(a), set()) - {e.name})
            if others and normalize(a) != normalize(e.name):
                dropped.append(DroppedAlias(e.name, a, f"shared_with:{others[0]}"))
            else:
                keep.append(a)
        out.append(e.model_copy(update={"aliases": keep}))
    return out, dropped
