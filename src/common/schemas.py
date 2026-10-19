"""Pydantic models for docs/contracts.md. Change only via a docs/decisions.md entry.

BaseChunk is what Phase 2 produces (Phase 3 fields absent or None, nothing invented).
EnrichedChunk is what Phase 3 produces and every later phase consumes (all fields required).
D10 records the split. Membership of `place` in the registry is checked where the registry is
loaded (Phase 3), not here.
"""
from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SourceType(str, Enum):
    reddit = "reddit"
    blog = "blog"
    youtube = "youtube"
    wikivoyage = "wikivoyage"
    official = "official"
    reference = "reference"  # D22: Wikipedia, Sahapedia (encyclopedic, rarely time-sensitive)


class Aspect(str, Enum):
    timing = "timing"
    closure = "closure"
    price = "price"
    scenery = "scenery"
    food = "food"
    safety = "safety"
    history = "history"
    other = "other"


class State(str, Enum):
    TN = "TN"
    KL = "KL"
    KA = "KA"
    AP = "AP"
    TS = "TS"
    PY = "PY"  # Union Territory of Puducherry incl. Karaikal, Mahe, Yanam (D18)


def make_chunk_id(url: str, offset: int) -> str:
    """Stable, deterministic id: hash of url + offset."""
    return hashlib.sha256(f"{url}\x1f{offset}".encode()).hexdigest()[:16]


def _check_date(v: str | None) -> str | None:
    if v is None:
        return None
    if not _DATE_RE.match(v):
        raise ValueError("date must be YYYY-MM-DD or null")
    from datetime import date

    date.fromisoformat(v)  # rejects 2024-13-45
    return v


class BaseChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_note: str = ""  # contract: "" before Phase 3
    source_type: SourceType
    url: str = Field(min_length=1)
    date: str | None  # required key, null if unknown (neutral freshness, D3)
    state: State
    # Phase 3 fields: absent until enrichment, never filled with placeholder values
    place: str | None = None
    place_type: str | None = None
    season_tags: list[str] | None = None
    aspect: Aspect | None = None
    is_warning: bool | None = Field(default=None, strict=True)  # "yes" must not coerce to True

    _v_date = field_validator("date")(_check_date)


class EnrichedChunk(BaseChunk):
    context_note: str = Field(min_length=1)
    place: str = Field(min_length=1)
    place_type: str = Field(min_length=1)
    season_tags: list[str]
    aspect: Aspect
    is_warning: bool = Field(strict=True)


class PlaceRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    aliases: list[str]
    state: State
    region_cluster: str = Field(min_length=1)
    popularity: Literal["popular", "hidden_gem"]
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    # Optional provenance (D19). Contract fields above are unchanged.
    place_type: str | None = None  # e.g. "region" for district-level items (D21)
    hub: bool | None = None  # flag only (large city / transit hub); never an exclusion
    holdout: bool | None = None  # reserved for the unseen-place test; never tune on it
    wikidata_id: str | None = None
    wikipedia_title: str | None = None
    wikivoyage_title: str | None = None
    see_do_listings: int | None = None  # tourism filter count; None = no Wikivoyage article
    see_do_words: int | None = None
    core_source: Literal["auto", "include", "list"] | None = None  # how it got on the core list
    core_reason: str | None = None  # the reason recorded in config/core_places_list.json
    coord_source: str | None = None
    region_cluster_source: str | None = None
    popularity_source: str | None = None
    popularity_signal: float | None = None
