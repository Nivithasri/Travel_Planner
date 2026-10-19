"""Settings (loaded from the environment / .env) and the pydantic models for docs/contracts.md
(change only via docs/decisions.md)."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ==== config ======================================================================
# Project settings, loaded from the environment / .env. Nothing is hardcoded here.

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED = (
    "OLLAMA_BASE_URL",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "EMBED_MODEL",
    "EVAL_REFERENCE_DATE",
)


DEFAULT_LLM_SEED = 42  # fixed so cached LLM outputs stay reproducible
DEFAULT_RETRY_BUDGET_S = 300.0  # total wall-clock time one LLM call may spend, retries included


class ConfigError(RuntimeError):
    pass


def _optional_number(name: str, default, cast):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    generator_model: str
    judge_model: str
    embed_model: str
    eval_reference_date: date
    cache_dir: Path
    llm_seed: int = DEFAULT_LLM_SEED
    llm_retry_budget_s: float = DEFAULT_RETRY_BUDGET_S


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    """Read settings. Real environment variables win over .env values."""
    load_dotenv(env_file if env_file is not None else REPO_ROOT / ".env", override=False)
    missing = [k for k in _REQUIRED if not os.environ.get(k, "").strip()]
    if missing:
        raise ConfigError(f"Missing required settings: {', '.join(missing)} (see .env.example)")

    generator = os.environ["GENERATOR_MODEL"].strip()
    judge = os.environ["JUDGE_MODEL"].strip()
    if generator == judge:
        raise ConfigError(f"GENERATOR_MODEL and JUDGE_MODEL must differ (both are {generator!r})")

    try:
        ref_date = date.fromisoformat(os.environ["EVAL_REFERENCE_DATE"].strip())
    except ValueError as exc:
        raise ConfigError(f"EVAL_REFERENCE_DATE must be YYYY-MM-DD: {exc}") from exc

    cache_dir = Path(os.environ.get("CACHE_DIR", "").strip() or REPO_ROOT / ".cache")
    budget = _optional_number("LLM_RETRY_BUDGET_S", DEFAULT_RETRY_BUDGET_S, float)
    if budget <= 0:
        raise ConfigError("LLM_RETRY_BUDGET_S must be positive")
    return Settings(
        llm_seed=_optional_number("LLM_SEED", DEFAULT_LLM_SEED, int),
        llm_retry_budget_s=budget,
        ollama_base_url=os.environ["OLLAMA_BASE_URL"].strip().rstrip("/"),
        generator_model=generator,
        judge_model=judge,
        embed_model=os.environ["EMBED_MODEL"].strip(),
        eval_reference_date=ref_date,
        cache_dir=cache_dir,
    )


@dataclass(frozen=True)
class CollectSettings:
    """Settings for data collection. Separate from Settings so collection does not need Ollama."""

    contact_email: str
    user_agent: str
    cache_dir: Path
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    youtube_api_key: str | None = None
    search_api_key: str | None = None


def _opt(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def load_tokenizer_dir(env_file: str | os.PathLike | None = None) -> Path | None:
    """TOKENIZER_DIR (optional): a local folder holding the bge-m3 tokenizer.json (D48). Unset means
    the regex token proxy is used. Never downloads anything."""
    load_dotenv(env_file if env_file is not None else REPO_ROOT / ".env", override=False)
    raw = _opt("TOKENIZER_DIR")
    return Path(raw) if raw else None


def load_collect_settings(env_file: str | os.PathLike | None = None) -> CollectSettings:
    """CONTACT_EMAIL is required: Wikimedia and polite crawling need a way to reach us. The other
    credentials are optional; a collector without its credentials is skipped, never faked."""
    load_dotenv(env_file if env_file is not None else REPO_ROOT / ".env", override=False)
    email = os.environ.get("CONTACT_EMAIL", "").strip()
    if "@" not in email:
        raise ConfigError("CONTACT_EMAIL is required for data collection (see .env.example)")
    cache_dir = Path(os.environ.get("CACHE_DIR", "").strip() or REPO_ROOT / ".cache")
    return CollectSettings(
        contact_email=email,
        user_agent=f"TravelPlannerResearch/0.1 (academic IR project; {email})",
        cache_dir=cache_dir,
        reddit_client_id=_opt("REDDIT_CLIENT_ID"),
        reddit_client_secret=_opt("REDDIT_CLIENT_SECRET"),
        youtube_api_key=_opt("YOUTUBE_API_KEY"),
        search_api_key=_opt("SEARCH_API_KEY"),
    )


# ==== schemas =====================================================================
# Pydantic models for docs/contracts.md. Change only via a docs/decisions.md entry.
#
# BaseChunk is what Phase 2 produces (Phase 3 fields absent or None, nothing invented).
# EnrichedChunk is what Phase 3 produces and every later phase consumes (all fields required).
# D10 records the split. Membership of `place` in the registry is checked where the registry is
# loaded (Phase 3), not here.

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


class DateKind(str, Enum):
    """What the chunk's date evidence is (D48, O9). Freshness counts `published` only."""

    published = "published"  # page's own publish date -> `date`
    revised = "revised"  # wiki revision timestamp from the API -> `revised_date`, `date` null
    sitemap_lastmod = "sitemap_lastmod"  # only a sitemap <lastmod>; not a content date, `date` null
    none = "none"  # no date evidence


class BaseChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_note: str = ""  # contract: "" before Phase 3
    source_type: SourceType
    url: str = Field(min_length=1)
    date: str | None  # required key; publish date only, null otherwise (neutral freshness, D3, D48)
    date_kind: DateKind  # D48
    revised_date: str | None = None  # only when date_kind is revised (D48)
    state: State
    # Phase 3 fields: absent until enrichment, never filled with placeholder values
    place: str | None = None
    place_type: str | None = None
    season_tags: list[str] | None = None
    aspect: Aspect | None = None
    is_warning: bool | None = Field(default=None, strict=True)  # "yes" must not coerce to True

    _v_date = field_validator("date", "revised_date")(_check_date)

    @model_validator(mode="after")
    def _date_invariants(self):
        # D48: published <=> date set; revised_date only (and always) for revised
        if (self.date_kind is DateKind.published) != (self.date is not None):
            raise ValueError("date must be set exactly when date_kind is published")
        if (self.date_kind is DateKind.revised) != (self.revised_date is not None):
            raise ValueError("revised_date must be set exactly when date_kind is revised")
        return self


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
