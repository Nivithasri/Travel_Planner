"""Shared types for on-demand ingestion (D17, docs/contracts.md "On-demand ingestion")."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from src.common.schemas import PlaceRegistryEntry

# Sources the orchestrator knows. `web` covers the allowlisted domains (config/sources.json).
SOURCES = ("wikivoyage", "wikipedia", "web", "reddit", "youtube")
# Raw record `source` -> chunk `source_type` (D22)
SOURCE_TYPE = {
    "wikivoyage": "wikivoyage", "wikipedia": "reference", "sahapedia": "reference",
    "reddit": "reddit", "youtube": "youtube", "blog": "blog", "official": "official",
}
DEFAULT_TTL_S = {
    "wikivoyage": 30 * 86400, "wikipedia": 30 * 86400, "web": 30 * 86400,
    "reddit": 7 * 86400, "youtube": 14 * 86400,
}
ERROR_RETRY_S = 3600  # a failed source is not retried for an hour


class BlockDetected(RuntimeError):
    """A site refused us (HTTP 403/429, or YouTube blocking transcripts). The whole run stops: no
    retry, no other client, no workaround."""


def check_block(exc: BaseException, where: str) -> None:
    status = getattr(exc, "status", None)
    if status in (403, 429):
        raise BlockDetected(f"{where}: HTTP {status}: {str(exc)[:160]}") from exc


@dataclass
class Budget:
    """A wall-clock deadline (epoch seconds on the injected clock) plus per-source request caps."""

    deadline: float | None = None
    caps: dict[str, int] = field(default_factory=dict)

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline

    def cap_for(self, source: str, default: int) -> int:
        return min(self.caps[source], default) if source in self.caps else default


@dataclass
class SourceResult:
    n_raw: int = 0
    n_requests: int = 0
    error: str | None = None
    skipped: str | None = None  # why nothing was tried (no credentials, not allowlisted, ...)
    complete: bool = True  # False when a deadline or cap stopped it; it is resumed next call
    notes: list[str] = field(default_factory=list)  # non-fatal problems, kept even on success


class Collector(Protocol):
    source: str

    def collect(self, entry: PlaceRegistryEntry, budget: Budget, now: Callable[[], float],
                raw_root: Path) -> SourceResult: ...


@dataclass
class PlaceStatus:
    place: str | None
    state: str | None
    status: str  # new|ingesting|collected|ready|stale|rejected
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
    chunk_counts: dict | None = None  # null until Phase 2
    network_requests: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
