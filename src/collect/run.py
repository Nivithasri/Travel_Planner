"""Collection runs: shared types, the raw record store, the live store, coverage (D23), ensure_place
(D17) and the orchestrator."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.cache import DiskCache, append_jsonl, atomic_write_text, read_jsonl
from src.common.http import DEFAULT_REQUEST_LOG, Allowlist, HttpClient, RequestLog
from src.common.schemas import REPO_ROOT, CollectSettings, PlaceRegistryEntry
from src.places.places import Registry, Resolution, slugify

# ==== types ======================================================================
# Shared types for on-demand ingestion (D17, docs/contracts.md "On-demand ingestion").

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


# ==== raw ========================================================================
# Raw record store: data/raw/<source>/<place_slug>/records.jsonl (docs/contracts.md "Raw record").
#
# One JSON object per line. A record is identified by (url, content_sha256): writing the same page
# with the same content twice is a no-op, so a refresh that finds nothing new adds nothing. The
# optional manifest (url, source, place, retrieved_at, sha256, bytes; no content) is the committed
# trace of what was collected.

RAW_SOURCES = ("wikivoyage", "wikipedia", "sahapedia", "reddit", "youtube", "blog", "official")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RawRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    place: str
    url: str = Field(min_length=1)
    retrieved_at: str
    publish_date: str | None = None
    lastmod: str | None = None  # never used as a publish date
    content_sha256: str
    license: str = ""
    title: str = ""
    body: str = Field(min_length=1)
    extra: dict = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if v not in RAW_SOURCES:
            raise ValueError(f"unknown raw source {v!r}")
        return v

    @field_validator("publish_date", "lastmod")
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE.match(v):
            raise ValueError("date must be YYYY-MM-DD or null")
        return v


def make_record(source: str, place: str, url: str, body: str, retrieved_at: float,
                **kw) -> RawRecord:
    return RawRecord(source=source, place=place, url=url, body=body,
                     retrieved_at=iso_utc(retrieved_at), content_sha256=sha256_text(body), **kw)


class RawStore:
    def __init__(self, raw_root: str | Path, manifest: str | Path | None = None):
        self.root = Path(raw_root)
        self.manifest = Path(manifest) if manifest else None

    def path_for(self, source: str, place: str) -> Path:
        return self.root / source / slugify(place) / "records.jsonl"

    def read(self, source: str, place: str) -> list[dict]:
        p = self.path_for(source, place)
        return list(read_jsonl(p)) if p.exists() else []

    def write(self, rec: RawRecord) -> bool:
        """Append the record; False when the same url with the same content is already stored."""
        path = self.path_for(rec.source, rec.place)
        if any(r["url"] == rec.url and r["content_sha256"] == rec.content_sha256
               for r in self.read(rec.source, rec.place)):
            return False
        append_jsonl(path, [rec.model_dump(mode="json")])
        if self.manifest is not None:
            append_jsonl(self.manifest, [{
                "url": rec.url, "source": rec.source, "place": rec.place,
                "retrieved_at": rec.retrieved_at, "sha256": rec.content_sha256,
                "bytes": len(rec.body.encode("utf-8"))}])
        return True


# ==== store ======================================================================
# Live store (D17): data/live/places/<slug>.json per place and data/live/registry.json.
#
# Separate from the frozen snapshot (data/processed, data/registry/core_places.json); nothing here
# ever writes there. All writes are atomic.

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


# ==== coverage ===================================================================
# Coverage of the core snapshot places by source type (D22, D23).
#
# A place passes when it has raw data from at least two distinct chunk source types AND at least one
# of them is not wikivoyage or reference. Hold-out places are listed but never counted. A collector
# that was skipped (no credentials, not allowed) is reported with its reason; it never counts.

REFERENCE_FAMILY = {"wikivoyage", "reference"}


@dataclass
class PlaceCoverage:
    place: str
    state: str
    holdout: bool
    records_by_raw_source: dict[str, int] = field(default_factory=dict)
    types: list[str] = field(default_factory=list)
    passes: bool = False
    reason: str = ""


def passes_rule(types: set[str]) -> bool:
    return len(types) >= 2 and bool(types - REFERENCE_FAMILY)


def place_coverage(entry: PlaceRegistryEntry, raw_root: Path) -> PlaceCoverage:
    store = RawStore(raw_root)
    by_source: dict[str, int] = {}
    for raw_source in SOURCE_TYPE:
        n = len(store.read(raw_source, entry.name))
        if n:
            by_source[raw_source] = n
    types = {SOURCE_TYPE[s] for s in by_source}
    cov = PlaceCoverage(entry.name, entry.state.value, bool(entry.holdout), by_source,
                        sorted(types))
    if cov.holdout:
        cov.reason = "hold-out: not collected by design"
        return cov
    cov.passes = passes_rule(types)
    if not cov.passes:
        if not types:
            cov.reason = "nothing collected yet"
        elif len(types) < 2:
            cov.reason = f"only {len(types)} source type(s)"
        else:
            cov.reason = "only reference-wiki sources (wikivoyage/reference)"
    return cov


def skipped_sources(status_root: Path, place: str) -> dict[str, str]:
    """source -> reason, from the per-place status file written by ensure_place."""
    try:
        doc = json.loads((status_root / "places" / f"{slugify(place)}.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for src, rec in doc.get("sources", {}).items():
        if rec.get("skipped"):
            out[src] = f"skipped: {rec['skipped']}"
        elif rec.get("error"):
            out[src] = f"error: {rec['error']}"
        elif rec.get("notes"):
            out[src] = "notes: " + " | ".join(rec["notes"])
    return out


def render(entries: list[PlaceRegistryEntry], raw_root: Path, status_root: Path,
           skipped_everywhere: dict[str, str] | None = None) -> tuple[str, dict]:
    rows = [place_coverage(e, raw_root) for e in entries]
    counted = [r for r in rows if not r.holdout]
    n_pass = sum(r.passes for r in counted)
    totals: dict[str, int] = {}
    for r in rows:
        for s, n in r.records_by_raw_source.items():
            totals[s] = totals.get(s, 0) + n
    lines = ["# Coverage report", "",
             f"Rule (D23): >= 2 source types and >= 1 that is not wikivoyage/reference. "
             f"**{n_pass} of {len(counted)}** counted places pass "
             f"({len(rows) - len(counted)} hold-out(s) excluded).", "",
             "## Records per raw source", "", "| source | records |", "|---|---:|"]
    lines += [f"| {s} | {totals.get(s, 0)} |" for s in SOURCE_TYPE]
    notes: dict[str, str] = dict(skipped_everywhere or {})
    for r in rows:
        for s, why in skipped_sources(status_root, r.place).items():
            notes.setdefault(s, why)
    if notes:
        lines += ["", "## Collectors skipped or failed", ""]
        lines += [f"- **{s}**: {why}" for s, why in sorted(notes.items())]
    lines += ["", "## Per place", "", "| place | state | types | pass | note |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.place} | {r.state} | {', '.join(r.types) or '-'} | "
                     f"{'hold-out' if r.holdout else ('yes' if r.passes else 'NO')} | {r.reason} |")
    data = {"passing": n_pass, "counted": len(counted), "records_by_source": totals,
            "notes": notes, "places": [r.__dict__ for r in rows]}
    return "\n".join(lines) + "\n", data


def raw_stats(raw_root: Path) -> dict[str, dict]:
    """Per raw source: records, bytes on disk (records.jsonl), places with data, body characters."""
    out: dict[str, dict] = {}
    for src in SOURCE_TYPE:
        files = list((raw_root / src).glob("*/records.jsonl")) if (raw_root / src).exists() else []
        n = chars = 0
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    n += 1
                    chars += len(json.loads(line).get("body", ""))
        out[src] = {"records": n, "bytes": sum(f.stat().st_size for f in files),
                    "places": len(files), "body_chars": chars}
    return out


def transcript_outcomes(cache_dir: Path, status_root: Path) -> dict[str, int]:
    """ok / none / disabled from the transcript cache; blocked from the per-place status files
    (a block is never cached, it aborts the run)."""
    out = {"ok": 0, "none": 0, "disabled": 0, "blocked": 0}
    for f in (cache_dir / "youtube_transcript").glob("*/*.json") if (
            cache_dir / "youtube_transcript").exists() else []:
        try:
            val = json.loads(f.read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError):
            continue
        if val.get("status") == "ok":
            out["ok"] += 1
        elif val.get("reason") == "TranscriptsDisabled":
            out["disabled"] += 1
        else:
            out["none"] += 1
    for f in (status_root / "places").glob("*.json") if (status_root / "places").exists() else []:
        err = (json.loads(f.read_text(encoding="utf-8")).get("sources", {})
               .get("youtube", {}).get("error") or "")
        out["blocked"] += "blocked" in err
    return out


# ==== ensure =====================================================================
# ensure_place(place_text, budget) -> PlaceStatus  (docs/contracts.md, D17).
#
# Resolve free text, reject places outside scope, then run every collector whose source record is
# missing, expired or incomplete. A fresh place makes no network call. Time comes from an injected
# clock, so freshness is testable without waiting. Phase 1 never sets status "ready" (chunks are
# not indexed yet).
#
# Raw data for live places goes under the live store (data/live/raw), never into data/raw, which
# belongs to the frozen snapshot.

Resolver = Callable[[str, list[Registry]], Resolution]


class PlaceIngestor:
    def __init__(
        self,
        store: PlaceStore,
        core: Registry,
        collectors: list[Collector],
        resolver: Resolver,
        *,
        clock: Callable[[], float],
        request_counter: Callable[[], int] = lambda: 0,
        ttl_s: Mapping[str, float] | None = None,
        raw_root: Path | None = None,
        request_log: RequestLog | None = None,
    ):
        self.store, self.core, self.collectors = store, core, collectors
        self.resolver, self.clock, self.request_counter = resolver, clock, request_counter
        self.ttl_s = {**DEFAULT_TTL_S, **(ttl_s or {})}
        self.raw_root = raw_root if raw_root is not None else store.raw_root
        self.request_log = request_log

    def ensure_place(self, place_text: str, budget: Budget | None = None) -> PlaceStatus:
        budget = budget or Budget()
        before = self.request_counter()
        live = self.store.load_registry()
        res = self.resolver(place_text, [self.core, live])
        if res.status != "resolved" or res.entry is None:
            return PlaceStatus(
                place=None, state=None, status="rejected", reason=f"{res.status}: {res.reason}",
                candidates=res.candidates, network_requests=self.request_counter() - before)
        entry = res.entry
        if res.new:
            live.add(entry)
            self.store.save_registry(live)

        doc = self.store.load(entry.name) or {"place": entry.name, "sources": {}}
        sources: dict[str, dict] = doc["sources"]
        stale_unrefreshed: list[str] = []
        pending: list[str] = []
        for col in self.collectors:
            rec = sources.get(col.source)
            now = self.clock()
            usable = rec is not None and not rec.get("skipped")
            if usable and rec.get("complete", True) and now < rec["expires_at"]:
                continue  # fresh: no network
            if budget.expired(now):
                (stale_unrefreshed if usable and rec.get("complete", True) else pending).append(
                    col.source)
                continue
            if self.request_log is not None:
                self.request_log.source = col.source
            result = col.collect(entry, budget, self.clock, self.raw_root)
            sources[col.source] = self._record(col.source, result, self.clock())
            if not result.complete:
                pending.append(col.source)
            self.store.save(entry.name, {**doc, "state": entry.state.value, "sources": sources})
        self.store.save(entry.name, {**doc, "state": entry.state.value, "sources": sources})

        if pending:
            status = "ingesting" if sources else "new"
        elif stale_unrefreshed:
            status = "stale"
        else:
            status = "collected"
        waiting = pending + stale_unrefreshed
        return PlaceStatus(
            place=entry.name, state=entry.state.value, status=status,
            reason=("not finished: " + ", ".join(waiting)) if waiting else "", sources=sources,
            network_requests=self.request_counter() - before)

    def _record(self, source: str, r: SourceResult, now: float) -> dict:
        base = {"ingested_at": now, "n_raw": r.n_raw, "n_requests": r.n_requests, "error": r.error,
                "skipped": r.skipped, "complete": r.complete, "notes": r.notes[:10]}
        if r.skipped:  # nothing was tried; looked at again on the next call (cheap, no network)
            return {**base, "ingested_at": None, "expires_at": None, "complete": True}
        if r.error:
            return {**base, "expires_at": now + ERROR_RETRY_S, "complete": True}
        return {**base, "expires_at": now + self.ttl_s[source]}


# ==== orchestrator ===============================================================
# Run the collectors over the core snapshot places (data/raw) or over any place on demand.
#
# The snapshot run never touches hold-out places (D17) and never resolves anything over the network:
# places come from data/registry/core_places.json. Per-place progress is stored under
# data/raw/_status (git-ignored) so a stopped run resumes where it stopped and a finished one makes
# no request.

# YouTube is refused (D27, D30): transcripts are dropped and the Data API stays off for stored data.
SKIPPED_D27 = "skipped (D27): YouTube transcripts are dropped; robots.txt disallows the paths"

RAW_ROOT = REPO_ROOT / "data" / "raw"
STATUS_ROOT = RAW_ROOT / "_status"
MANIFEST = REPO_ROOT / "data" / "manifests" / "raw_manifest.jsonl"
CORE_REGISTRY = REPO_ROOT / "data" / "registry" / "core_places.json"
HOST_DELAYS = {"en.wikivoyage.org": 0.5, "en.wikipedia.org": 0.5}


class HoldoutError(RuntimeError):
    pass


REQUEST_LOG = DEFAULT_REQUEST_LOG
_DEFAULT = object()
SOURCES_CONFIG = REPO_ROOT / "config" / "sources.json"
WIKI_HOSTS = ("en.wikivoyage.org", "en.wikipedia.org")


def load_run_config(path: Path = SOURCES_CONFIG) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    reason = doc.get("unreachable_reason", "unreachable")
    allow = [d["domain"].lower() for d in doc.get("web_allowlist", [])]
    return {
        "max_requests": int(doc.get("run_limits", {}).get("max_requests_per_run", 250)),
        "unreachable": {h.lower(): reason for h in doc.get("unreachable_not_retested", [])},
        "redirect_hosts": set(allow) | {"www." + h for h in allow} | set(WIKI_HOSTS),
        "youtube_transcripts": bool(doc.get("youtube_transcripts", False)),  # D27
    }


def make_http(cfg: CollectSettings, *, request_log: RequestLog | None | object = _DEFAULT,
              max_requests: int | None = None, run_config: dict | None = None,
              transport=None, host_delays: dict[str, float] | None = None) -> HttpClient:
    """The one way to build the project's HttpClient. It LOGS EVERY REQUEST to REQUEST_LOG by
    default; pass request_log=None only in tests. `transport` is for tests (a fake)."""
    rc = run_config or load_run_config()
    cap = rc["max_requests"] if max_requests is None else min(max_requests, rc["max_requests"])
    log = RequestLog(REQUEST_LOG) if request_log is _DEFAULT else request_log
    extra = {} if transport is None else {"transport": transport}
    allow = Allowlist.from_config(SOURCES_CONFIG)  # D43: any other host or path is refused
    return HttpClient(
        DiskCache(cfg.cache_dir), cfg.user_agent,
        allowlist=allow, host_caps=allow.caps,
        host_delays={**HOST_DELAYS, **allow.delays, **(host_delays or {})},
        request_log=log, max_requests=cap, redirect_hosts=rc["redirect_hosts"],
        unreachable_hosts=rc["unreachable"], **extra)


def build_collectors(cfg: CollectSettings, http: HttpClient, manifest: Path | None = MANIFEST,
                     only: list[str] | None = None,
                     run_config: dict | None = None) -> list[Collector]:
    # imported here, not at the top: src/collect/sources.py imports this module (no import cycle)
    from src.collect.sources import RedditCollector, WebCollector, WikiCollector

    transcripts = (run_config or load_run_config())["youtube_transcripts"]
    all_: dict[str, Collector] = {
        "wikivoyage": WikiCollector("wikivoyage", http, manifest),
        "wikipedia": WikiCollector("wikipedia", http, manifest),
        "web": WebCollector(http, manifest=manifest),
        "reddit": RedditCollector(http, cfg.reddit_client_id, cfg.reddit_client_secret, manifest),
        "youtube": SimpleNamespace(  # youtube refused (D27): zero requests, reported as skipped
            source="youtube", collect=lambda *a, **k: SourceResult(skipped=SKIPPED_D27)),
    }
    names = only or [n for n in SOURCES if n != "youtube" or transcripts]  # D27: off by default
    unknown = [n for n in names if n not in all_]
    if unknown:
        raise ValueError(f"unknown source(s): {unknown}; known: {list(SOURCES)}")
    return [all_[n] for n in SOURCES if n in names]


def core_only_resolver(text: str, regs: list[Registry]) -> Resolution:
    """Snapshot runs resolve against the core registry only (no network). Hold-outs are refused."""
    hits = [e for r in regs for e in r.find(text)]
    if len(hits) != 1:
        return Resolution("not_found" if not hits else "ambiguous",
                          reason=f"{text!r} is not a unique core place")
    if hits[0].holdout:
        raise HoldoutError(f"{hits[0].name} is a hold-out: it is never collected into the snapshot")
    return Resolution("resolved", hits[0])


def snapshot_ingestor(collectors: list[Collector], core: Registry, clock: Callable[[], float],
                      request_counter: Callable[[], int],
                      request_log: RequestLog | None = None) -> PlaceIngestor:
    return PlaceIngestor(PlaceStore(STATUS_ROOT), core, collectors, core_only_resolver,
                         clock=clock, request_counter=request_counter, raw_root=RAW_ROOT,
                         request_log=request_log)


def collect_snapshot(names: list[str], ingestor: PlaceIngestor,
                     budget: Budget | None = None) -> list[PlaceStatus]:
    return [ingestor.ensure_place(n, budget) for n in names]
