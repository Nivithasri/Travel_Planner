"""ensure_place(place_text, budget) -> PlaceStatus  (docs/contracts.md, D17).

Resolve free text, reject places outside scope, then run every collector whose source record is
missing, expired or incomplete. A fresh place makes no network call. Time comes from an injected
clock, so freshness is testable without waiting. Phase 1 never sets status "ready" (chunks are
not indexed yet).

Raw data for live places goes under the live store (data/live/raw), never into data/raw, which
belongs to the frozen snapshot.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from src.collect.store import PlaceStore
from src.collect.types import (
    DEFAULT_TTL_S,
    ERROR_RETRY_S,
    Budget,
    Collector,
    PlaceStatus,
    SourceResult,
)
from src.common.requestlog import RequestLog
from src.places.registry import Registry
from src.places.resolve import Resolution

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
