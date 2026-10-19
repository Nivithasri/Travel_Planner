"""Run the collectors over the core snapshot places (data/raw) or over any place on demand.

The snapshot run never touches hold-out places (D17) and never resolves anything over the network:
places come from data/registry/core_places.json. Per-place progress is stored under
data/raw/_status (git-ignored) so a stopped run resumes where it stopped and a finished one makes
no request.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.collect.ensure import PlaceIngestor
from src.collect.reddit import RedditCollector
from src.collect.store import PlaceStore
from src.collect.types import SOURCES, Budget, Collector, PlaceStatus
from src.collect.web import WebCollector
from src.collect.wiki import WikiCollector
from src.collect.youtube import YouTubeCollector
from src.common.allowlist import Allowlist
from src.common.cache import DiskCache
from src.common.config import REPO_ROOT, CollectSettings
from src.common.http import HttpClient
from src.common.requestlog import DEFAULT_REQUEST_LOG, RequestLog
from src.places.registry import Registry
from src.places.resolve import Resolution

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
    transcripts = (run_config or load_run_config())["youtube_transcripts"]
    all_: dict[str, Collector] = {
        "wikivoyage": WikiCollector("wikivoyage", http, manifest),
        "wikipedia": WikiCollector("wikipedia", http, manifest),
        "web": WebCollector(http, manifest=manifest),
        "reddit": RedditCollector(http, cfg.reddit_client_id, cfg.reddit_client_secret, manifest),
        "youtube": YouTubeCollector(),  # D27: refusal stub, never runs
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
