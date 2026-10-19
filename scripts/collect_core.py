"""Collect raw data for core snapshot places into data/raw (never hold-outs).

Always prints a PRE-FLIGHT first (cap, sources, masked User-Agent, hold-out refusals, domains per
place with their scopes, and a no-network simulation of the hosts the run would contact).
Without --execute that is all it does. Live use needs --execute. Every request goes through the
polite cached client, is written to data/manifests/request_log.jsonl, and reruns come from cache.

    python scripts/collect_core.py --places Wayanad,Kodagu --sources wikivoyage,wikipedia,web
    python scripts/collect_core.py --places Wayanad,Kodagu --sources wikivoyage,wikipedia,web \
        --execute
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.ensure import PlaceIngestor  # noqa: E402
from src.collect.orchestrator import (  # noqa: E402
    CORE_REGISTRY,
    REQUEST_LOG,
    HoldoutError,
    build_collectors,
    collect_snapshot,
    core_only_resolver,
    load_run_config,
    make_http,
    snapshot_ingestor,
)
from src.collect.store import PlaceStore  # noqa: E402
from src.collect.types import BlockDetected, Budget  # noqa: E402
from src.collect.web import applies_to  # noqa: E402
from src.common.config import CollectSettings, load_collect_settings  # noqa: E402
from src.common.http import (  # noqa: E402
    CIRCUIT_THRESHOLD,
    HttpClient,
    RequestBudgetExhausted,
    TransportResponse,
)
from src.places.registry import Registry  # noqa: E402

WIKI_STUB = json.dumps({"query": {"pages": [{"title": "X", "extract": "stub text " * 50,
                                             "revisions": [{"revid": 1, "timestamp":
                                                            "2026-01-01T00:00:00Z"}]}]}})


def mask_ua(cfg: CollectSettings) -> str:
    return cfg.user_agent.replace(cfg.contact_email, "<CONTACT_EMAIL from .env, masked>")


def simulate(cfg: CollectSettings, core: Registry, names: list[str], only: list[str] | None,
             rc: dict) -> collections.Counter:
    """Run the real orchestrator code against a FAKE transport (no network, temp cache and
    store) and count the hosts it would contact."""
    seen: collections.Counter = collections.Counter()

    def fake(method, url, headers, timeout):
        seen[url.split("/")[2]] += 1
        if "api.php" in url:
            return TransportResponse(200, {}, WIKI_STUB)
        return TransportResponse(404, {}, "")

    with tempfile.TemporaryDirectory() as tmp:
        sim_cfg = CollectSettings(cfg.contact_email, cfg.user_agent, Path(tmp) / "cache",
                                  reddit_client_id=cfg.reddit_client_id,
                                  reddit_client_secret=cfg.reddit_client_secret,
                                  youtube_api_key=cfg.youtube_api_key)
        http = make_http(sim_cfg, transport=fake, request_log=None, run_config=rc)
        http.min_delay_s = 0
        http.host_delays.clear()
        http._sleep = lambda s: None
        cols = build_collectors(sim_cfg, http, None, only, run_config=rc)
        ing = PlaceIngestor(PlaceStore(Path(tmp) / "live"), core, cols, core_only_resolver,
                            clock=time.time, request_counter=lambda: http.requests_made)
        collect_snapshot(names, ing, Budget())
    return seen


def circuit_report(http: HttpClient) -> str:
    return (f"circuit open hosts: {http.circuit_open_hosts or 'none'} | skipped calls: "
            f"{http.circuit_skips} {http.circuit_skips_by_host or ''}")


def preflight(cfg, core, names, entries, collectors, only, args, rc) -> None:
    cap = rc["max_requests"]
    if args.max_requests is not None:
        cap = min(args.max_requests, cap)
    print("PRE-FLIGHT")
    print(f"  config run_limits.max_requests_per_run = {rc['max_requests']}"
          f" (effective cap this run: {cap}; cache hits and TLS/DNS failures do not count)")
    print(f"  config youtube_transcripts = {rc['youtube_transcripts']}")
    srcs = [c.source for c in collectors]
    print(f"  sources that will run: {srcs}")
    print(f"  youtube in the source list: {'YES' if 'youtube' in srcs else 'no'}")
    print(f"  User-Agent: {mask_ua(cfg)}")
    refused = []
    for e in core.entries:
        if e.holdout:
            try:
                core_only_resolver(e.name, [core])
            except HoldoutError:
                refused.append(e.name)
    print(f"  hold-outs refused before any request: {refused}")
    print(f"  places ({len(names)}): {names}")
    web = next((c for c in collectors if c.source == "web"), None)
    for n in names:
        e = entries[n]
        doms = []
        for d, v in (web.allow.items() if web else []):
            if applies_to(v, e):
                scope = v.get("states") or v.get("places")
                doms.append(d + (f" [scope {scope}]" if scope else ""))
        print(f"    {e.name} [{e.state.value}]: wikivoyage={'yes' if e.wikivoyage_title else 'no'}"
              f" wikipedia={'yes' if e.wikipedia_title else 'no'} web: {', '.join(doms) or '-'}")
    seen = simulate(cfg, core, names, only, rc)
    print(f"  SIMULATION (fake transport, no network): hosts contacted = {dict(seen)}")
    print(f"  googleapis.com requests in the simulation: "
          f"{sum(n for h, n in seen.items() if 'googleapis' in h)}; "
          f"youtube.com requests: {sum(n for h, n in seen.items() if 'youtube' in h)}")
    key_used = any(getattr(c, "api_key", None) for c in collectors)
    print(f"  YOUTUBE_API_KEY handed to any collector: {'YES' if key_used else 'no'}")
    print(f"  circuit breaker: opens a host after {CIRCUIT_THRESHOLD} consecutive connect/timeout "
          f"failures (skips are not counted against the cap)")
    print(f"  request log: {REQUEST_LOG}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", required=True, help="comma-separated core place names")
    ap.add_argument("--sources", help="comma-separated subset of wikivoyage,wikipedia,web,reddit")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument("--max-requests", type=int,
                    help="lower the hard cap from config/sources.json (cannot raise it)")
    args = ap.parse_args()
    cfg = load_collect_settings()
    core = Registry.load(CORE_REGISTRY)
    names = [n.strip() for n in args.places.split(",") if n.strip()]
    entries = {}
    for n in names:
        hits = core.find(n)
        if len(hits) != 1:
            print(f"not a unique core place: {n!r}")
            return 2
        if hits[0].holdout:
            print(f"{hits[0].name} is a hold-out and is never collected (D17)")
            return 2
        entries[n] = hits[0]
    rc = load_run_config()
    only = args.sources.split(",") if args.sources else None
    http = make_http(cfg, max_requests=args.max_requests, run_config=rc)  # logs by default
    http.max_retries = 0  # a failure is reported, never retried in a loop
    collectors = build_collectors(cfg, http, only=only, run_config=rc)
    preflight(cfg, core, names, entries, collectors, only, args, rc)
    if not args.execute:
        print("\n(plan only: nothing was fetched; add --execute to run)")
        return 0
    print("\nEXECUTING")
    ing = snapshot_ingestor(collectors, core, time.time, lambda: http.requests_made,
                            http.request_log)
    try:
        results = collect_snapshot(names, ing, Budget(deadline=time.time() + args.deadline_s))
    except HoldoutError as exc:
        print(exc)
        return 2
    except BlockDetected as exc:
        print(f"\nSTOPPED ON A BLOCK: {exc}\nnetwork requests so far: {http.requests_made}")
        print(circuit_report(http))
        print("No retry and no workaround. Report it; do not rerun until the owner decides.")
        return 3
    except RequestBudgetExhausted as exc:
        print("STOPPED: " + str(exc))
        print("per host:", http.requests_by_host(), "| unreachable:", http.unreachable_hosts)
        print(circuit_report(http))
        print("Kept on disk: raw records so far, finished collectors' status, caches, request log.")
        return 4
    for st in results:
        print(f"{st.place}: {st.status} requests={st.network_requests} {st.reason}")
        for s, rec in st.sources.items():
            print(f"   {s:11} raw={rec['n_raw']} req={rec['n_requests']} "
                  f"{'SKIPPED: ' + rec['skipped'] if rec['skipped'] else ''}"
                  f"{'ERROR: ' + rec['error'] if rec['error'] else ''}")
            for note in rec.get("notes") or []:
                print(f"      note: {note}")
    print(f"total network requests counted against the cap: {http.requests_made}"
          f" (TLS/DNS attempts not counted: {http.unreachable_attempts})")
    print(circuit_report(http))
    print("per host:", http.requests_by_host())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
