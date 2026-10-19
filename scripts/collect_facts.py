"""Fetch raw facts (D37) for core snapshot places into data/raw/<source>/<place>/*.json.

    python scripts/collect_facts.py --source wikidata --places Munnar,Hampi            (plan only)
    python scripts/collect_facts.py --source wikidata --places Munnar,Hampi --execute

Sources: wikidata (Special:EntityData), overpass (airports once + stations per place),
open_meteo (ERA5 archive, one place per request). Hold-outs are refused (exit 2); a 403, 406 or 429
stops the run (exit 3); the cap ends it (exit 4). No place_facts table is built (Phase 5).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.orchestrator import (  # noqa: E402
    CORE_REGISTRY,
    MANIFEST,
    RAW_ROOT,
    load_run_config,
    make_http,
)
from src.collect.types import BlockDetected  # noqa: E402
from src.common.config import load_collect_settings  # noqa: E402
from src.common.http import RequestBudgetExhausted  # noqa: E402
from src.facts import openmeteo, overpass, wikidata  # noqa: E402
from src.facts.envelope import backfill_manifest  # noqa: E402
from src.facts.hosts import load_api_hosts  # noqa: E402
from src.places.registry import Registry, slugify  # noqa: E402

HOST_OF = {"wikidata": "www.wikidata.org", "overpass": "overpass-api.de",
           "open_meteo": "archive-api.open-meteo.com"}
# Lines the fetch tool reported for wikidata.org/robots.txt on 2026-10-18 (NOT the live file).
REPORTED_ROBOTS = """User-agent: *
Disallow: /w/
Allow: /w/api.php?action=mobileview&
Disallow: /wiki/Special:EntityData/
Allow: /wiki/Special:EntityData/*.
"""


def robots_verdict(text: str, ua: str, url: str) -> bool:
    rp = RobotFileParser()
    rp.parse(text.splitlines())
    return rp.can_fetch(ua, url)


def elevation_from_raw(place: str) -> float | None:
    path = RAW_ROOT / "wikidata" / slugify(place) / "entity.json"
    if not path.exists():
        return None
    env = json.loads(path.read_text(encoding="utf-8"))
    qid = next(iter(env["response"]["entities"]))
    basic = wikidata.extract_basic(env["response"]["entities"][qid])
    elev = (basic.get("elevation") or {}).get("value")
    return elev["amount"] if isinstance(elev, dict) else None


def planned_requests(source: str, n_places: int) -> int:
    return 1 if source == "overpass" else n_places  # overpass: the one airport query (D40)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(HOST_OF))
    ap.add_argument("--places")
    ap.add_argument("--backfill-manifest", action="store_true",
                    help="add raw-manifest lines for envelopes already on disk (no network)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=3000.0)
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--no-elevation", action="store_true",
                    help="open_meteo: omit elevation; save a separate *_noelev envelope")
    ap.add_argument("--min-delay-s", type=float,
                    help="override the host delay for this run (not a config change)")
    args = ap.parse_args()
    if args.backfill_manifest:
        n = backfill_manifest(RAW_ROOT, args.source, MANIFEST)
        print(f"manifest lines added for {args.source}: {n} (no network)")
        return 0
    if not args.places and args.source != "overpass":
        ap.error("--places is required")
    cfg = load_collect_settings()
    core = Registry.load(CORE_REGISTRY)
    entries = []
    for n in [x.strip() for x in (args.places or "").split(",") if x.strip()]:
        hits = core.find(n)
        if len(hits) != 1:
            print(f"not a unique core place: {n!r}")
            return 2
        if hits[0].holdout:
            print(f"{hits[0].name} is a hold-out and is never collected (D17)")
            return 2
        entries.append(hits[0])
    hosts = load_api_hosts()
    host = hosts.get(HOST_OF[args.source])
    if host is None:
        print(f"{HOST_OF[args.source]} is not in config api_hosts (D37)")
        return 2
    cap = min(args.max_requests or host.max_requests_per_run, host.max_requests_per_run)
    rc = load_run_config()
    delay = args.min_delay_s if args.min_delay_s is not None else host.min_delay_s
    http = make_http(cfg, max_requests=cap, run_config=rc, host_delays={host.host: delay})
    http.max_retries = 0
    if host.read_timeout_s:
        http.timeout = (10.0, host.read_timeout_s)
    if args.source == "overpass":
        http.timeout = (10.0, 90.0)  # above the server's own [timeout:60]
    http.request_log.source = args.source if http.request_log else None
    n_req = planned_requests(args.source, len(entries))
    print("PRE-FLIGHT")
    print(f"  source: {args.source} | host: {host.host} | path prefix {host.path_prefix}")
    print(f"  planned requests: {n_req} on {host.host}"
          f"{' (+1 robots.txt, informational)' if args.source == 'wikidata' else ''}"
          f" | cap {cap} | min delay {delay:g} s | read timeout "
          f"{host.read_timeout_s or http.timeout[1]:g} s"
          f" | robots_exempt {host.robots_exempt} (D36)")
    print(f"  User-Agent: {cfg.user_agent.replace(cfg.contact_email, '<CONTACT_EMAIL masked>')}")
    refused = [e.name for e in core.entries if e.holdout]
    print(f"  hold-outs refused before any request (exit 2): {refused}")
    print(f"  places ({len(entries)}): {[e.name for e in entries]}")
    if args.source == "wikidata" and entries:
        url = wikidata.entity_url(entries[0].wikidata_id)
        v = robots_verdict(REPORTED_ROBOTS, cfg.user_agent, url)
        print(f"  robots parser on the lines the fetch tool reported (NOT the live file): "
              f"can_fetch({url.split('.org')[1]}) = {v}")
    if not args.execute:
        print("\n(plan only: nothing was fetched; add --execute to run)")
        return 0
    print("\nEXECUTING")
    run_start = time.time()
    deadline = run_start + args.deadline_s
    failed: list[str] = []
    rows: list[dict] = []

    def table() -> None:
        if args.source != "open_meteo":
            return
        print("STATUS TABLE: place | status | cache | days | nulls | bytes | elapsed_s | check")
        for r in rows:
            if "error" in r:
                print(f"  {r['place']} | ERROR | {r['error']}")
                continue
            print(f"  {r['place']} | {r['status']} | {'cache' if r['from_cache'] else 'network'} |"
                  f" {r['days']} | {r['nulls']} | {r['bytes']} | {r.get('elapsed_s')} |"
                  f" {'ok' if r['check_ok'] else 'FAIL'}")
        print(f"places failing the 3653-days/0-nulls check: {failed or 'none'}")

    try:
        if args.source == "wikidata":
            rob = http.fetch("https://www.wikidata.org/robots.txt", check_robots=False)
            url = wikidata.entity_url(entries[0].wikidata_id)
            print(f"  live robots.txt HTTP {rob.status}: parser can_fetch(Special:EntityData) = "
                  f"{robots_verdict(rob.text, cfg.user_agent, url)} (informational under D36)")
        if args.source == "overpass":
            rep = overpass.fetch_airports(http, host, RAW_ROOT, deadline, MANIFEST)
            print(" ", rep)
            if "error" in rep:
                print(f"STOPPED: {rep['error']}")
                return 5
        for e in ([] if args.source == "overpass" else entries):
            t0 = time.time()
            if args.source == "wikidata":
                rep = wikidata.fetch_place(http, host, e, RAW_ROOT, deadline, MANIFEST)
            else:
                rep = openmeteo.fetch_place(
                    http, host, e, RAW_ROOT,
                    None if args.no_elevation else elevation_from_raw(e.name), deadline, MANIFEST,
                    "archive_2015_2024_noelev.json" if args.no_elevation
                    else "archive_2015_2024.json")
            rep["elapsed_s"] = round(time.time() - t0, 1)
            print(" ", rep)
            rows.append(rep)
            if "error" in rep:
                print(f"STOPPED on the first failure: {rep['error']}")
                table()
                return 5
            if args.source == "open_meteo" and not rep.get("check_ok", True):
                failed.append(e.name)
    except BlockDetected as exc:
        print(f"\nSTOPPED ON A BLOCK: {exc}\nnetwork requests so far: {http.requests_made}")
        print("No retry and no workaround. Report it; do not rerun until the owner decides.")
        table()
        return 3
    except RequestBudgetExhausted as exc:
        print("STOPPED: " + str(exc))
        table()
        return 4
    table()
    print(f"total wall time: {time.time() - run_start:.1f} s")
    print(f"total network requests counted against the cap: {http.requests_made}")
    print("per host:", http.requests_by_host())
    print(f"circuit open hosts: {http.circuit_open_hosts or 'none'} | skips {http.circuit_skips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
