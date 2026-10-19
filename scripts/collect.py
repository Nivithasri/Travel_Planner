"""Phase 1 collection runner. Subcommands (each keeps its old flags and exit codes):

    python scripts/collect.py core  --places <names> [--sources ...] [--execute]   (collect_core)
    python scripts/collect.py facts --source <s> --places <names> [--execute]      (collect_facts)
    python scripts/collect.py demo  --place <name> [--cap N] [--execute]           (ensure_place)

Without --execute every subcommand prints its plan and makes no request. Exit codes: 0 ok,
2 hold-out or bad input, 3 block (HTTP 403/406/429), 4 request cap reached, 5 (facts) a fetch error
stopped the run. `python scripts/collect.py <subcommand> --help` lists each subcommand's flags.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.run import (  # noqa: E402  # noqa: E402
    CORE_REGISTRY,
    MANIFEST,
    RAW_ROOT,
    REQUEST_LOG,
    BlockDetected,
    Budget,
    HoldoutError,
    PlaceIngestor,
    PlaceStore,
    build_collectors,
    collect_snapshot,
    core_only_resolver,
    load_run_config,
    make_http,
    snapshot_ingestor,
)
from src.collect.sources import applies_to  # noqa: E402
from src.common.http import (  # noqa: E402
    CIRCUIT_THRESHOLD,
    HttpClient,
    RequestBudgetExhausted,
    TransportResponse,
)
from src.common.schemas import (  # noqa: E402
    REPO_ROOT,  # noqa: E402
    CollectSettings,
    load_collect_settings,
)
from src.facts import facts as openmeteo  # noqa: E402
from src.facts import facts as overpass  # noqa: E402
from src.facts import facts as wikidata  # noqa: E402
from src.facts.facts import (  # noqa: E402
    backfill_manifest,
    load_api_hosts,
)
from src.places.places import (  # noqa: E402
    Registry,  # noqa: E402
    WikidataClient,
    normalize,
    resolve_place,
    slugify,  # noqa: E402
)

# ==== core (was scripts/collect_core.py) ========================================
__doc_core__ = """Collect raw data for core snapshot places into data/raw (never hold-outs).

Always prints a PRE-FLIGHT first (cap, sources, masked User-Agent, hold-out refusals, domains per
place with their scopes, and a no-network simulation of the hosts the run would contact).
Without --execute that is all it does. Live use needs --execute. Every request goes through the
polite cached client, is written to data/manifests/request_log.jsonl, and reruns come from cache.

    python scripts/collect.py core --places Wayanad,Kodagu --sources wikivoyage,wikipedia,web
    python scripts/collect.py core --places Wayanad,Kodagu --execute"""

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


def core_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="collect.py core", description=__doc_core__)
    ap.add_argument("--places", required=True, help="comma-separated core place names")
    ap.add_argument("--sources", help="comma-separated subset of wikivoyage,wikipedia,web,reddit")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=900.0)
    ap.add_argument("--max-requests", type=int,
                    help="lower the hard cap from config/sources.json (cannot raise it)")
    args = ap.parse_args(argv)
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


# ==== facts (was scripts/collect_facts.py) ========================================
__doc_facts__ = """Fetch raw facts (D37) for core places into data/raw/<source>/<place>/*.json.

    python scripts/collect.py facts --source wikidata --places Munnar,Hampi            (plan only)
    python scripts/collect.py facts --source wikidata --places Munnar,Hampi --execute

Sources: wikidata (Special:EntityData), overpass (the one airport query, D40),
open_meteo (ERA5 archive, one place per request). Hold-outs are refused (exit 2); a 403, 406 or 429
stops the run (exit 3); the cap ends it (exit 4). No place_facts table is built (Phase 5)."""

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


def facts_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="collect.py facts", description=__doc_facts__)
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
    args = ap.parse_args(argv)
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
                rep = wikidata.wikidata_fetch_place(http, host, e, RAW_ROOT, deadline, MANIFEST)
            else:
                rep = openmeteo.open_meteo_fetch_place(
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


# ==== demo (was scripts/ensure_place_demo.py) ========================================
__doc_demo__ = """Never-seen-place demo: ensure_place for one free-text place (D17), live store
only.

    python scripts/collect.py demo --place Yercaud            (plan only; cap defaults to 25)
    python scripts/collect.py demo --place Yercaud --execute

Writes only under data/live/ (registry.json, places/<slug>.json, raw/...). data/raw and the raw
manifest are never touched; youtube and reddit are off. A second identical run must print
network_requests: 0. Exit 2: refused before any request; 3: block (403/429); 4: request cap."""

LIVE_ROOT = REPO_ROOT / "data" / "live"
DEFAULT_CAP = 25  # hard cap on network requests (resolver and collectors); no retry (D43 note)
SOURCES = ["wikivoyage", "wikipedia", "web"]


def assert_live_only(store: PlaceStore, raw_root: Path = RAW_ROOT) -> None:
    """The live store must sit under data/live and never under data/raw (the frozen snapshot)."""
    live, snap = store.root.resolve(), Path(raw_root).resolve()
    if snap == live or snap in live.parents or live in snap.parents:
        raise ValueError(f"live store {live} overlaps the snapshot raw root {snap}")
    if store.raw_root.resolve() == snap or snap in store.raw_root.resolve().parents:
        raise ValueError("live raw root is inside data/raw")


def known_as(reg: Registry, text: str) -> list[str]:
    """Registry entries (hold-outs marked) that match the text by name or alias."""
    return [f"{e.name}{' [hold-out]' if e.holdout else ''}" for e in reg.find(text)]


def tree_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def live_files() -> list[str]:
    return sorted(p.relative_to(LIVE_ROOT).as_posix()
                  for p in LIVE_ROOT.rglob("*") if p.is_file())


def print_plan(args, cfg, core, store, collectors, live) -> None:
    holdouts = [e.name for e in core.entries if e.holdout]
    is_holdout = any(normalize(args.place) == normalize(h) for h in holdouts)
    srcs = [c.source for c in collectors]
    print("PLAN")
    print(f"  place text: {args.place!r}")
    print(f"  in the core registry (name or alias): {known_as(core, args.place) or 'no'}")
    print(f"  in the live registry: {known_as(live, args.place) or 'no'}")
    print(f"  is a hold-out {holdouts}: {'YES' if is_holdout else 'no'}")
    reddit = "ON" if cfg.reddit_client_id else "off (no credentials)"
    print(f"  sources that will run: {srcs} | reddit: {reddit} | "
          f"youtube: {'in list' if 'youtube' in srcs else 'off (D27)'}")
    print("  resolver hosts (hard-coded in src/places, NOT in config/sources.json; D37 note):")
    print("    www.wikidata.org /w/api.php, query.wikidata.org /sparql, wikimedia.org pageviews")
    web = next(c for c in collectors if c.source == "web")
    print("  collector hosts: en.wikivoyage.org and en.wikipedia.org /w/api.php; web domains: "
          + ", ".join(web.allow))
    print(f"  request cap (resolver and collectors share it): {args.cap}; the resolver alone can "
          "use about 1 + 2 per candidate (up to 7) + 1 pageviews request")
    print(f"  output directory: {LIVE_ROOT} (raw files: {store.raw_root})")
    print("  data/raw writable by this script: no (live store is outside data/raw, manifest=None)")
    print(f"  data/raw fingerprint before: {tree_fingerprint(RAW_ROOT)}")


def demo_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="collect.py demo", description=__doc_demo__)
    ap.add_argument("--place", required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=900.0)
    args = ap.parse_args(argv)
    cfg = load_collect_settings()
    core = Registry.load(CORE_REGISTRY)
    store = PlaceStore(LIVE_ROOT)
    assert_live_only(store)
    live = store.load_registry()
    rc = load_run_config()
    http = make_http(cfg, max_requests=args.cap, run_config=rc)
    http.max_retries = 0
    collectors = build_collectors(cfg, http, manifest=None, only=SOURCES, run_config=rc)
    print_plan(args, cfg, core, store, collectors, live)
    if any("[hold-out]" in n for n in known_as(core, args.place)):
        print("  refused: hold-outs are never ingested (D17)")
        return 2
    if not args.execute:
        print("\n(plan only: nothing was fetched; add --execute to run)")
        return 0
    meta = core.meta or {}
    as_of = date.fromisoformat(meta.get("built_for_date", "2026-10-18"))
    wd = WikidataClient(http)

    def resolver(text, regs):
        return resolve_place(text, regs, wd, as_of, meta.get("popular_cutoff_views"))

    ing = PlaceIngestor(store, core, collectors, resolver, clock=time.time,
                        request_counter=lambda: http.requests_made, raw_root=store.raw_root,
                        request_log=http.request_log)
    if http.request_log:
        http.request_log.source = "ensure_place"
    print("\nEXECUTING")
    try:
        status = ing.ensure_place(args.place, Budget(deadline=time.time() + args.deadline_s))
    except BlockDetected as exc:
        print(f"STOPPED ON A BLOCK: {exc}; requests so far: {http.requests_made}")
        return 3
    except RequestBudgetExhausted as exc:
        print(f"STOPPED AT THE CAP: {exc}; per host: {http.requests_by_host()}")
        print(f"files under data/live now: {live_files()}")
        print(f"data/raw fingerprint after: {tree_fingerprint(RAW_ROOT)}")
        return 4
    doc = asdict(status)
    print(f"place: {doc['place']} | state: {doc['state']} | status: {doc['status']}"
          f" | reason: {doc['reason']}")
    for src, rec in (doc.get("sources") or {}).items():
        print(f"  {src}: n_raw={rec.get('n_raw')} n_requests={rec.get('n_requests')} "
              f"error={rec.get('error')} skipped={rec.get('skipped')}")
    print(f"network_requests: {doc['network_requests']}")
    print(f"requests counted this process: {http.requests_made} | per host: "
          f"{http.requests_by_host()}")
    print(f"files under data/live: {live_files()}")
    print(f"data/raw fingerprint after: {tree_fingerprint(RAW_ROOT)}")
    return 0


SUBCOMMANDS = {"core": core_main, "facts": facts_main, "demo": demo_main}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] not in SUBCOMMANDS:
        print(f"unknown subcommand {argv[0]!r}; use one of {', '.join(SUBCOMMANDS)}",
              file=sys.stderr)
        return 2
    return SUBCOMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
