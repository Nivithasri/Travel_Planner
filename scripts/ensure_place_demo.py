"""Never-seen-place demo: ensure_place for one free-text place (D17), live store only.

    python scripts/ensure_place_demo.py --place Yercaud            (plan only; cap defaults to 25)
    python scripts/ensure_place_demo.py --place Yercaud --execute

Writes only under data/live/ (registry.json, places/<slug>.json, raw/...). data/raw and the raw
manifest are never touched; youtube and reddit are off. A second identical run must print
network_requests: 0. Exit 2: refused before any request; 3: block (403/429); 4: request cap.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.ensure import PlaceIngestor  # noqa: E402
from src.collect.orchestrator import (  # noqa: E402
    CORE_REGISTRY,
    RAW_ROOT,
    build_collectors,
    load_run_config,
    make_http,
)
from src.collect.store import PlaceStore  # noqa: E402
from src.collect.types import BlockDetected, Budget  # noqa: E402
from src.common.config import REPO_ROOT, load_collect_settings  # noqa: E402
from src.common.http import RequestBudgetExhausted  # noqa: E402
from src.places.registry import Registry, normalize  # noqa: E402
from src.places.resolve import resolve_place  # noqa: E402
from src.places.wikidata import WikidataClient  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", required=True)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=900.0)
    args = ap.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
