"""Resolve config/core_places_list.json against Wikidata/Wikipedia/Wikivoyage and print the table.

Live network use (cached). Needs CONTACT_EMAIL in .env. Writes data/registry/core_places.json only
with --write, which must wait for the project owner's approval of the printed table.

The popular/hidden_gem threshold is the one the auto-selection experiment pool produced (D19: top
20% of the filter-passing non-hub candidates). That experiment code was removed; the threshold and
pool size are now read from the existing data/registry/core_places.json meta. Labels are never set
by hand.

    python scripts/historical/build_core_list.py --historical [--as-of YYYY-MM-DD] [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.allowlist import refuse_without_historical  # noqa: E402
from src.common.cache import DiskCache  # noqa: E402
from src.common.config import REPO_ROOT, load_collect_settings  # noqa: E402
from src.common.http import HttpClient  # noqa: E402
from src.common.requestlog import DEFAULT_REQUEST_LOG, RequestLog  # noqa: E402
from src.places.core_list import (  # noqa: E402
    CORE_HUB_CAP,
    POPULAR_SHARE,
    label_for,
    load_core_list,
    resolve_listed,
    to_entry,
)
from src.places.registry import (
    Registry,  # noqa: E402
    clean_aliases,  # noqa: E402
)
from src.places.wikidata import WikidataClient  # noqa: E402
from src.places.wv_lists import fetch_wikitext  # noqa: E402

OUT = REPO_ROOT / "data" / "registry" / "core_places.json"
HOST_DELAYS = {"wikimedia.org": 0.2, "query.wikidata.org": 1.0, "www.wikidata.org": 0.5,
               "en.wikivoyage.org": 0.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--historical", action="store_true",
                    help="allow this one-off script to run: it bypasses the D43 allowlist")
    args = ap.parse_args()
    refused = refuse_without_historical("build_core_list.py", args.historical)
    if refused:
        return refused
    as_of = date.fromisoformat(args.as_of)
    cfg = load_collect_settings()
    http = HttpClient(DiskCache(cfg.cache_dir), cfg.user_agent, host_delays=HOST_DELAYS,
                      request_log=RequestLog(DEFAULT_REQUEST_LOG))
    wd = WikidataClient(http)
    listed = load_core_list()

    def wikitext_of(title: str) -> str | None:
        got = fetch_wikitext(http, title)
        return got[1] if got else None

    # threshold from the experiment pool, stored in the existing registry meta (the experiment
    # code is removed; the D19 formula is frozen)
    stored = json.loads(OUT.read_text("utf-8"))["meta"]
    threshold, pool_size = stored["popular_cutoff_views"], stored["popular_pool_size"]

    results = [resolve_listed(wd, p, wikitext_of, as_of) for p in listed]
    ok = [r.candidate for r in results if r.candidate is not None]
    holdouts = [c for c in ok if c.holdout]  # explicit, from the list file (D17)
    ranked = sorted(ok, key=lambda c: (-(c.views or 0), c.name))
    labels = {c.qid: label_for(c.views, threshold) for c in ranked}

    n_hub = sum(p.hub for p in listed)
    print(f"as_of={args.as_of} listed={len(listed)} resolved={len(ok)} hubs={n_hub} "
          f"(cap {CORE_HUB_CAP}) network_requests={http.requests_made}")
    print(f"POPULAR THRESHOLD = {threshold:.0f} views (top 20% of {pool_size} filter-passing "
          f"non-hub pool candidates, experiment rules)")
    by_state: dict[str, int] = {}
    for c in ranked:
        by_state[c.state] = by_state.get(c.state, 0) + 1
    print("per state:", ", ".join(f"{k} {v}" for k, v in sorted(by_state.items())))
    print(f"popular={sum(v == 'popular' for v in labels.values())} "
          f"hidden_gem={sum(v == 'hidden_gem' for v in labels.values())}")

    print(f"\n{'rank':>4}  {'name':26} {'state':5} {'signal':>8}  {'label':10} {'hub':3} "
          f"{'hold-out':8} {'listings':>8} {'words':>5}  {'wikidata':10} coord source / flags")
    for i, c in enumerate(ranked, 1):
        r = next(x for x in results if x.candidate is c)
        n = "n/a" if c.n_see_do is None else str(c.n_see_do)
        w = "n/a" if c.see_do_words is None else str(c.see_do_words)
        sig = "-" if c.views is None else str(c.views)
        print(f"{i:>4}  {c.name[:26]:26} {c.state:5} {sig:>8}  {labels[c.qid]:10} "
              f"{'yes' if c.hub else '':3} {'HOLD-OUT' if c.holdout else '':8} {n:>8} {w:>5}  "
              f"{c.qid:10} wikidata:P625" + (f"  FLAGS: {' | '.join(r.flags)}" if r.flags else ""))

    print("\nRESOLVED ITEMS (verify these are the right things)")
    for c in ranked:
        print(f"  {c.name:20} {c.qid:10} en.wikipedia={c.wp_title!s:34} "
              f"types={','.join(c.types[:3])} | district={c.district} | {c.lat:.4f},{c.lon:.4f}")
    print("\nHOLD-OUTS (explicit, not collected into the snapshot):",
          ", ".join(f"{c.name} ({c.state})" for c in holdouts) or "none")
    print("\nSTATE x LABEL")
    for st in ("TN", "KL", "KA", "AP", "TS", "PY"):
        row = [c for c in ranked if c.state == st]
        print(f"  {st}: total {len(row)}  popular {sum(labels[c.qid] == 'popular' for c in row)}"
              f"  hidden_gem {sum(labels[c.qid] == 'hidden_gem' for c in row)}"
              f"  hold-out {sum(c.holdout for c in row)}")

    failed = [r for r in results if not r.ok]
    print(f"\nFAILED TO RESOLVE ({len(failed)})")
    for r in failed:
        print(f"  {r.place.name} ({r.place.state}): {r.failure}")
    print("\nCHECK THESE (flags)")
    for r in results:
        if r.candidate and (r.flags or r.alternatives):
            print(f"  {r.candidate.name} -> {r.candidate.qid}: {' | '.join(r.flags) or 'ok'}"
                  + (f"  [other matches: {'; '.join(r.alternatives)}]" if r.alternatives else ""))

    if args.write:
        if failed:
            print("\nrefusing to write: some places did not resolve")
            return 1
        entries, dropped = clean_aliases([to_entry(c, labels[c.qid]) for c in ranked])
        for d in dropped:
            print(f"alias dropped: {d.place!r} {d.alias!r} ({d.reason})")
        reg = Registry(
            entries,
            {"built_for_date": args.as_of, "source": "config/core_places_list.json",
             "popular_cutoff_views": threshold, "popular_pool_size": pool_size,
             "popular_share": POPULAR_SHARE, "list_size": len(ranked),
             "aliases_dropped": [{"place": d.place, "alias": d.alias, "reason": d.reason}
                                 for d in dropped],
             "holdouts": sorted(c.name for c in holdouts),
             "holdout_rule": "explicit in config/core_places_list.json (D17)"},
        )
        reg.save(OUT)
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
