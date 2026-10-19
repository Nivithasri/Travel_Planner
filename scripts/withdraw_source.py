"""Withdraw collected data from the snapshot and prove nothing is left.

Two modes.

1. A source (raw records + one cache kind), e.g. the YouTube transcripts (D27):
    python scripts/withdraw_source.py --source youtube --cache-kind youtube_transcript --reason D27
    (add --execute to delete)

2. All cache entries of one host, e.g. the YouTube Data API search responses (30-day rule, D30):
    python scripts/withdraw_source.py --cache-host www.googleapis.com --reason D30
    (add --execute to delete)

Without --execute only lists what would be deleted. With --execute it picks phrases from the
data (memory only, never written, never on a command line), scans the working tree and git
history BEFORE deleting (a positive control: the scanner must see the data it is about to remove),
deletes, scans again, and prints counts only. Exit 5 if a tracked file or git history matches
after deletion.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.orchestrator import MANIFEST, RAW_ROOT, STATUS_ROOT  # noqa: E402
from src.collect.withdraw import (  # noqa: E402
    area_of,
    count_dir_files,
    delete_files,
    phrases_from,
    plan_cache_host,
    plan_withdrawal,
    retrieved_range,
    scan,
    search_metadata_phrases,
    withdraw,
)
from src.common.config import REPO_ROOT, load_collect_settings  # noqa: E402


def report_scan(label: str, counts: dict) -> None:
    files = counts.pop("files_with_matches")
    areas = collections.Counter(area_of(f) for f in files)
    print(f"{label}: {counts}")
    print(f"   files with matches by area (counts only): {dict(areas)}")


def host_mode(args, cfg) -> int:
    files = plan_cache_host(cfg.cache_dir, args.cache_host)
    span = retrieved_range(files)
    before = count_dir_files(cfg.cache_dir)
    print(f"cache entries of host {args.cache_host!r} in .cache/http: {len(files)}")
    print(f"oldest fetch: {span[0] if span else '-'} | newest fetch: {span[1] if span else '-'}")
    print(f"BEFORE .cache: {before}")
    if not args.execute:
        print("\n(dry run: nothing deleted; add --execute)")
        return 0
    phrases = search_metadata_phrases(files)  # titles and channel names, memory only
    print(f"\nphrases taken from the entries (titles and channel names incl. JSON spellings): "
          f"{len(phrases)} (not shown)")
    report_scan("PRE-DELETE scan (positive control: must find the data)", scan(REPO_ROOT, phrases))
    print(f"deleted {delete_files(files)} entries")
    print(f"AFTER  .cache: {count_dir_files(cfg.cache_dir)}")
    after = scan(REPO_ROOT, phrases)
    flagged = after["tracked_files"] or after["git_grep_revisions"] or after["git_log_p"]
    report_scan("POST-DELETE scan", after)
    if flagged:
        print("STOP: a tracked file or git history contains these titles or channel names.")
        return 5
    return 0


def source_mode(args, cfg) -> int:
    plan = plan_withdrawal(RAW_ROOT, cfg.cache_dir, args.source, args.cache_kind)
    print(f"raw record files ({len(plan.raw_files)}, {len(plan.records)} records), "
          f"cache entries of kind {args.cache_kind!r}: {len(plan.cache_files)}")
    print(f"BEFORE data/raw/{args.source}: {count_dir_files(RAW_ROOT / args.source)}")
    print(f"BEFORE .cache: {count_dir_files(cfg.cache_dir)}")
    if not args.execute:
        print("\n(dry run: nothing deleted; add --execute)")
        return 0
    phrases = phrases_from(plan)  # in memory only
    res = withdraw(plan, RAW_ROOT, STATUS_ROOT, MANIFEST, args.source, args.reason)
    print(f"\nwithdrawn: {res}")
    print(f"AFTER  data/raw/{args.source}: {count_dir_files(RAW_ROOT / args.source)}")
    print(f"AFTER  .cache: {count_dir_files(cfg.cache_dir)}")
    counts = scan(REPO_ROOT, phrases)
    report_scan("phrase scan", counts)
    flagged = counts["tracked_files"] or counts["git_grep_revisions"] or counts["git_log_p"]
    return 5 if flagged else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--cache-kind")
    ap.add_argument("--cache-host", help="delete the .cache/http entries of this host")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    cfg = load_collect_settings()
    if args.cache_host:
        return host_mode(args, cfg)
    if not (args.source and args.cache_kind):
        ap.error("give --cache-host, or --source together with --cache-kind")
    return source_mode(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
