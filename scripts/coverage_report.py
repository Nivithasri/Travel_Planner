"""Coverage of the core places by source type (D23). Offline: reads data/raw only.

    python scripts/coverage_report.py [--out data/manifests/coverage_report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.run import (  # noqa: E402  # noqa: E402
    CORE_REGISTRY,
    RAW_ROOT,
    STATUS_ROOT,
    raw_stats,
    render,
    transcript_outcomes,
)
from src.common.schemas import REPO_ROOT, load_collect_settings  # noqa: E402
from src.places.places import Registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "manifests" / "coverage_report.md"))
    args = ap.parse_args()
    cfg = load_collect_settings()
    skipped = {}
    if not (cfg.reddit_client_id and cfg.reddit_client_secret):
        skipped["reddit"] = "skipped: no REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET in .env (access " \
                            "requested, approval pending)"
    if not cfg.youtube_api_key:
        skipped["youtube"] = "skipped: no YOUTUBE_API_KEY in .env"
    text, data = render(Registry.load(CORE_REGISTRY).entries, RAW_ROOT, STATUS_ROOT, skipped)
    stats = raw_stats(RAW_ROOT)
    tr = transcript_outcomes(cfg.cache_dir, STATUS_ROOT)
    extra = ["", "## Raw size per source", "", "| source | records | places | bytes | body chars |",
             "|---|---:|---:|---:|---:|"]
    extra += [f"| {s} | {v['records']} | {v['places']} | {v['bytes']} | {v['body_chars']} |"
              for s, v in stats.items()]
    extra += ["", "## YouTube transcript outcomes (cached results; blocked from status files)", "",
              "| ok | none | disabled | blocked |", "|---:|---:|---:|---:|",
              f"| {tr['ok']} | {tr['none']} | {tr['disabled']} | {tr['blocked']} |"]
    text += "\n".join(extra) + "\n"
    data["raw_stats"], data["transcripts"] = stats, tr
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
