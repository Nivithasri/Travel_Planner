"""Fetch the YouTube API policy pages listed in config/sources.json -> policy_fetch (D29) and print
the sentences by topic. One request per page, logged, robots-checked.

    python scripts/fetch_policy_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.orchestrator import make_http  # noqa: E402
from src.collect.policy_pages import by_topic, fetch_pages, load_policy_config  # noqa: E402
from src.common.config import load_collect_settings  # noqa: E402


def main() -> int:
    cfg = load_collect_settings()
    pcfg = load_policy_config()
    http = make_http(cfg)
    http.max_retries = 0
    pages = fetch_pages(http, pcfg)
    print(f"network requests made: {http.requests_made} (log: data/manifests/request_log.jsonl)")
    for url, p in pages.items():
        print(f"\n##### {url}\n  status={p['status']} chars={p['chars']} "
              f"from_cache={p.get('from_cache')} {p.get('error', '')}")
        if not p["text"]:
            continue
        for topic, sents in by_topic(p["text"]).items():
            print(f"  --- {topic} ({len(sents)} sentences)")
            for s in sents[:14]:
                print("   *", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
