"""Fetch robots.txt for allowlisted API hosts (one request each, via HttpClient) and print the
RFC 9309 verdict (src/facts/robots.py) for a sample URL on each host. Evidence for D36.

    python scripts/check_robots.py            (plan only)
    python scripts/check_robots.py --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.orchestrator import load_run_config, make_http  # noqa: E402
from src.common.config import load_collect_settings  # noqa: E402
from src.facts import robots  # noqa: E402
from src.facts.hosts import load_api_hosts  # noqa: E402

CHECKS = {
    "overpass-api.de": "https://overpass-api.de/api/interpreter?data=%5Bout%3Ajson%5D",
    "archive-api.open-meteo.com": ("https://archive-api.open-meteo.com/v1/archive?latitude=11.4"
                                   "&longitude=76.7&start_date=2015-01-01&end_date=2024-12-31"),
}


def report(text: str, ua: str, url: str) -> list[str]:
    v = robots.check(text, ua, url)
    out = [f"  url: {url}", f"  group used: {v.group!r}"]
    for r in v.matches:
        out.append(f"  matching rule: {r.directive.capitalize()}: {r.pattern}  (line {r.line})")
    if not v.matches:
        out.append("  matching rules: none (no rule matches: allowed)")
    out.append(f"  verdict: {'ALLOWED' if v.allowed else 'DISALLOWED'}"
               + (f" by {v.rule.directive.capitalize()}: {v.rule.pattern} (line {v.rule.line})"
                  if v.rule else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    cfg = load_collect_settings()
    hosts = load_api_hosts()
    for h in CHECKS:
        if h not in hosts:
            print(f"{h} is not in api_hosts (D37)")
            return 2
    ua = cfg.user_agent
    print("User-Agent:", ua.replace(cfg.contact_email, "<CONTACT_EMAIL masked>"))
    print(f"planned requests: {len(CHECKS)} (one robots.txt per host)")
    if not args.execute:
        print("(plan only)")
        return 0
    http = make_http(cfg, max_requests=len(CHECKS), run_config=load_run_config())
    http.max_retries = 0
    http.request_log.source = "robots_check"
    for host, url in CHECKS.items():
        res = http.fetch(f"https://{host}/robots.txt", check_robots=False)
        print(f"\n{host}: robots.txt HTTP {res.status}, {len(res.text)} bytes,"
              f" from_cache={res.from_cache}")
        if res.status != 200:
            print("  no usable robots.txt (status above); no verdict")
            continue
        for line in report(res.text, ua, url):
            print(line)
        lines = res.text.splitlines()
        print("  file lines (first 12):", [x for x in lines[:12]])
    print("\nnetwork requests:", http.requests_made, "| per host:", http.requests_by_host())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
