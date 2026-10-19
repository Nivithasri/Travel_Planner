from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sources.json"
PLAIN_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")

# (host, path_prefix, min_delay_s, max_requests_per_run, robots_exempt): the three original
# entries (D37) plus the five D43 entries. Any change to this list is a reviewed config change.
GOLDEN = [
    ("www.wikidata.org", "/wiki/Special:EntityData/", 1.0, 30, True),
    ("overpass-api.de", "/api/interpreter", 30.0, 30, True),
    ("archive-api.open-meteo.com", "/v1/archive", 2.0, 30, True),
    ("en.wikipedia.org", "/w/api.php", 1.0, 30, True),
    ("en.wikivoyage.org", "/w/api.php", 1.0, 30, True),
    ("www.wikidata.org", "/w/api.php", 1.0, 30, True),
    ("query.wikidata.org", "/sparql", 1.0, 30, True),
    ("wikimedia.org", "/api/rest_v1/metrics/pageviews/per-article/", 1.0, 30, True),
]


def entries():
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    return doc["api_hosts"]["hosts"]


def test_api_hosts_match_the_golden_list_exactly():
    got = [(h["host"], h["path_prefix"], h["min_delay_s"], h["max_requests_per_run"],
            h["robots_exempt"]) for h in entries()]
    assert got == GOLDEN  # fails on any entry added, removed, reordered or changed
    assert len(got) == 8


def test_every_api_host_value_is_a_plain_hostname():
    for h in entries():
        host = h["host"]
        assert isinstance(host, str) and PLAIN_HOST.fullmatch(host), host
        for bad in ("/", "[", "]", "(", ")", ":", "@", " ", "http"):
            assert bad not in host, (host, bad)
