from __future__ import annotations

import json

import pytest

from src.collect.policy_pages import by_topic, fetch_pages, load_policy_config, strip_html
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse


def test_shipped_policy_config_is_narrow_and_consistent():
    cfg = load_policy_config()
    assert cfg["hosts"] == ["developers.google.com"] and len(cfg["pages"]) == 3
    assert all(p.startswith("https://developers.google.com/youtube/") for p in cfg["pages"])


def test_pages_outside_the_listed_hosts_are_refused(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"policy_fetch": {"hosts": ["a.org"], "pages": ["https://b.org/x"]}}))
    with pytest.raises(ValueError):
        load_policy_config(p)


def test_sentences_are_sorted_into_topics_and_html_is_stripped():
    text = strip_html("<p>You must refresh stored data every thirty calendar days.</p>"
                      "<script>x()</script>"
                      "<p>We may suspend your API key or project at any time.</p>"
                      "<p>Scraping outside the API is not permitted for any client.</p>")
    t = by_topic(text)
    assert any("calendar days" in s for s in t["storage_retention"])
    assert any("API key" in s for s in t["consequences"])
    assert any("Scraping" in s for s in t["automated_scraping"])


def test_fetch_makes_one_request_per_page_and_records_failures(tmp_path):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        if url.endswith("robots.txt"):
            return TransportResponse(404, {}, "")
        return TransportResponse(200, {}, "<p>Quota cost is 100 units for search.list calls.</p>")

    http = HttpClient(DiskCache(tmp_path / "c"), "UA/0.1 (x@example.org)", transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0)
    cfg = {"hosts": ["h.org"], "pages": ["https://h.org/a", "https://h.org/b"]}
    out = fetch_pages(http, cfg)
    assert [u for u in calls if not u.endswith("robots.txt")] == cfg["pages"]
    assert all(v["status"] == 200 and "units" in v["text"] for v in out.values())
