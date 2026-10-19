from __future__ import annotations

import json

import pytest

from src.collect.raw import RawRecord, RawStore, make_record
from src.collect.types import Budget
from src.collect.wiki import WikiCollector
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry

UA = "Test/0.1 (x@example.org)"


def entry(**kw):
    base = dict(name="Munnar", aliases=[], state="KL", region_cluster="KL-Idukki",
                popularity="popular", lat=10.0, lon=77.0, wikipedia_title="Munnar",
                wikivoyage_title="Munnar")
    return PlaceRegistryEntry(**{**base, **kw})


def api(page):
    return TransportResponse(200, {}, json.dumps({"query": {"pages": [page]}}))


def client(tmp_path, response):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        return response

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0)
    return http, calls


GOOD = {"title": "Munnar", "extract": "Munnar is a hill station.\n\n== See ==\nTea museum.",
        "revisions": [{"revid": 7, "timestamp": "2026-05-04T10:00:00Z"}]}


def test_collects_one_record_with_lastmod_not_publish_date(tmp_path):
    http, calls = client(tmp_path, api(GOOD))
    col = WikiCollector("wikipedia", http, tmp_path / "m.jsonl")
    r = col.collect(entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    assert (r.n_raw, r.error, r.n_requests) == (1, None, 1)
    [rec] = RawStore(tmp_path / "raw").read("wikipedia", "Munnar")
    RawRecord(**rec)
    assert rec["publish_date"] is None and rec["lastmod"] == "2026-05-04"
    assert rec["url"] == "https://en.wikipedia.org/wiki/Munnar" and rec["extra"]["revid"] == 7
    assert json.loads((tmp_path / "m.jsonl").read_text("utf-8"))["sha256"] == rec["content_sha256"]
    assert calls and all("wikipedia.org" in c for c in calls)


def test_same_content_twice_is_not_duplicated(tmp_path):
    http, _ = client(tmp_path, api(GOOD))
    col = WikiCollector("wikivoyage", http)
    col.collect(entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    r2 = col.collect(entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    assert r2.n_raw == 0 and len(RawStore(tmp_path / "raw").read("wikivoyage", "Munnar")) == 1


def test_no_article_is_skipped_without_network(tmp_path):
    http, calls = client(tmp_path, api(GOOD))
    r = WikiCollector("wikivoyage", http).collect(
        entry(wikivoyage_title=None), Budget(), lambda: 0, tmp_path / "raw")
    assert r.skipped and calls == [] and http.requests_made == 0


def test_missing_page_is_an_error_not_an_empty_record(tmp_path):
    http, _ = client(tmp_path, api({"title": "X", "missing": True}))
    r = WikiCollector("wikipedia", http).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error and r.n_raw == 0
    assert RawStore(tmp_path / "raw").read("wikipedia", "Munnar") == []


def test_http_error_is_reported(tmp_path):
    http, _ = client(tmp_path, TransportResponse(404, {}, ""))
    http.max_retries = 0
    r = WikiCollector("wikipedia", http).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error == "HTTP 404 from wikipedia" and r.n_raw == 0


def test_record_validation_rejects_bad_dates_empty_body_and_unknown_source():
    with pytest.raises(ValueError):
        make_record("wikipedia", "P", "u", "text", 0.0, publish_date="12/05/2024")
    with pytest.raises(ValueError):
        make_record("wikipedia", "P", "u", "", 0.0)
    with pytest.raises(ValueError):
        make_record("twitter", "P", "u", "t", 0.0)


@pytest.mark.parametrize("status", [403, 429])
def test_block_status_aborts_the_run(tmp_path, status):
    from src.collect.types import BlockDetected
    http, calls = client(tmp_path, TransportResponse(status, {}, ""))
    http.max_retries = 0
    with pytest.raises(BlockDetected):
        WikiCollector("wikipedia", http).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert len(calls) == 1  # no retry
