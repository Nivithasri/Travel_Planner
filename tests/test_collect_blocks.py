"""A block (HTTP 403/429, transcript block) in ANY source aborts the whole run, even when other
sources succeeded before it, and nothing after the blocked source is tried."""
from __future__ import annotations

import json

import pytest

from src.collect.ensure import PlaceIngestor
from src.collect.raw import RawStore
from src.collect.store import PlaceStore
from src.collect.types import BlockDetected, SourceResult
from src.collect.web import WebCollector
from src.collect.wiki import WikiCollector
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import Registry
from src.places.resolve import Resolution

UA = "Test/0.1 (x@example.org)"
ENTRY = PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=10.0, lon=77.0, wikivoyage_title="Munnar")
WIKI_OK = json.dumps({"query": {"pages": [{"title": "Munnar", "extract": "Munnar text " * 30,
                                           "revisions": [{"revid": 1,
                                                          "timestamp": "2026-01-02T00:00:00Z"}]}]}})
ALLOW = {"example.org": {"domain": "example.org", "source_type": "official",
                         "raw_source": "official", "license": "x", "min_delay_s": 0.0,
                         "max_pages_per_place": 2}}


class Third:
    source = "reddit"
    calls = 0

    def collect(self, entry, budget, now, raw_root):
        Third.calls += 1
        return SourceResult()


def run(tmp_path, sitemap_status):
    def transport(method, url, headers, timeout):
        if "wikivoyage" in url:
            return TransportResponse(200, {}, WIKI_OK)
        if url.endswith("/robots.txt"):
            return TransportResponse(404, {}, "")
        if url.endswith("/sitemap.xml"):
            return TransportResponse(sitemap_status, {}, "")
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    Third.calls = 0
    cols = [WikiCollector("wikivoyage", http), WebCollector(http, ALLOW), Third()]
    store = PlaceStore(tmp_path / "live")
    ing = PlaceIngestor(store, Registry(), cols, lambda t, r: Resolution("resolved", ENTRY),
                        clock=lambda: 1_000_000.0, request_counter=lambda: http.requests_made)
    return ing, store


@pytest.mark.parametrize("status", [403, 429])
def test_block_in_one_source_aborts_the_run_after_another_source_succeeded(tmp_path, status):
    ing, store = run(tmp_path, status)
    with pytest.raises(BlockDetected):
        ing.ensure_place("munnar")
    assert Third.calls == 0  # nothing after the blocked source was tried
    # the earlier success is kept on disk and in the status file
    assert len(RawStore(store.raw_root).read("wikivoyage", "Munnar")) == 1
    assert store.load("Munnar")["sources"]["wikivoyage"]["n_raw"] == 1
    assert "web" not in store.load("Munnar")["sources"]  # the blocked source is not recorded done


def test_without_a_block_the_same_run_completes(tmp_path):
    ing, _ = run(tmp_path, 404)
    st = ing.ensure_place("munnar")
    assert st.status == "collected" and Third.calls == 1


