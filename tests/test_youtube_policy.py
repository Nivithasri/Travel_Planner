"""D27: YouTube transcripts are dropped. The collector must refuse to run them, with zero
requests, and a normal orchestrator run must never touch googleapis.com or youtube.com."""
from __future__ import annotations

import json

from src.collect.ensure import PlaceIngestor
from src.collect.orchestrator import build_collectors, load_run_config, make_http
from src.collect.store import PlaceStore
from src.collect.types import SOURCES, Budget
from src.common.config import REPO_ROOT, CollectSettings
from src.common.http import TransportResponse
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import Registry
from src.places.resolve import Resolution

ENTRY = PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=10.0, lon=77.0,
                           wikipedia_title="Munnar", wikivoyage_title="Munnar")
WIKI = json.dumps({"query": {"pages": [{"title": "Munnar", "extract": "Munnar text " * 40,
                                        "revisions": [{"revid": 1,
                                                       "timestamp": "2026-01-02T00:00:00Z"}]}]}})


class Spy:
    def __init__(self):
        self.urls = []

    def __call__(self, method, url, headers, timeout):
        self.urls.append(url)
        if "api.php" in url:
            return TransportResponse(200, {}, WIKI)
        return TransportResponse(404, {}, "")


def cfg(tmp_path):
    return CollectSettings("someone@example.org", "UA/0.1 (someone@example.org)",
                           tmp_path / "cache", youtube_api_key="FAKE-KEY-FOR-TEST")


def ingestor(tmp_path, only=None):
    spy = Spy()
    c = cfg(tmp_path)
    http = make_http(c, transport=spy, request_log=None)
    cols = build_collectors(c, http, None, only)
    http.min_delay_s = 0  # collectors set per-host delays; no real waiting in tests
    http.host_delays.clear()
    http._sleep = lambda s: None
    store = PlaceStore(tmp_path / "live")
    ing = PlaceIngestor(store, Registry(), cols, lambda t, r: Resolution("resolved", ENTRY),
                        clock=lambda: 1_000_000.0, request_counter=lambda: http.requests_made)
    return ing, spy, store, cols


# ---- the collector itself (strict) --------------------------------------------------------




# ---- config and defaults ------------------------------------------------------------------
def test_shipped_config_has_transcripts_switched_off_and_youtube_is_not_a_default_source():
    doc = json.loads((REPO_ROOT / "config" / "sources.json").read_text("utf-8"))
    assert doc["youtube_transcripts"] is False
    assert load_run_config()["youtube_transcripts"] is False
    assert "youtube" in SOURCES  # still a known name, only off by default


def test_default_source_list_has_no_youtube(tmp_path):
    c = cfg(tmp_path)
    cols = build_collectors(c, make_http(c, transport=Spy(), request_log=None), None)
    assert [x.source for x in cols] == ["wikivoyage", "wikipedia", "web", "reddit"]


# ---- a full orchestrator run (fake transports) --------------------------------------------
def test_full_run_never_invokes_youtube_or_googleapis(tmp_path):
    ing, spy, store, cols = ingestor(tmp_path)
    st = ing.ensure_place("munnar", Budget())
    assert st.status == "collected" and "youtube" not in st.sources
    assert spy.urls, "the run did make (fake) requests"
    assert not [u for u in spy.urls if "googleapis.com" in u or "youtube.com" in u]
    assert not (store.raw_root / "youtube").exists()


def test_explicit_youtube_source_is_reported_as_skipped_d27_not_as_a_failure(tmp_path):
    ing, spy, store, cols = ingestor(tmp_path, only=["wikipedia", "youtube"])
    st = ing.ensure_place("munnar", Budget())
    rec = st.sources["youtube"]
    assert st.status == "collected" and rec["error"] is None
    assert "D27" in rec["skipped"] and rec["n_raw"] == 0 and rec["n_requests"] == 0
    assert not [u for u in spy.urls if "googleapis.com" in u or "youtube.com" in u]


def test_resume_does_not_retry_or_change_a_skipped_or_withdrawn_youtube_record(tmp_path):
    ing, spy, store, cols = ingestor(tmp_path, only=["wikipedia", "youtube"])
    ing.ensure_place("munnar", Budget())
    n = len(spy.urls)
    again = ing.ensure_place("munnar", Budget())
    assert len(spy.urls) == n  # everything cached or skipped: no new network call at all
    assert again.network_requests == 0 and "D27" in again.sources["youtube"]["skipped"]
    # a record withdrawn by scripts/withdraw_source.py is left alone when youtube is not run
    doc = store.load("Munnar")
    doc["sources"]["youtube"] = {"skipped": "withdrawn (D27)", "n_raw": 0, "n_requests": 0,
                                 "error": None, "ingested_at": None, "expires_at": None,
                                 "complete": True}
    store.save("Munnar", doc)
    ing2, _, store2, _ = ingestor(tmp_path)  # default list: no youtube collector
    st = ing2.ensure_place("munnar", Budget())
    assert st.sources["youtube"]["skipped"] == "withdrawn (D27)" and st.status == "collected"
