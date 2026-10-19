from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.collect.ensure import PlaceIngestor
from src.collect.store import PlaceStore
from src.collect.types import Budget, SourceResult
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import Registry
from src.places.resolve import Resolution

DAY = 86400.0


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def entry(name="Munnar"):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="KL-Idukki",
                              popularity="popular", lat=10.0, lon=77.0)


class FakeCollector:
    def __init__(self, source, clock, counter, cost=1, result=None, tick=0.0):
        self.source, self.clock, self.counter = source, clock, counter
        self.cost, self.result, self.tick, self.calls = cost, result, tick, 0

    def collect(self, entry, budget, now, raw_root):
        self.calls += 1
        self.counter[0] += self.cost
        self.clock.t += self.tick
        return self.result or SourceResult(n_raw=3, n_requests=self.cost)


def make(tmp_path, resolution=None, sources=("wikivoyage", "reddit"), **kw):
    clock, counter = FakeClock(), [0]
    cols = [FakeCollector(s, clock, counter, **kw.get(s, {})) for s in sources]
    res = resolution or Resolution("resolved", entry())
    ing = PlaceIngestor(PlaceStore(tmp_path / "live"), Registry(), cols, lambda t, regs: res,
                        clock=clock, request_counter=lambda: counter[0])
    return ing, clock, cols, counter


def test_new_place_is_collected_then_a_repeat_makes_no_network_calls(tmp_path):
    ing, clock, cols, _ = make(tmp_path)
    first = ing.ensure_place("munnar")
    assert first.status == "collected" and first.network_requests == 2
    assert first.chunk_counts is None and first.state == "KL"
    second = ing.ensure_place("munnar")
    assert second.status == "collected" and second.network_requests == 0
    assert [c.calls for c in cols] == [1, 1]


def test_only_expired_sources_are_refreshed(tmp_path):
    ing, clock, cols, _ = make(tmp_path)
    ing.ensure_place("munnar")
    clock.t += 8 * DAY  # reddit TTL is 7 days, wikivoyage 30
    again = ing.ensure_place("munnar")
    assert [c.calls for c in cols] == [1, 2]
    assert again.status == "collected" and again.network_requests == 1


def test_expired_source_with_no_budget_left_is_reported_stale(tmp_path):
    ing, clock, cols, _ = make(tmp_path)
    ing.ensure_place("munnar")
    clock.t += 8 * DAY
    st = ing.ensure_place("munnar", Budget(deadline=clock.t - 1))
    assert st.status == "stale" and st.network_requests == 0 and "reddit" in st.reason


def test_deadline_mid_run_returns_ingesting_and_a_later_call_resumes(tmp_path):
    ing, clock, cols, _ = make(tmp_path, wikivoyage={"tick": 5.0})
    st = ing.ensure_place("munnar", Budget(deadline=clock.t + 3))
    assert st.status == "ingesting" and "reddit" in st.reason
    assert [c.calls for c in cols] == [1, 0]
    done = ing.ensure_place("munnar")
    assert done.status == "collected" and [c.calls for c in cols] == [1, 1]


def test_expired_deadline_before_anything_runs_is_new(tmp_path):
    ing, clock, cols, _ = make(tmp_path)
    st = ing.ensure_place("munnar", Budget(deadline=clock.t - 1))
    assert st.status == "new" and all(c.calls == 0 for c in cols)


def test_incomplete_source_is_resumed(tmp_path):
    partial = SourceResult(n_raw=1, n_requests=1, complete=False)
    ing, clock, cols, _ = make(tmp_path, sources=("reddit",), reddit={"result": partial})
    assert ing.ensure_place("munnar").status == "ingesting"
    cols[0].result = None
    assert ing.ensure_place("munnar").status == "collected" and cols[0].calls == 2


def test_error_is_recorded_not_retried_for_an_hour_and_skipped_is_not_an_error(tmp_path):
    err = SourceResult(error="HTTP 500")
    skip = SourceResult(skipped="no credentials")
    ing, clock, cols, _ = make(tmp_path, reddit={"result": skip}, wikivoyage={"result": err})
    st = ing.ensure_place("munnar")
    assert st.status == "collected"
    assert st.sources["wikivoyage"]["error"] == "HTTP 500"
    assert st.sources["reddit"]["skipped"] == "no credentials"
    ing.ensure_place("munnar")
    assert cols[0].calls == 1  # error not retried within the hour
    assert cols[1].calls == 2  # skipped sources are looked at again (no network)
    clock.t += 3601
    ing.ensure_place("munnar")
    assert cols[0].calls == 2


def test_rejected_and_ambiguous_ingest_nothing(tmp_path):
    for res in (Resolution("rejected", reason="outside scope"),
                Resolution("ambiguous", candidates=[{"name": "A"}, {"name": "B"}])):
        ing, _, cols, _ = make(tmp_path, resolution=res)
        st = ing.ensure_place("x")
        assert st.status == "rejected" and st.place is None
        assert all(c.calls == 0 for c in cols)
    assert st.candidates == [{"name": "A"}, {"name": "B"}]
    assert not (tmp_path / "live" / "places").exists()


def test_new_place_is_added_to_the_live_registry_only(tmp_path):
    ing, *_ = make(tmp_path, resolution=Resolution("resolved", entry("Vagamon"), new=True))
    ing.ensure_place("vagamon")
    assert PlaceStore(tmp_path / "live").load_registry().get("Vagamon") is not None


# ---- merged from tests/test_ensure_demo.py: 

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "ensure_place_demo", ROOT / "scripts" / "ensure_place_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_store_must_not_overlap_the_snapshot_raw_root(tmp_path):
    mod = load()
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    mod.assert_live_only(PlaceStore(tmp_path / "data" / "live"), raw)  # fine
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(raw), raw)  # the snapshot itself
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(raw / "live"), raw)  # inside the snapshot
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(tmp_path / "data"), raw)  # a parent of the snapshot


def test_default_request_cap_is_25_and_stays_a_hard_cap(monkeypatch):
    mod = load()
    assert mod.DEFAULT_CAP == 25
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "--place", "Yercaud"])
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=mod.DEFAULT_CAP)
    assert ap.parse_args([]).cap == 25
    src = (ROOT / "scripts" / "ensure_place_demo.py").read_text("utf-8")
    assert "default=DEFAULT_CAP" in src and "max_retries = 0" in src  # wired in, no retry


def test_shipped_live_root_is_outside_data_raw():
    mod = load()
    mod.assert_live_only(PlaceStore(mod.LIVE_ROOT))
    assert mod.LIVE_ROOT.name == "live"
