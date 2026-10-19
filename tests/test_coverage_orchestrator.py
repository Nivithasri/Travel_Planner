from __future__ import annotations

import json

import pytest

from src.collect.coverage import passes_rule, place_coverage, render
from src.collect.orchestrator import HoldoutError, build_collectors, core_only_resolver
from src.collect.raw import RawStore, make_record
from src.common.config import CollectSettings
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import Registry


def entry(name, holdout=None):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0, holdout=holdout)


def put(root, source, place, n=1):
    for i in range(n):
        RawStore(root).write(make_record(source, place, f"https://x/{source}/{i}", "text", 0.0))


def test_rule_needs_two_types_and_one_outside_the_reference_family():
    assert not passes_rule({"wikivoyage", "reference"})  # two types, both reference family
    assert not passes_rule({"reddit"})
    assert passes_rule({"wikivoyage", "reddit"})
    assert passes_rule({"reference", "official"})
    assert not passes_rule(set())


def test_place_coverage_maps_raw_sources_to_types(tmp_path):
    put(tmp_path, "wikipedia", "Munnar")
    put(tmp_path, "sahapedia", "Munnar")
    c = place_coverage(entry("Munnar"), tmp_path)
    assert c.types == ["reference"] and not c.passes and c.reason == "only 1 source type(s)"
    put(tmp_path, "wikivoyage", "Munnar")
    two = place_coverage(entry("Munnar"), tmp_path)
    assert not two.passes and "reference-wiki" in two.reason
    put(tmp_path, "youtube", "Munnar")
    assert place_coverage(entry("Munnar"), tmp_path).passes


def test_holdout_is_listed_but_never_counted(tmp_path):
    put(tmp_path, "reddit", "Thekkady")
    put(tmp_path, "youtube", "Thekkady")
    text, data = render([entry("Munnar"), entry("Thekkady", holdout=True)], tmp_path,
                        tmp_path / "status", {"reddit": "skipped: no credentials"})
    assert data["counted"] == 1 and data["passing"] == 0
    assert "0 of 1" in text and "hold-out" in text
    assert "**reddit**: skipped: no credentials" in text


def test_skipped_collectors_come_from_status_files_too(tmp_path):
    status = tmp_path / "status" / "places"
    status.mkdir(parents=True)
    (status / "munnar.json").write_text(json.dumps(
        {"sources": {"youtube": {"skipped": "no YOUTUBE_API_KEY in .env"},
                     "web": {"error": "sahapedia.org: no sitemap"}}}))
    text, _ = render([entry("Munnar")], tmp_path, tmp_path / "status")
    assert "skipped: no YOUTUBE_API_KEY" in text and "error: sahapedia.org" in text


def test_snapshot_resolver_refuses_holdouts_and_unknowns():
    reg = Registry([entry("Munnar"), entry("Thekkady", holdout=True)])
    assert core_only_resolver("munnar", [reg]).status == "resolved"
    assert core_only_resolver("Atlantis", [reg]).status == "not_found"
    with pytest.raises(HoldoutError):
        core_only_resolver("Thekkady", [reg])


def test_build_collectors_in_fixed_order_and_unknown_source_is_refused(tmp_path):
    cfg = CollectSettings("a@example.org", "UA a@example.org", tmp_path)
    from src.collect.orchestrator import make_http
    cols = build_collectors(cfg, make_http(cfg), None, ["youtube", "wikivoyage"])
    assert [c.source for c in cols] == ["wikivoyage", "youtube"]
    with pytest.raises(ValueError):
        build_collectors(cfg, make_http(cfg), None, ["twitter"])


def test_nothing_collected_is_reported_as_not_collected_yet(tmp_path):
    assert place_coverage(entry("Munnar"), tmp_path).reason == "nothing collected yet"
