"""The committed query set (D70) and the exclusion check (D68)."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from eval.qrels import load_queries
from src.places.places import Registry

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_queries", ROOT / "scripts" / "check_queries.py")
cq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cq)

pytestmark = pytest.mark.skipif(not cq.REGISTRY.exists(), reason="registry not present")


@pytest.fixture(scope="module")
def registry():
    return Registry.load(cq.REGISTRY)


@pytest.fixture(scope="module")
def excluded():
    return set(json.loads(cq.EXCLUDED.read_text(encoding="utf-8"))["excluded"])


def q(qid="Q1", text="a query", **kw):
    base = {"id": qid, "text": text, "split": "dev", "type": "family", "time_sensitive": False,
            "freshness_slice": False, "place_scope": "free", "places": []}
    base.update(kw)
    return base


def test_committed_queries_pass_the_exclusion_check(registry, excluded):
    assert cq.check(load_queries(), registry, excluded) == []


def test_committed_counts_match_d70():
    qs = load_queries()
    assert len(qs) == 40
    assert Counter(x["split"] for x in qs) == {"dev": 13, "test": 27}
    assert sum(x["freshness_slice"] for x in qs) == 10
    assert sum(x["freshness_slice"] for x in qs if x["split"] == "dev") == 3
    assert sum(x["time_sensitive"] for x in qs) == 12
    assert {x["id"] for x in qs if x["freshness_slice"]} == {
        "Q16", "Q17", "Q22", "Q24", "Q35", "Q36", "Q37", "Q38", "Q39", "Q40"}
    assert {x["id"] for x in qs if x["split"] == "dev"} == {
        "Q01", "Q03", "Q06", "Q09", "Q11", "Q16", "Q18", "Q21", "Q28", "Q31", "Q33", "Q37", "Q39"}
    assert {"family", "backpacker", "honeymoon", "pilgrimage", "monsoon", "budget"} <= {
        x["type"] for x in qs}
    assert all(x["time_sensitive"] for x in qs if x["freshness_slice"])


@pytest.mark.parametrize("text", [
    "Tea gardens in Munnar", "Backwaters near Alleppey", "Kodaikanal lake boating",
    "Wayanad wildlife trip", "Varkala cliff cafes", "Temples of Rameswaram",
    "Auroville community visit", "Srisailam dam", "Bhadrachalam temple",
])
def test_excluded_places_and_aliases_are_rejected(registry, excluded, text):
    problems = cq.check([q(text=text)], registry, excluded)
    assert any("excluded" in p or "free but" in p for p in problems), problems


@pytest.mark.parametrize("text", [
    "Boating at Thekkady", "Visit Tanjore temples", "Brihadeeswara temple tour",
    "Araku Valley train ride", "Chikmagalur coffee", "Periyar Lake boat", "Borra Caves",
])
def test_holdouts_are_rejected(registry, excluded, text):
    problems = cq.check([q(text=text)], registry, excluded)
    assert any("hold-out" in p for p in problems), problems


def test_named_and_free_consistency(registry, excluded):
    ok = q(text="Cheap eats in Hyderabad", place_scope="named", places=["Hyderabad"])
    assert cq.check([ok], registry, excluded) == []
    alias = q(text="Coorg coffee estates", place_scope="named", places=["Kodagu"])
    assert cq.check([alias], registry, excluded) == []  # an alias counts as naming the place
    missing = q(text="Cheap eats", place_scope="named", places=["Hyderabad"])
    assert any("does not appear" in p for p in cq.check([missing], registry, excluded))
    sneaky = q(text="Cheap eats in Hyderabad")  # free, but names a place
    assert any("free but" in p for p in cq.check([sneaky], registry, excluded))
    unknown = q(text="Trip to Atlantis", place_scope="named", places=["Atlantis"])
    assert any("unknown place" in p for p in cq.check([unknown], registry, excluded))
    listed = q(text="Trip to Munnar", place_scope="named", places=["Munnar"])
    assert any("excluded or hold-out" in p for p in cq.check([listed], registry, excluded))
