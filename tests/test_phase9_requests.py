"""The Phase 9 request set: structure, scope rules and the approval gate (D123)."""
import copy
import json
from types import SimpleNamespace

import pytest

from eval.phase9 import request_set as R

ENTRIES = [
    SimpleNamespace(name="Hampi", aliases=[], holdout=False),
    SimpleNamespace(name="Mysuru", aliases=["Mysore"], holdout=False),
    SimpleNamespace(name="Munnar", aliases=[], holdout=False),
    SimpleNamespace(name="Thekkady", aliases=[], holdout=True),
]
EXCLUDED = {"Munnar"}


def req(i, *, ts=False, named=("Hampi",), text="Two days in Hampi.", **over):
    o = {"duration_days": 3, "budget_tier": "mid", "party": "couple", "month": 12,
         "start_city": "Bengaluru", "interests": ["history"]}
    if ts:
        o["time_sensitive"] = True
    o.update(over)
    return {"id": f"P{i:02d}", "type": "x", "time_sensitive": ts, "text": text,
            "named_places": list(named), "overrides": o}


def doc(n=10, n_ts=4):
    reqs = [req(i, ts=i <= n_ts) for i in range(1, n + 1)]
    return {"meta": {"status": "DRAFT", "time_sensitive_min": 4}, "requests": reqs}


def problems(d, **kw):
    return R.validate_requests(d, registry_entries=ENTRIES, excluded=EXCLUDED, **kw)


def test_a_valid_document_has_no_problems():
    assert problems(doc()) == []


def test_count_ids_and_time_sensitive_minimum_are_checked():
    assert any("expected 10" in p for p in problems(doc(n=9)))
    d = doc()
    d["requests"][1]["id"] = "P01"
    assert any("not unique" in p for p in problems(d))
    assert any("at least 4 time-sensitive" in p for p in problems(doc(n_ts=3)))


def test_excluded_and_hold_out_places_are_refused_in_text_and_in_named_places():
    d = doc()
    d["requests"][4]["text"] = "A day near Munnar."
    assert any("hold-out or D33-excluded" in p for p in problems(d))
    d = doc()
    d["requests"][4]["text"] = "Wildlife at Thekkady."
    assert any("hold-out or D33-excluded" in p for p in problems(d))
    d = doc()
    d["requests"][4]["named_places"] = ["Munnar"]
    assert any("D33-excluded" in p for p in problems(d))
    d = doc()
    d["requests"][4]["named_places"] = ["Atlantis"]
    assert any("not a registry place" in p for p in problems(d))
    d = doc()
    d["requests"][4]["named_places"] = ["Mysore"]  # an alias, not the registry name
    assert any("must be the registry name 'Mysuru'" in p for p in problems(d))


def test_time_sensitive_requests_must_name_a_place_and_carry_the_override():
    d = doc()
    d["requests"][0]["named_places"] = []
    assert any("must name a place" in p for p in problems(d))
    d = doc()
    del d["requests"][0]["overrides"]["time_sensitive"]
    assert any("needs overrides.time_sensitive" in p for p in problems(d))


def test_override_fields_are_validated():
    cases = [({"duration_days": 0}, "duration_days"), ({"budget_tier": "cheap"}, "budget_tier"),
             ({"party": "crowd"}, "party"), ({"month": 13}, "month"),
             ({"interests": ["festivals"]}, "closed list"), ({"colour": "red"}, "unknown override")]
    for over, needle in cases:
        d = doc()
        d["requests"][5]["overrides"].update(over)
        assert any(needle in p for p in problems(d)), over
    d = doc()
    assert any("routing table" in p for p in problems(d, city_ok=lambda c: False))


def test_a_missing_field_is_reported_not_raised():
    d = doc()
    del d["requests"][2]["text"]
    assert any("missing field" in p for p in problems(d))


def test_the_approval_gate(tmp_path):
    p = tmp_path / "requests.json"
    d = doc()
    p.write_text(json.dumps(d), encoding="utf-8")
    assert len(R.load_requests(p)) == 10
    with pytest.raises(R.NotApproved):
        R.load_requests(p, require_approved=True)
    approved = copy.deepcopy(d)
    approved["meta"]["status"] = R.APPROVED
    p.write_text(json.dumps(approved), encoding="utf-8")
    assert len(R.load_requests(p, require_approved=True)) == 10


def test_the_committed_request_file_is_valid_against_the_real_registry():
    from src.retrieval.query_understanding import load_core_registry

    excluded = json.loads(R.REPO_ROOT.joinpath("data", "manifests", "d33_excluded.json")
                          .read_text(encoding="utf-8"))["excluded"]
    d = R.load_document()
    assert R.validate_requests(d, registry_entries=load_core_registry().entries,
                               excluded=excluded) == []
    assert len(d["requests"]) == 10 and d["meta"]["status"] in ("DRAFT", R.APPROVED)
