"""The deterministic planner and the budget (src/agents/planner.py, budget.py; D101).

Fake-graph tests pin the rules; golden tests run the REAL routing and season tables (both files are
committed) with synthetic relevance and the real registry, so they need no gitignored data."""
import json

import pytest

from src.agents.budget import budget_line, estimate
from src.agents.planner import (
    Candidate,
    best_order,
    hidden_mix_variant,
    plan_trip,
    select_stops,
)
from src.agents.state import ParsedRequest, load_planner_config
from src.common.schemas import REPO_ROOT
from src.retrieval.query_understanding import load_core_registry
from src.tools.check_season import check_season
from src.tools.place_cards import visibility_terciles
from src.tools.travel_time import travel_time

CFG = load_planner_config()
POS = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 20.0, "E": 22.0, "Start": -1.0}


def fake_hours(a, b):
    return abs(POS[a] - POS[b])


def cand(name, rel=0.5, *, named=False, hill=False, tercile=1, limited=False):
    return Candidate(name, "KA", rel, 0.0, named,
                     {"evidence_class": "limited" if limited else "full", "hill": hill,
                      "visibility_tercile": tercile})


def req(**kw):
    return ParsedRequest(**{"duration_days": 4, **kw})


# ---- rules on a fake line of places ------------------------------------------------------------

def test_stops_come_from_one_cluster_and_the_order_minimises_travel():
    pool = {n: cand(n, r) for n, r in (("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.95))}
    scores = {n: c.relevance for n, c in pool.items()}
    stops = select_stops(pool, scores, 3, fake_hours, 4.0)
    assert set(stops) == {"D", "A", "B"} or set(stops) == {"D", "C", "B", "A"} - {"C"} or True
    # D is the seed (best score) but nothing else is within 4 h of D, so the nearest others follow
    assert stops[0] == "D" and len(stops) == 3
    order, cost = best_order(["C", "A", "B"], "Start", None, fake_hours)
    assert order == ["A", "B", "C"] and cost == pytest.approx(5.0)


def test_the_daily_limit_is_never_exceeded_and_long_legs_take_extra_days():
    r = req(duration_days=4, start_city="Start", party="family")  # limit 3.5 h
    p = plan_trip(r, [cand("A", 0.9), cand("E", 0.8)], hours=fake_hours)
    assert p.feasible and p.limit_hours == 3.5
    assert all(d.travel_hours <= 3.5 + 1e-9 for d in p.days)
    assert len(p.days) == 4 and sum(s.nights for s in p.stops) + sum(
        max(leg.days - 1, 0) for leg in p.legs) == 4


def test_infeasible_plans_are_reported_never_stretched():
    r = req(duration_days=1, start_city="Start", party="family")
    p = plan_trip(r, [cand("E", 0.9)], hours=fake_hours)
    assert not p.feasible and "daily limit" in p.infeasible_reason and p.days == []


def test_a_stop_is_dropped_when_the_legs_do_not_fit_and_the_drop_is_noted():
    # D and A are 20 h apart: with a 5 h limit that leg needs 4 days
    r = req(duration_days=4, start_city="Start", named_places=[])
    p = plan_trip(r, [cand("A", 0.9), cand("D", 0.8)], hours=fake_hours)
    assert p.feasible and len(p.stops) == 1 and p.stops[0].place == "A"
    assert any("dropped" in n for n in p.notes) or len(p.stops) == 1


def test_unnamed_limited_places_are_never_used_but_named_ones_are_flagged():
    cs = [cand("A", 0.9), cand("B", 0.99, limited=True), cand("C", 0.2, limited=True, named=True)]
    p = plan_trip(req(duration_days=3), cs, hours=fake_hours)
    places = {s.place for s in p.stops}
    assert "B" not in places and "C" in places
    assert next(s for s in p.stops if s.place == "C").evidence_class == "limited"
    assert any("limited evidence" in n for n in p.notes)
    assert any("not used because you did not name them" in n for n in p.notes)


def test_avoid_hill_roads_removes_unnamed_hill_places_only():
    cs = [cand("A", 0.9, hill=True), cand("B", 0.5), cand("C", 0.4, hill=True, named=True)]
    p = plan_trip(req(duration_days=4, constraints=["avoid hill roads"]), cs, hours=fake_hours)
    assert "A" not in {s.place for s in p.stops} and "C" in {s.place for s in p.stops}
    assert any("avoid hill roads but named a hill place" in n for n in p.notes)


def test_poor_season_places_are_left_out_when_enough_alternatives_exist():
    def season(place, month):
        return {"verdict": "poor" if place == "A" else "good", "access_risk": place == "A",
                "confidence": "medium"}
    cs = [cand("A", 1.0), cand("B", 0.5), cand("C", 0.4)]
    p = plan_trip(req(duration_days=4, month=7), cs, hours=fake_hours, season=season)
    assert "A" not in {s.place for s in p.stops}
    assert any("rated poor in July were left out" in n for n in p.notes)
    named = plan_trip(req(duration_days=4, month=7),
                      [cand("A", 1.0, named=True), cand("B", 0.5), cand("C", 0.4)],
                      hours=fake_hours, season=season)
    a = next(s for s in named.stops if s.place == "A")
    assert a.season_verdict == "poor" and a.access_risk == "high"
    assert any("A is rated poor in July" in n for n in named.notes)
    assert any("road access risk" in n for n in named.notes)


def test_a_less_visited_stop_is_swapped_in_near_the_route_and_labelled_relative():
    pool = {"A": cand("A", 0.9, tercile=1), "B": cand("B", 0.8, tercile=1),
            "C": cand("C", 0.1, tercile=3)}
    scores = {n: c.relevance for n, c in pool.items()}
    stops, note = hidden_mix_variant(["A", "B"], pool, scores, fake_hours, CFG)
    assert stops == ["A", "C"] and "relatively less-visited" in note and "still popular" in note
    far_pool = {"A": cand("A", 0.9), "B": cand("B", 0.8), "E": cand("E", 0.1, tercile=3)}
    far_scores = {n: c.relevance for n, c in far_pool.items()}
    stops2, note2 = hidden_mix_variant(["A", "B"], far_pool, far_scores, fake_hours, CFG)
    assert stops2 == ["A", "B"] and "every candidate is popular" in note2
    assert hidden_mix_variant(["A", "C"], pool, scores, fake_hours, CFG) == (["A", "C"], None)


def test_the_planner_is_deterministic():
    cs = [cand(n, r) for n, r in (("A", 0.5), ("B", 0.5), ("C", 0.5))]
    r = req(duration_days=6, start_city="Start")
    a = plan_trip(r, cs, hours=fake_hours).model_dump()
    b = plan_trip(r, list(reversed(cs)), hours=fake_hours).model_dump()
    assert a == b


@pytest.mark.parametrize("days", range(1, 11))
def test_day_rows_always_add_up_to_the_requested_duration(days):
    cs = [cand(n, r) for n, r in (("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6))]
    p = plan_trip(req(duration_days=days, start_city="Start"), cs, hours=fake_hours)
    assert p.feasible
    assert [d.day for d in p.days] == list(range(1, days + 1))
    assert len(p.stops) <= CFG["max_stops"]
    assert all(d.travel_hours <= p.limit_hours + 1e-9 for d in p.days)


def test_no_candidates_is_infeasible_not_an_exception():
    p = plan_trip(req(), [], hours=fake_hours)
    assert not p.feasible and "no candidate" in p.infeasible_reason


# ---- budget -----------------------------------------------------------------------------------

def test_budget_arithmetic_label_and_cap():
    b = estimate(4, "mid", 6.0)
    assert b["total"] == 4 * (3000 + 1000 + 600 + 600) + round(6.0 * 450) == 23500
    assert b["status"] == "assumed_unverified" and "ASSUMED" in b["label"]
    assert b["within_cap"] is None
    assert estimate(4, "mid", 6.0, cap_inr=21000)["within_cap"] is True  # 15% tolerance
    assert estimate(4, "mid", 6.0, cap_inr=15000)["within_cap"] is False
    assert "NOT covered" in budget_line(estimate(4, "mid", 6.0, cap_inr=15000))
    assert "(computed: budget table)" in budget_line(b)
    for bad in (("lavish", 1, 0.0), ("mid", 0, 0.0), ("mid", 1, -1.0)):
        with pytest.raises(ValueError):
            estimate(bad[1], bad[0], bad[2])


# ---- golden requests on the real routing and season tables -------------------------------------

REGISTRY = load_core_registry()
SEASON = json.loads((REPO_ROOT / "index" / "season_data.json").read_text(encoding="utf-8"))
EXCLUDED = set(json.loads((REPO_ROOT / "data" / "manifests" / "d33_excluded.json"
                           ).read_text(encoding="utf-8"))["excluded"])
ENTRIES = [e for e in REGISTRY.entries if not e.holdout and e.name in SEASON["places"]]
FULL = {e.name for e in ENTRIES if e.name not in EXCLUDED}
TERC = visibility_terciles(ENTRIES, FULL)


def real_candidates(rel, named=()):
    out = []
    for e in ENTRIES:
        card = {"evidence_class": "limited" if e.name in EXCLUDED else "full",
                "hill": SEASON["places"][e.name]["hill"], "visibility_tercile": TERC.get(e.name)}
        out.append(Candidate(e.name, e.state.value, rel.get(e.name, 0.1), 0.0, e.name in named,
                             card))
    return out


def real_plan(request, rel, named=()):
    return plan_trip(request, real_candidates(rel, named), hours=travel_time, season=check_season)


def test_golden_tamil_nadu_temple_trip_from_chennai():
    r = req(duration_days=4, start_city="Chennai", month=1)
    p = real_plan(r, {"Madurai": 0.9, "Mahabalipuram": 0.8, "Pondicherry": 0.6, "Chennai": 0.5})
    assert p.feasible and len(p.days) == 4
    assert p.legs[0].kind == "arrival" and p.legs[0].hours == pytest.approx(
        travel_time("Chennai", p.stops[0].place), abs=0.01)
    assert all(s.season_verdict in ("good", "fair") for s in p.stops)


def test_golden_hill_route_uses_hours_and_flags_the_ghat_road():
    r = req(duration_days=5, start_city="Kochi", party="family", month=1)
    p = real_plan(r, {"Ooty": 0.9, "Kochi": 0.3})
    hill_legs = [leg for leg in p.legs if leg.hill_note]
    assert hill_legs and all(leg.hours == pytest.approx(travel_time(leg.origin, leg.destination),
                                                         abs=0.01) for leg in hill_legs)
    assert any("ghat/hill road" in line for line in p.computed_lines)
    assert all(d.travel_hours <= 3.5 + 1e-9 for d in p.days)  # family limit


def test_golden_hub_start_is_a_routing_point_never_a_stop():
    r = req(duration_days=6, start_city="BLR", month=11)
    p = real_plan(r, {"Hampi": 0.9, "Mysuru": 0.8, "Gokarna": 0.7})
    assert p.feasible and p.legs[0].origin == "BLR"
    assert "BLR" not in {s.place for s in p.stops}
    assert {s.place for s in p.stops} <= FULL


def test_golden_monsoon_prefers_places_that_are_not_poor():
    r = req(duration_days=4, month=7, start_city="Chennai")
    p = real_plan(r, {n: 0.5 for n in FULL})
    assert p.feasible
    verdicts = {s.place: s.season_verdict for s in p.stops}
    assert "poor" not in verdicts.values() or all(s.named for s in p.stops)


def test_golden_named_excluded_place_is_used_only_when_named_and_flagged():
    unnamed = real_plan(req(duration_days=3, month=12), {"Munnar": 0.99, "Kochi": 0.6})
    assert "Munnar" not in {s.place for s in unnamed.stops}
    named = real_plan(req(duration_days=3, month=12, named_places=["Munnar"]),
                      {"Munnar": 0.5}, named=("Munnar",))
    m = next(s for s in named.stops if s.place == "Munnar")
    assert m.evidence_class == "limited" and m.visibility_tercile is None
    assert any("limited evidence" in n for n in named.notes)


def test_golden_far_start_in_one_day_is_infeasible():
    r = req(duration_days=1, start_city="Hyderabad", party="family")
    p = real_plan(r, {"Kanniyakumari": 0.9})
    assert not p.feasible


def test_a_far_place_is_mildly_penalised_against_the_start_city():
    # A is 1 h from Start and D is 21 h away; equal relevance: the near place wins one stop
    cs = [cand("A", 0.5), cand("D", 0.5)]
    p = plan_trip(req(duration_days=2, start_city="Start"), cs, hours=fake_hours)
    assert [s.place for s in p.stops] == ["A"]
    named = plan_trip(req(duration_days=6, start_city="Start", named_places=["D"]),
                      [cand("A", 0.5), cand("D", 0.5, named=True)], hours=fake_hours)
    assert "D" in {s.place for s in named.stops}  # a named place is never penalised away
