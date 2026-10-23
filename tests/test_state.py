"""State models, defaults and the live configuration gates (src/agents/state.py, src/live/config.py,
src/common/schemas.py LiveSettings; D100, D101)."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.agents.state import (
    ParsedRequest,
    RunState,
    load_budget_tiers,
    load_planner_config,
    request_from_parse,
)
from src.common.schemas import REPO_ROOT, LiveSettings, load_live_settings
from src.live.config import LiveConfig, load_live_config


def parsed(**kw):
    base = dict(text="a trip", places=[], states=[], month=None, duration_days=None,
                budget_tier=None, travel_style=None, interests=[], place_types=[],
                time_sensitive=False, notes=[])
    base.update(kw)
    return SimpleNamespace(**base)


def test_defaults_are_applied_and_noted():
    req = request_from_parse(parsed())
    assert (req.duration_days, req.budget_tier, req.month, req.party) == (4, "mid", None, "group")
    joined = " | ".join(req.notes)
    assert "duration not given" in joined and "budget tier not given" in joined
    assert "no month given" in joined and "party not given" in joined


def test_parsed_values_and_overrides_win_over_defaults():
    p = parsed(text="temples in Karnataka", duration_days=6, budget_tier="luxury", month=7,
               travel_style="family", places=["Hampi"], states=["KA"], interests=["temples"],
               time_sensitive=True)
    req = request_from_parse(p, overrides={"duration_days": 3, "start_city": "Bengaluru",
                                           "constraints": ["avoid hill roads"]})
    assert (req.duration_days, req.budget_tier, req.month) == (3, "luxury", 7)
    assert (req.party, req.named_places, req.states, req.time_sensitive) == (
        "family", ["Hampi"], ["KA"], True)
    assert req.start_city == "Bengaluru" and req.avoid_hill_roads


def test_validation_rejects_out_of_range_values():
    for bad in ({"duration_days": 0}, {"duration_days": 11}, {"month": 13}, {"month": 0},
                {"max_daily_travel_hours": 0}, {"max_daily_travel_hours": 15},
                {"budget_tier": "lavish"}, {"party": "crowd"}, {"surprise": 1}):
        with pytest.raises((ValidationError, ValueError)):
            ParsedRequest(**bad)


def test_daily_limit_comes_from_the_party_unless_given():
    assert ParsedRequest(party="family").daily_limit_hours() == 3.5
    assert ParsedRequest(party="solo").daily_limit_hours() == 6.0
    assert ParsedRequest(party="family", max_daily_travel_hours=2.5).daily_limit_hours() == 2.5


def test_constraint_words_are_recognised():
    r = ParsedRequest(constraints=["No night travel please", "Avoid Hill roads"])
    assert r.no_night_travel and r.avoid_hill_roads
    assert not ParsedRequest().no_night_travel


def test_the_config_files_are_consistent():
    cfg = load_planner_config()
    assert cfg["min_days"] <= cfg["default_days"] <= cfg["max_days"]
    assert cfg["daily_travel_limit_hours"]["family"] < cfg["daily_travel_limit_hours"]["default"]
    b = load_budget_tiers()
    assert b["status"] == "assumed_unverified"
    assert set(b["per_day"]) == set(b["intercity_per_hour"]) == {"budget", "mid", "luxury"}
    assert (b["per_day"]["budget"]["stay"] < b["per_day"]["mid"]["stay"]
            < b["per_day"]["luxury"]["stay"])


def test_run_state_starts_empty():
    s = RunState()
    assert s.live_status == "not_run" and s.plan is None and s.warnings == []


# ---- live configuration --------------------------------------------------------------------

def test_the_committed_live_config_records_the_owner_risk_acceptance_and_keeps_the_cache_off():
    cfg = load_live_config()
    assert cfg.terms_status == "owner_accepted_risk" and cfg.real_calls_status() == "ok"
    assert cfg.credits_per_search == 2 and cfg.max_credits_per_request == 6
    assert cfg.endpoint_host == "api.tavily.com" and not cfg.cache_enabled
    raw = json.loads((REPO_ROOT / "config" / "live.json").read_text(encoding="utf-8"))
    assert "assumed, not measured" in raw["credits_basis"]
    with pytest.raises(ValidationError):  # the terms are never 'verified' (D109)
        LiveConfig(**{**raw, "terms_status": "verified"})


def test_real_calls_status_follows_enabled_and_terms():
    base = json.loads((REPO_ROOT / "config" / "live.json").read_text(encoding="utf-8"))
    assert LiveConfig(**{**base, "enabled": False, "terms_status": "owner_accepted_risk"}
                      ).real_calls_status() == "disabled"
    assert LiveConfig(**{**base, "terms_status": "owner_accepted_risk"}).real_calls_status() == "ok"
    assert LiveConfig(**{**base, "terms_status": "not_verified"}
                      ).real_calls_status() == "terms_not_verified"


def test_inconsistent_live_budgets_are_refused():
    base = json.loads((REPO_ROOT / "config" / "live.json").read_text(encoding="utf-8"))
    for bad in ({"default_searches_per_request": 4}, {"max_credits_per_request": 5},
                {"monthly_stop_at": 1001}):
        with pytest.raises(ValidationError):
            LiveConfig(**{**base, **bad})


def test_live_key_is_never_in_a_repr(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("LIVE_SEARCH_API_KEY=tvly-secret-value\n", encoding="utf-8")
    monkeypatch.delenv("LIVE_SEARCH_API_KEY", raising=False)
    s = load_live_settings(env)
    assert s.has_key and "secret" not in repr(s)
    monkeypatch.delenv("LIVE_SEARCH_API_KEY", raising=False)
    assert not LiveSettings().has_key


# ---- states are a restriction only when the user named them (D115) -----------------------------

def test_states_of_the_named_places_are_not_a_restriction():
    p = parsed(text="Places: Pondicherry, Mahabalipuram.", places=["Pondicherry", "Mahabalipuram"],
               states=["PY"])
    req = request_from_parse(p, overrides={"text": p.text})
    assert req.states == [] and req.named_places == ["Pondicherry", "Mahabalipuram"]
    assert any("PY came from the places you named" in n for n in req.notes)


def test_a_state_named_in_the_text_is_kept_and_an_inferred_one_is_dropped():
    p = parsed(text="Kerala backwaters and Hampi", places=["Hampi"], states=["KL", "KA"])
    req = request_from_parse(p, overrides={"text": p.text})
    assert req.states == ["KL"]
    assert any("KA came from the places" in n for n in req.notes)


def test_explicit_states_override_the_text_rule():
    p = parsed(text="a trip", states=["KA"])
    req = request_from_parse(p, overrides={"states": ["TN"], "text": p.text})
    assert req.states == ["TN"] and not any("came from" in n for n in req.notes)


def test_states_named_in_matches_whole_state_names_only():
    from src.agents.state import states_named_in

    assert states_named_in("Tamil Nadu and Kerala, maybe ANDHRA") == ["TN", "KL", "AP"]
    assert states_named_in("Pondicherry and Hampi") == []  # a town, not the territory
    assert states_named_in("Puducherry beaches") == ["PY"]
    assert states_named_in("Keralan food, Telanganas") == []  # not a whole word
