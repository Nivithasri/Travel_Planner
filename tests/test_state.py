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
    p = parsed(duration_days=6, budget_tier="luxury", month=7, travel_style="family",
               places=["Hampi"], states=["KA"], interests=["temples"], time_sensitive=True)
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

def test_the_committed_live_config_is_gated_off_until_the_owner_verifies_the_terms():
    cfg = load_live_config()
    assert cfg.terms_status == "not_verified" and cfg.real_calls_status() == "terms_not_verified"
    assert cfg.credits_per_search == 2 and cfg.max_credits_per_request == 6
    assert cfg.endpoint_host == "api.tavily.com" and cfg.cache_enabled


def test_real_calls_status_follows_enabled_and_terms():
    base = json.loads((REPO_ROOT / "config" / "live.json").read_text(encoding="utf-8"))
    assert LiveConfig(**{**base, "enabled": False, "terms_status": "verified"}
                      ).real_calls_status() == "disabled"
    assert LiveConfig(**{**base, "terms_status": "verified"}).real_calls_status() == "ok"


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
