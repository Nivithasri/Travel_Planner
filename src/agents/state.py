"""State models of the Phase 7 pipeline (D99 to D104): the parsed request with its defaults, the
plan, the evidence items and the run state. Pure data, no I/O except the config loaders."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.schemas import REPO_ROOT

PLANNER_CONFIG = REPO_ROOT / "config" / "planner.json"
BUDGET_CONFIG = REPO_ROOT / "config" / "budget_tiers.json"

BudgetTier = Literal["budget", "mid", "luxury"]
Party = Literal["solo", "couple", "family", "group"]
LiveStatus = Literal["ok", "empty", "disabled", "terms_not_verified", "no_key", "quota", "error",
                     "not_run"]
EvidenceClass = Literal["full", "limited"]
STYLE_TO_PARTY = {"family": "family", "honeymoon": "couple", "backpacker": "solo"}
HILL_AVOID_WORDS = ("avoid hill", "no hill", "no ghat", "avoid ghat", "avoid mountain")
NIGHT_WORDS = ("no night", "avoid night", "daytime only", "no overnight travel")
# How a state is NAMED in a request (D115). "Pondicherry" is deliberately absent from PY: it is
# also the town, and naming the town is not asking for the territory only.
STATE_WORDS = {"TN": ("tamil nadu", "tamilnadu"), "KL": ("kerala",), "KA": ("karnataka",),
               "AP": ("andhra pradesh", "andhra"), "TS": ("telangana",), "PY": ("puducherry",)}


def load_planner_config(path: Path = PLANNER_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_budget_tiers(path: Path = BUDGET_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ParsedRequest(BaseModel):
    """The request after parsing and validation, with every default applied (see `notes`)."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""  # the user's free text, used for retrieval only; NEVER sent to the live search
    duration_days: int = 4
    budget_tier: BudgetTier = "mid"
    budget_cap_inr: int | None = Field(default=None, gt=0)
    month: int | None = None  # None: season checks are skipped and the output says so
    start_city: str | None = None
    end_city: str | None = None
    interests: list[str] = Field(default_factory=list)
    travel_style: str | None = None
    party: Party = "group"
    constraints: list[str] = Field(default_factory=list)
    max_daily_travel_hours: float | None = Field(default=None, gt=0, le=14)
    named_places: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    place_types: list[str] = Field(default_factory=list)
    time_sensitive: bool = False
    notes: list[str] = Field(default_factory=list)  # every default applied, every drop

    @field_validator("duration_days")
    @classmethod
    def _days(cls, v: int) -> int:
        cfg = load_planner_config()
        if not cfg["min_days"] <= v <= cfg["max_days"]:
            raise ValueError(f"duration_days must be {cfg['min_days']} to {cfg['max_days']}")
        return v

    @field_validator("month")
    @classmethod
    def _month(cls, v: int | None) -> int | None:
        if v is not None and (isinstance(v, bool) or not 1 <= v <= 12):
            raise ValueError("month must be 1 to 12")
        return v

    @property
    def avoid_hill_roads(self) -> bool:
        return any(w in c.casefold() for c in self.constraints for w in HILL_AVOID_WORDS)

    @property
    def no_night_travel(self) -> bool:
        return any(w in c.casefold() for c in self.constraints for w in NIGHT_WORDS)

    def daily_limit_hours(self, cfg: dict | None = None) -> float:
        """The daily travel limit: an explicit value wins, else the party default."""
        if self.max_daily_travel_hours is not None:
            return float(self.max_daily_travel_hours)
        table = (cfg or load_planner_config())["daily_travel_limit_hours"]
        return float(table.get(self.party, table["default"]))


def states_named_in(text: str) -> list[str]:
    """State codes whose name appears in the text (whole words, any case)."""
    low = (text or "").casefold()
    return [code for code, words in STATE_WORDS.items()
            if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words)]


def request_from_parse(parsed: Any, *, overrides: dict | None = None) -> ParsedRequest:
    """Build a ParsedRequest from a ValidatedQuery (src/retrieval/query_understanding.py) or any
    object with the same attributes, plus explicit overrides (CLI flags or a form). Missing fields
    take the config defaults and each default is noted."""
    cfg = load_planner_config()
    o = dict(overrides or {})
    notes: list[str] = []

    def pick(name: str, parsed_value, default, label: str | None = None):
        if name in o and o[name] is not None:
            return o[name]
        if parsed_value not in (None, [], ""):
            return parsed_value
        if label:
            notes.append(f"{label} not given: default {default!r} used")
        return default

    style = getattr(parsed, "travel_style", None)
    party = o.get("party") or STYLE_TO_PARTY.get(style or "", None)
    if party is None:
        party = "group"
        notes.append("party not given: 'group' assumed")
    # A state restricts the plan only when the user named it (or passed it explicitly, as the CLI
    # does). The parse also lists the states of the places a request names; those are not a
    # restriction (D115): naming Mahabalipuram and Pondicherry did not ask for Puducherry only.
    if o.get("states"):
        states = list(o["states"])
    else:
        inferred = list(getattr(parsed, "states", []) or [])
        named = set(states_named_in(str(o.get("text", getattr(parsed, "text", "")) or "")))
        states = [s for s in inferred if s in named]
        dropped = [s for s in inferred if s not in named]
        if dropped:
            notes.append(f"state(s) {', '.join(dropped)} came from the places you named, not from "
                         "a state you named: not used to limit the plan")
    req = ParsedRequest(
        text=o.get("text", getattr(parsed, "text", "")),
        duration_days=pick("duration_days", getattr(parsed, "duration_days", None),
                           cfg["default_days"], "duration"),
        budget_tier=pick("budget_tier", getattr(parsed, "budget_tier", None),
                         cfg["default_budget_tier"], "budget tier"),
        budget_cap_inr=o.get("budget_cap_inr"),
        month=pick("month", getattr(parsed, "month", None), None, "month"),
        start_city=o.get("start_city"), end_city=o.get("end_city"),
        interests=list(o.get("interests") or getattr(parsed, "interests", []) or []),
        travel_style=style, party=party,
        constraints=list(o.get("constraints") or []),
        max_daily_travel_hours=o.get("max_daily_travel_hours"),
        named_places=list(o.get("named_places") or getattr(parsed, "places", []) or []),
        states=states,
        place_types=list(getattr(parsed, "place_types", []) or []),
        time_sensitive=bool(o.get("time_sensitive", getattr(parsed, "time_sensitive", False))),
        notes=notes + list(getattr(parsed, "notes", []) or []))
    if req.month is None:
        req.notes.append("no month given: season checks are skipped")
    return req


# ---- plan ---------------------------------------------------------------------------------

class PlanStop(BaseModel):
    place: str
    state: str
    nights: int
    evidence_class: EvidenceClass = "full"
    visibility_tercile: int | None = None
    season_verdict: str | None = None
    access_risk: str | None = None
    season_confidence: str | None = None
    hill: bool = False
    named: bool = False
    score: float = 0.0


class PlanLeg(BaseModel):
    origin: str
    destination: str
    hours: float
    days: int  # travel days this leg takes under the daily limit (0 for a zero-hour leg)
    hill_note: bool = False
    kind: Literal["arrival", "between", "departure"] = "between"


class PlanDay(BaseModel):
    day: int
    place: str
    travel_from: str | None = None
    travel_hours: float = 0.0
    note: str | None = None


class Plan(BaseModel):
    feasible: bool
    infeasible_reason: str | None = None
    stops: list[PlanStop] = Field(default_factory=list)
    legs: list[PlanLeg] = Field(default_factory=list)
    days: list[PlanDay] = Field(default_factory=list)
    limit_hours: float = 5.0
    total_travel_hours: float = 0.0
    budget: dict = Field(default_factory=dict)
    backups: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    computed_lines: list[str] = Field(default_factory=list)


# ---- evidence -----------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    chunk_id: str
    place: str
    source_type: str
    url: str
    text: str
    aspect: str | None = None
    place_type: str | None = None  # the chunk's subject (D53); used only to match interests
    is_warning: bool = False
    provenance: str = "unknown"  # llm | fallback | rule_only | live
    date: str | None = None
    fetch_date: str | None = None
    score: float = 0.0

    @property
    def is_live(self) -> bool:
        return self.source_type == "live" or self.chunk_id.startswith("live:")


class CriticFinding(BaseModel):
    rule: str  # R1 .. R7
    message: str
    day: int | None = None
    fixable: bool = True


class CriticReport(BaseModel):
    ok: bool
    findings: list[CriticFinding] = Field(default_factory=list)
    status: Literal["passed", "repaired", "failed_rules"] = "passed"


class RunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_text: str = ""
    overrides: dict = Field(default_factory=dict)  # CLI flags / form fields (no free text to live)
    request: ParsedRequest | None = None
    candidates: list[dict] = Field(default_factory=list)
    grades: dict = Field(default_factory=dict)  # per place: thin flag and why
    packed_obj: Any = None  # the evidence.Packed object (aliases, dropped items)
    corpus_evidence: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    live_evidence: list[EvidenceItem] = Field(default_factory=list)
    live_status: LiveStatus = "not_run"
    live_note: str | None = None
    plan: Plan | None = None
    packed: dict[str, str] = Field(default_factory=dict)  # alias (S1, L2) -> real chunk id
    draft: str = ""
    critic: CriticReport | None = None
    repaired: bool = False
    final: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, int] = Field(default_factory=dict)
