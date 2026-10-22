"""Metadata filters over a ranked candidate list (D85, D69; plan.md Phase 6 item 5). Pure.

Filters = {"state": [..], "season": winter|summer|monsoon|post_monsoon, "month": 1-12,
"place_type": [..]}. They come from the validated query parse (Filters.from_parsed) or are given
explicitly, and an explicit dict overrides the parsed one. An unknown key raises.

- STATE: the frozen registry-derived `state` field, so it applies to every provenance class.
- SEASON/MONTH: a chunk passes unless its season_tags name only other seasons; all_year and empty
  tags always pass. A month fixes the season (D53 table).
- PLACE_TYPE: a hard filter on provenance `llm` rows ONLY. `rule_only` and `fallback` rows carry
  RULE-BASED place_type values (D69), so they are EXEMPT: neither removed nor boosted. The
  `rule_based_place_type=True` switch (diagnostic 5b) applies the filter to every row; the trace
  then says so.
- BACK-FILL: partition() puts the failing candidates in `rejected`, in rank order; the caller
  appends them after the passing ones, so fewer than k passing candidates are topped up from the
  unfiltered ranking and the number used is reported.

mentioned_places is never read (D50).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.common.schemas import PlaceType, State

FILTER_KEYS = ("state", "season", "month", "place_type")
SEASONS = ("winter", "summer", "monsoon", "post_monsoon")
MONTH_SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
                6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post_monsoon",
                11: "post_monsoon"}
STATES = tuple(x.value for x in State)
PLACE_TYPES = tuple(x.value for x in PlaceType)
RULE_BASED = ("rule_only", "fallback")  # provenance classes whose place_type is rule-based (D69)


@dataclass(frozen=True)
class Filters:
    state: tuple[str, ...] | None = None
    season: str | None = None
    month: int | None = None
    place_type: tuple[str, ...] | None = None

    @property
    def target_season(self) -> str | None:
        """The season to match: a month fixes it, else the stated season."""
        return MONTH_SEASON[self.month] if self.month else self.season

    @property
    def empty(self) -> bool:
        return not (self.state or self.target_season or self.place_type)

    def describe(self) -> dict:
        return {"state": list(self.state) if self.state else None,
                "season": self.target_season,
                "place_type": list(self.place_type) if self.place_type else None}

    @classmethod
    def from_dict(cls, d: dict | None) -> Filters:
        d = d or {}
        unknown = set(d) - set(FILTER_KEYS)
        if unknown:
            raise ValueError(f"unknown filter keys {sorted(unknown)}; allowed: {FILTER_KEYS}")
        month = d.get("month")
        if month is not None and (isinstance(month, bool) or month not in MONTH_SEASON):
            raise ValueError("month must be 1-12")
        season = d.get("season")
        if season is not None and season not in SEASONS:
            raise ValueError(f"season must be one of {SEASONS}")
        state = _checked(d.get("state"), STATES, "state")
        place_type = _checked(d.get("place_type"), PLACE_TYPES, "place_type")
        return cls(state=state, season=season, month=month, place_type=place_type)

    @classmethod
    def from_parsed(cls, parsed: dict, explicit: dict | None = None) -> Filters:
        """Filters from a validated parse (ValidatedQuery.to_dict()), overridden key by key by an
        explicit dict. Reads only the parse fields states, season, month and place_types."""
        base = {"state": parsed.get("states") or None, "season": parsed.get("season"),
                "month": parsed.get("month"), "place_type": parsed.get("place_types") or None}
        # an explicit key overrides the parsed one even when its value is empty: None or []
        # clears that filter; an invalid value (month 0, an unknown state) raises
        return cls.from_dict({**base, **(explicit or {})})


def _checked(values, allowed: tuple[str, ...], name: str) -> tuple[str, ...] | None:
    """A non-empty tuple of allowed values, or None for no filter (None or an empty list)."""
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{name} must be a list of values, not a string")
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"unknown {name} value(s) {bad}; allowed: {allowed}")
    return tuple(values) or None


def load_provenance(path: Path) -> dict[str, str]:
    """{chunk_id: llm | fallback | rule_only} from data/enriched/provenance.jsonl."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["chunk_id"]] = r["source"]
    return out


def _season_ok(tags: list[str], target: str) -> bool:
    return not tags or "all_year" in tags or target in tags


def chunk_verdict(chunk: dict, provenance: str, f: Filters, *,
                  rule_based_place_type: bool = False) -> tuple[str | None, bool]:
    """(reason it fails or None, place_type filter skipped because the row is rule-based)."""
    if f.state and chunk.get("state") not in f.state:
        return "state", False
    target = f.target_season
    if target and not _season_ok(list(chunk.get("season_tags") or []), target):
        return "season", False
    exempt = False
    if f.place_type:
        if provenance in RULE_BASED and not rule_based_place_type:
            exempt = True
        elif chunk.get("place_type") not in f.place_type:
            return "place_type", False
    return None, exempt


def partition(ranked: list[tuple[str, float]], chunks: dict[str, dict],
              provenance: dict[str, str], f: Filters, *, rule_based_place_type: bool = False
              ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], dict]:
    """Split a ranking into (passing, rejected), both in rank order, plus a trace:
    {"filters", "rejected": {reason: n}, "rejected_by_provenance": {class: n}, "exempt": n,
    "rule_based_place_type": bool}. `exempt` counts passing rows whose place_type filter was
    skipped because their provenance is rule_only or fallback (D69)."""
    passing, rejected = [], []
    reasons: Counter = Counter()
    by_prov: Counter = Counter()
    exempt = 0
    for cid, score in ranked:
        why, skipped = chunk_verdict(chunks[cid], provenance.get(cid, "unknown"), f,
                                     rule_based_place_type=rule_based_place_type)
        if why:
            rejected.append((cid, score))
            reasons[why] += 1
            by_prov[provenance.get(cid, "unknown")] += 1
        else:
            passing.append((cid, score))
            exempt += skipped
    trace = {"filters": f.describe(), "rejected": dict(reasons),
             "rejected_by_provenance": dict(by_prov), "exempt": exempt,
             "rule_based_place_type": rule_based_place_type}
    return passing, rejected, trace
