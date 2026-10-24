"""The Phase 9 request set: loading and validation (D123).

The 10 itinerary requests live in eval/phase9/requests.json. Nothing may run before the owner has
set meta.status to APPROVED. `validate_requests` returns a list of problems (empty = valid) and
never raises on bad content, so the command-line check can print all of them at once.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from src.agents.evidence import INTEREST_MAP
from src.agents.scope import out_of_scope_matcher
from src.common.schemas import REPO_ROOT

REQUESTS_PATH = REPO_ROOT / "eval" / "phase9" / "requests.json"
APPROVED = "APPROVED"
N_REQUESTS = 10
BUDGETS = ("budget", "mid", "luxury")
PARTIES = ("solo", "couple", "family", "group")
OVERRIDE_KEYS = frozenset({
    "duration_days", "budget_tier", "party", "month", "start_city", "end_city", "interests",
    "constraints", "max_daily_travel_hours", "budget_cap_inr", "time_sensitive"})
REQUIRED = ("id", "type", "time_sensitive", "text", "named_places", "overrides")


class NotApproved(RuntimeError):
    """The request set has not been approved by the owner (meta.status is not APPROVED)."""


def load_requests(path: Path = REQUESTS_PATH, *, require_approved: bool = False) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_approved and doc.get("meta", {}).get("status") != APPROVED:
        raise NotApproved("eval/phase9/requests.json is not APPROVED: the owner has to approve "
                          "the request list before any system is run (D123)")
    return list(doc["requests"])


def load_document(path: Path = REQUESTS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(entries: Iterable) -> dict[str, str]:
    """name or alias (casefolded) -> canonical registry name, for non-hold-out entries."""
    out: dict[str, str] = {}
    for e in entries:
        if getattr(e, "holdout", False):
            continue
        out[e.name.casefold()] = e.name
        for a in getattr(e, "aliases", []) or []:
            out[a.casefold()] = e.name
    return out


def validate_requests(doc: dict, *, registry_entries: Iterable, excluded: Iterable[str],
                      city_ok: Callable[[str], bool] | None = None,
                      n_expected: int = N_REQUESTS) -> list[str]:
    """Problems found in the request document; an empty list means it is valid."""
    problems: list[str] = []
    entries = list(registry_entries)
    canon = _canonical(entries)
    excluded = set(excluded)
    reqs = doc.get("requests", [])
    if len(reqs) != n_expected:
        problems.append(f"expected {n_expected} requests, found {len(reqs)}")
    ids = [r.get("id") for r in reqs]
    if len(set(ids)) != len(ids):
        problems.append("request ids are not unique")
    nothing_allowed = out_of_scope_matcher(entries, excluded, [])
    n_ts = 0
    for r in reqs:
        rid = r.get("id", "?")
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"{rid}: missing field(s) {missing}")
            continue
        o = r["overrides"]
        unknown = sorted(set(o) - OVERRIDE_KEYS)
        if unknown:
            problems.append(f"{rid}: unknown override key(s) {unknown}")
        if not isinstance(o.get("duration_days"), int) or not 1 <= o["duration_days"] <= 10:
            problems.append(f"{rid}: duration_days must be an integer 1 to 10")
        if o.get("budget_tier") not in BUDGETS:
            problems.append(f"{rid}: budget_tier must be one of {BUDGETS}")
        if o.get("party") not in PARTIES:
            problems.append(f"{rid}: party must be one of {PARTIES}")
        month = o.get("month")
        if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
            problems.append(f"{rid}: month must be 1 to 12")
        for it in o.get("interests", []):
            if it not in INTEREST_MAP:
                problems.append(f"{rid}: interest {it!r} is not in the closed list")
        for field in ("start_city", "end_city"):
            city = o.get(field)
            if city and city_ok is not None and not city_ok(city):
                problems.append(f"{rid}: {field} {city!r} is not in the routing table")
        text = r["text"]
        if nothing_allowed(text):
            problems.append(f"{rid}: the text names a hold-out or D33-excluded place")
        for name in r["named_places"]:
            canonical = canon.get(name.casefold())
            if canonical is None:
                problems.append(f"{rid}: named place {name!r} is not a registry place")
            elif canonical in excluded:
                problems.append(f"{rid}: named place {name!r} is D33-excluded")
            elif canonical != name:
                problems.append(f"{rid}: named place {name!r} must be the registry name "
                                f"{canonical!r}")
        if r["time_sensitive"]:
            n_ts += 1
            if not r["named_places"]:
                problems.append(f"{rid}: a time-sensitive request must name a place")
            if o.get("time_sensitive") is not True:
                problems.append(f"{rid}: time_sensitive request needs overrides.time_sensitive")
    need = doc.get("meta", {}).get("time_sensitive_min", 4)
    if n_ts < need:
        problems.append(f"at least {need} time-sensitive requests are required, found {n_ts}")
    return problems
