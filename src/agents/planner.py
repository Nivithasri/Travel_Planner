"""The deterministic trip planner (Phase 7, D101): no LLM, no network, no clock, no randomness.

Inputs are the parsed request, scored candidates (from retrieval) and two injected lookups,
`hours(a, b)` (src/tools/travel_time.py) and `season(place, month)` (src/tools/check_season.py), so
tests can swap them. Rules:

* destinations are the full-evidence places plus places the user NAMED (a D33-excluded place only
  when named, then evidence_class "limited"); hubs and hold-outs are never stops;
* stops are taken from the best-scored travel-time cluster (single link, `cluster_hours`), ordered
  by brute force over at most `max_stops` places to minimise the start leg plus the legs;
* a leg of h hours takes ceil(h / limit) days; the daily limit is never exceeded: if the stops do
  not fit in the requested days the lowest-scored unnamed stop is dropped, and if even one stop
  does not fit the plan is reported infeasible (never silently stretched);
* distances are hours, not km (hill legs, D78); routing data is PROVISIONAL (D77);
* the popular/hidden mix is the lowest visibility tercile, labelled relative (D101).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import permutations

from src.agents.budget import budget_line, estimate
from src.agents.state import (
    ParsedRequest,
    Plan,
    PlanDay,
    PlanLeg,
    PlanStop,
    load_planner_config,
)

HoursFn = Callable[[str, str], float]
SeasonFn = Callable[[str, int], dict]
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")


@dataclass
class Candidate:
    name: str
    state: str
    relevance: float = 0.0  # 0..1, normalised retrieval relevance
    interest: float = 0.0  # 0..1, interest / place_type match
    named: bool = False
    card: dict = field(default_factory=dict)  # the place card (evidence_class, hill, tercile)

    @property
    def limited(self) -> bool:
        return self.card.get("evidence_class") == "limited"

    @property
    def hill(self) -> bool:
        return bool(self.card.get("hill"))

    @property
    def tercile(self) -> int | None:
        return self.card.get("visibility_tercile")


def _season(season: SeasonFn | None, place: str, month: int | None) -> dict | None:
    if season is None or month is None:
        return None
    try:
        return season(place, month)
    except ValueError:  # UnknownPlaceError is a ValueError: no climate row for this place
        return None


def score_candidates(cands: Sequence[Candidate], request: ParsedRequest, season: SeasonFn | None,
                     cfg: dict, start_hours: dict[str, float] | None = None) -> dict[str, float]:
    w = cfg["weights"]
    out = {}
    for c in cands:
        s = w["relevance"] * c.relevance + w["interest"] * c.interest + (w["named"] if c.named
                                                                         else 0.0)
        sd = _season(season, c.name, request.month)
        if sd is not None:
            s -= {"poor": w["season_poor"], "fair": w["season_fair"]}.get(sd["verdict"], 0.0)
        if c.tercile == cfg["visibility_terciles"]:
            s += w["hidden_bonus"]
        if start_hours and not c.named:  # a far place costs travel time: mild preference for near
            s -= w["start_hours"] * min(start_hours.get(c.name, 0.0), w["start_hours_cap"])
        out[c.name] = round(s, 6)
    return out


def _clusters_from(seed: str, names: Sequence[str], hours: HoursFn, limit: float) -> list[str]:
    """Names reachable from the seed through hops of at most `limit` hours (single link)."""
    seen, frontier = {seed}, [seed]
    while frontier:
        cur = frontier.pop()
        for n in names:
            if n not in seen and hours(cur, n) <= limit:
                seen.add(n)
                frontier.append(n)
    return sorted(seen)


def select_stops(pool: dict[str, Candidate], scores: dict[str, float], n: int, hours: HoursFn,
                 cluster_hours: float) -> list[str]:
    """n stop names: named places first (they bypass the cluster), then the best-scored candidates
    of the seed's cluster, then the nearest others. Deterministic: ties by name."""
    order = sorted(pool, key=lambda k: (-scores[k], k))
    named = [k for k in order if pool[k].named]
    chosen = named[:n]
    seed = chosen[0] if chosen else order[0]
    if seed not in chosen:
        chosen.append(seed)
    cluster = _clusters_from(seed, order, hours, cluster_hours)
    for k in order:
        if len(chosen) >= n:
            break
        if k in cluster and k not in chosen:
            chosen.append(k)
    if len(chosen) < n:  # the cluster is smaller than n: nearest others to the seed
        rest = sorted((k for k in order if k not in chosen), key=lambda k: (hours(seed, k), k))
        chosen.extend(rest[:n - len(chosen)])
    return chosen[:n]


def best_order(stops: Sequence[str], start: str | None, end: str | None, hours: HoursFn
               ) -> tuple[list[str], float]:
    """The permutation of `stops` with the least start leg + legs + end leg (hours)."""
    best: tuple[float, tuple[str, ...]] | None = None
    for perm in permutations(sorted(stops)):
        cost = hours(start, perm[0]) if start else 0.0
        cost += sum(hours(a, b) for a, b in zip(perm, perm[1:]))
        cost += hours(perm[-1], end) if end else 0.0
        if best is None or cost < best[0] - 1e-9:
            best = (cost, perm)
    assert best is not None
    return list(best[1]), best[0]


def _extra_days(h: float, limit: float) -> int:
    return 0 if h <= 0 else math.ceil(h / limit - 1e-9) - 1


def _legs(order: Sequence[str], start: str | None, end: str | None, hours: HoursFn, limit: float,
          cards: dict[str, dict]) -> list[PlanLeg]:
    legs = []
    seq = ([start] if start else []) + list(order)
    for i, (a, b) in enumerate(zip(seq, seq[1:])):
        h = round(hours(a, b), 2)
        hill = bool(cards.get(a, {}).get("hill") or cards.get(b, {}).get("hill"))
        kind = "arrival" if (start and i == 0) else "between"
        legs.append(PlanLeg(origin=a, destination=b, hours=h, days=0 if h <= 0 else
                            _extra_days(h, limit) + 1, hill_note=hill, kind=kind))
    if end:
        h = round(hours(order[-1], end), 2)
        hill = bool(cards.get(order[-1], {}).get("hill"))
        legs.append(PlanLeg(origin=order[-1], destination=end, hours=h,
                            days=0 if h <= 0 else _extra_days(h, limit) + 1, hill_note=hill,
                            kind="departure"))
    return legs


def _hours_between(hours: HoursFn, a: str, b: str) -> float:
    return 0.0 if a.casefold() == b.casefold() else float(hours(a, b))


def plan_trip(request: ParsedRequest, candidates: Sequence[Candidate], *, hours: HoursFn,
              season: SeasonFn | None = None, cfg: dict | None = None,
              budget_cfg: dict | None = None) -> Plan:
    cfg = cfg or load_planner_config()
    limit = request.daily_limit_hours(cfg)
    notes: list[str] = []
    h = lambda a, b: _hours_between(hours, a, b)  # noqa: E731
    pool = {c.name: c for c in candidates if c.named or not c.limited}
    if len(pool) < len(candidates):
        notes.append("excluded places were not used because you did not name them (D33)")
    if request.states:  # the request names states: other states are not offered unless named
        keep = {k: c for k, c in pool.items() if c.state in request.states or c.named}
        if len(keep) < len(pool):
            outside = sorted(k for k, c in keep.items()
                             if c.named and c.state not in request.states)
            where = ", ".join(request.states)
            if outside:
                notes.append(f"unnamed places were limited to {where}, as you asked; places you "
                             f"named outside it were kept ({', '.join(outside)})")
            else:
                notes.append(f"only places in {where} were considered, as asked")
        pool = keep
        if not pool:
            return Plan(feasible=False, infeasible_reason=(
                f"no evaluated destination with evidence in {', '.join(request.states)}"),
                limit_hours=limit, notes=notes)
    if request.avoid_hill_roads:
        for name in [k for k, c in pool.items() if c.hill and not c.named]:
            del pool[name]
        if any(c.hill and c.named for c in pool.values()):
            notes.append("you asked to avoid hill roads but named a hill place; it is kept")
    if not pool:
        return Plan(feasible=False, infeasible_reason="no candidate destination with evidence",
                    limit_hours=limit, notes=notes)
    start_hours = None
    if request.start_city:
        start_hours = {k: h(request.start_city, k) for k in pool}
    scores = score_candidates(list(pool.values()), request, season, cfg, start_hours)
    named_n = sum(1 for c in pool.values() if c.named)
    target = max(request.duration_days // cfg["days_per_stop"], 1, min(named_n, cfg["max_stops"]))
    target = min(target, cfg["max_stops"], request.duration_days, len(pool))
    poor = {k for k in pool if not pool[k].named and (sd := _season(season, k, request.month))
            and sd["verdict"] == "poor"}
    if poor and len(pool) - len(poor) >= target:  # a poor-season place is used only if needed
        for k in poor:
            del pool[k]
        notes.append(f"{len(poor)} place(s) rated poor in {MONTHS[request.month - 1]} were left "
                     "out because enough better-rated places exist (climate normals)")
    cards = {k: c.card for k, c in pool.items()}
    start, end = request.start_city, request.end_city

    order: list[str] = []
    legs: list[PlanLeg] = []
    nights: dict[str, int] = {}
    feasible = False
    reason = None
    for n in range(target, 0, -1):
        base = select_stops(pool, scores, n, h, cfg["cluster_hours"])
        variants = [(base, None)]
        if n >= 2:
            mixed, mix_note = hidden_mix_variant(base, pool, scores, h, cfg)
            variants = [(mixed, mix_note), (base, None)] if mixed != base else [(base, mix_note)]
        for stops, mix_note in variants:
            order, _ = best_order(stops, start, end, h)
            legs = _legs(order, start, end, h, limit, cards)
            extra = sum(max(leg.days - 1, 0) for leg in legs if leg.kind != "departure")
            if len(stops) + extra <= request.duration_days:
                nights = _spread_nights(order, scores, request.duration_days - len(stops) - extra)
                feasible = True
                if mix_note:
                    notes.append(mix_note)
                if n < target:
                    notes.append(f"{target - n} stop(s) dropped so the legs fit "
                                 f"{request.duration_days} days under the {limit:g} h daily "
                                 "travel limit")
                break
            reason = (f"{len(stops)} stop(s) need {len(stops) + extra} day(s) with the "
                      f"{limit:g} h daily limit, but only {request.duration_days} day(s) were "
                      "asked")
        if feasible:
            break
    if not feasible:
        return Plan(feasible=False, infeasible_reason=reason, limit_hours=limit, notes=notes)

    plan_stops = []
    for k in order:
        c = pool[k]
        sd = _season(season, k, request.month)
        plan_stops.append(PlanStop(
            place=k, state=c.state, nights=nights[k],
            evidence_class="limited" if c.limited else "full", visibility_tercile=c.tercile,
            season_verdict=sd["verdict"] if sd else None,
            access_risk="high" if sd and sd["access_risk"] else ("none" if sd else None),
            season_confidence=sd["confidence"] if sd else None, hill=c.hill, named=c.named,
            score=scores[k]))
        if sd and sd["verdict"] == "poor":
            notes.append(f"{k} is rated poor in {MONTHS[request.month - 1]} (climate normals)")
        if sd and sd["access_risk"]:
            notes.append(f"{k}: road access risk in {MONTHS[request.month - 1]} (heavy rain on "
                         "hill roads)")
        if c.limited:
            notes.append(f"{k}: limited evidence (excluded from the evaluated corpus, D33)")

    rows: list[PlanDay] = []
    day = 1
    by_dest = {leg.destination: leg for leg in legs if leg.kind != "departure"}
    for stop in plan_stops:
        leg = by_dest.get(stop.place)
        arrival_days = max(leg.days, 1) if leg else 1
        for t in range(arrival_days):
            rows.append(PlanDay(day=day, place=stop.place,
                                travel_from=leg.origin if leg and leg.hours > 0 else None,
                                travel_hours=round(leg.hours / arrival_days, 2) if leg else 0.0,
                                note=(f"travel day {t + 1} of {arrival_days}"
                                      if arrival_days > 1 else None)))
            day += 1
        for _ in range(stop.nights - 1):
            rows.append(PlanDay(day=day, place=stop.place))
            day += 1
    assert day - 1 == request.duration_days
    total_hours = round(sum(leg.hours for leg in legs), 2)
    for leg in legs:
        if leg.kind == "departure" and leg.days > 1:
            notes.append(f"the departure leg to {leg.destination} (about {leg.hours:g} h) exceeds "
                         f"the {limit:g} h daily limit: plan an extra travel day after the trip")
    b = estimate(request.duration_days, request.budget_tier, total_hours,
                 cap_inr=request.budget_cap_inr, cfg=budget_cfg)
    plan = Plan(feasible=True, stops=plan_stops, legs=legs, days=rows, limit_hours=limit,
                total_travel_hours=total_hours, budget=b, notes=notes)
    plan.backups = _backups(pool, scores, order, h, season, request, cfg)
    plan.computed_lines = _computed_lines(plan, request)
    return plan


def hidden_mix_variant(stops: list[str], pool: dict[str, Candidate], scores: dict[str, float],
                       hours: HoursFn, cfg: dict) -> tuple[list[str], str | None]:
    """Swap the lowest-scored unnamed stop for the best less-visited candidate that sits near the
    REMAINING stops. Returns (stops, note); unchanged stops with a note when no such candidate
    exists, unchanged with no note when a less-visited stop is already there."""
    top = cfg["visibility_terciles"]
    if any(pool[k].tercile == top for k in stops):
        return stops, None
    swappable = sorted((k for k in stops if not pool[k].named), key=lambda k: (scores[k], k))
    no_note = ("no less-visited candidate near this route; the popular/hidden mix was not "
               "applied (every candidate is popular by registry label)")
    if not swappable:
        return stops, no_note
    drop = swappable[0]
    rest = [k for k in stops if k != drop]
    near = [k for k in pool if k not in stops and pool[k].tercile == top and not pool[k].limited
            and min(hours(k, s) for s in rest) <= cfg["cluster_hours"] * 1.5]
    if not near:
        return stops, no_note
    pick = sorted(near, key=lambda k: (-scores[k], k))[0]
    return ([pick if k == drop else k for k in stops],
            f"{pick} was included as a relatively less-visited stop (lowest visibility tercile "
            "among the 17 evaluated places; still popular by registry label)")


def _spread_nights(order: list[str], scores: dict[str, float], spare: int) -> dict[str, int]:
    """One night per stop, spare nights to the best-scored stops in turn."""
    nights = {k: 1 for k in order}
    by_score = sorted(order, key=lambda k: (-scores[k], k))
    for i in range(spare):
        nights[by_score[i % len(by_score)]] += 1
    return nights


def _backups(pool: dict[str, Candidate], scores: dict[str, float], order: list[str], hours: HoursFn,
             season: SeasonFn | None, request: ParsedRequest, cfg: dict) -> list[dict]:
    rest = sorted((k for k in pool if k not in order and not pool[k].limited),
                  key=lambda k: (-scores[k], k))[:cfg["backups"]]
    out = []
    for k in rest:
        near = min(order, key=lambda s: (hours(k, s), s))
        sd = _season(season, k, request.month)
        out.append({"kind": "destination", "place": k, "state": pool[k].state,
                    "nearest_stop": near, "hours_from_nearest_stop": round(hours(k, near), 2),
                    "season_verdict": sd["verdict"] if sd else None})
    return out


def _computed_lines(plan: Plan, request: ParsedRequest) -> list[str]:
    lines = []
    for leg in plan.legs:
        if leg.hours <= 0:  # the start city is the first stop: no leg to describe
            continue
        what = {"arrival": "arrival", "between": "transfer", "departure": "departure"}[leg.kind]
        extra = f", {leg.days} days" if leg.days > 1 else ""
        hill = " (includes a ghat/hill road: allow extra time)" if leg.hill_note else ""
        lines.append(f"{what.capitalize()}: {leg.origin} to {leg.destination}, about "
                     f"{leg.hours:g} h by road{extra}{hill} (computed: travel_time, provisional)")
    if request.month is not None:
        for s in plan.stops:
            if s.season_verdict is None:
                continue
            risk = "; road access risk in heavy rain" if s.access_risk == "high" else ""
            lines.append(f"{s.place} in {MONTHS[request.month - 1]}: {s.season_verdict} "
                         f"(climate normals, confidence {s.season_confidence}{risk}) "
                         "(computed: check_season)")
    lines.append(budget_line(plan.budget))
    return lines
