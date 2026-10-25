"""The rule-based critic and the single repair pass (Phase 7, D104).

Rules R1..R7 run on the writer's draft (aliases S#/L# still in place). Findings are never hidden:
after at most ONE repair pass whatever remains is reported with status "failed_rules".

R1 plan feasibility and structure   R2 season conflicts mentioned   R3 budget fit
R4 citations (every bullet tagged, every tag real)   R5 scope (hold-outs, excluded places, limited
caveat)   R6 untrusted text and recency words   R7 time-sensitive request without live evidence
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.agents.evidence import INJECTION, Packed, lead
from src.agents.state import CriticFinding, CriticReport, ParsedRequest, Plan
from src.agents.writer import ALIAS, NO_DETAILS, normalize_output

HEADING = re.compile(r"^#{2,3}\s*Day\s+(\d+)\s*[:\-]\s*(.+?)\s*$", re.I)
BACKUP = re.compile(r"^#{2,3}\s*Backup options\s*$", re.I)
BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
URL = re.compile(r"https?://|www\.", re.I)
RECENCY = re.compile(r"\b(recent|recently|currently|latest|up[- ]to[- ]date|right now|nowadays)\b",
                     re.I)
SEASON_WORDS = ("rain", "monsoon", "season", "heat", "hot ", "weather", "climate", "access",
                "landslide", "flood", "humid")
STATUS_LINE = ("- Current status (closures, timings, prices) could not be confirmed: no live "
               "evidence was available for this request.")


@dataclass
class DaySection:
    number: int
    place: str
    bullets: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # heading + every line of the section


@dataclass
class Draft:
    days: list[DaySection]
    backups: list[str]
    preamble: list[str]


def parse_draft(text: str) -> Draft:
    days: list[DaySection] = []
    backups: list[str] = []
    pre: list[str] = []
    cur: DaySection | None = None
    in_backup = False
    for line in normalize_output(text).splitlines():
        m = HEADING.match(line.strip())
        if m:
            cur = DaySection(int(m.group(1)), m.group(2).strip(), [], [line])
            days.append(cur)
            in_backup = False
            continue
        if BACKUP.match(line.strip()):
            cur, in_backup = None, True
            continue
        b = BULLET.match(line)
        if in_backup:
            if b:
                backups.append(b.group(1).strip())
            continue
        if cur is None:
            pre.append(line)
            continue
        cur.lines.append(line)
        if b:
            cur.bullets.append(b.group(1).strip())
    return Draft(days, backups, pre)


def render(draft: Draft) -> str:
    out = list(draft.preamble)
    for d in draft.days:
        out.extend(d.lines)
    if draft.backups:
        out.append("## Backup options")
        out.extend(f"- {b}" for b in draft.backups)
    return "\n".join(out).strip() + "\n"


@dataclass
class CriticContext:
    request: ParsedRequest
    plan: Plan
    packed: Packed
    live_status: str = "not_run"
    excluded_names: frozenset[str] = frozenset()
    holdout_names: frozenset[str] = frozenset()


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text, re.I) is not None


NO_DETAILS_WORDS = re.compile(r"no source[- ]backed details", re.I)


def is_no_details(bullet: str) -> bool:
    """The no-details line, also when the model wraps it in brackets or changes the full stop."""
    return bool(NO_DETAILS_WORDS.search(bullet)) and len(bullet) <= 70 and not ALIAS.search(bullet)


def _is_exempt(bullet: str) -> bool:
    return (f"- {bullet}".strip() == NO_DETAILS or bullet.startswith("Current status")
            or is_no_details(bullet))


def blocked_names(ctx: CriticContext) -> list[str]:
    """Hold-out and excluded place names the answer must not mention: not a stop, not named in the
    request, not a backup of the plan (the R5 rule)."""
    allowed = {s.place.casefold() for s in ctx.plan.stops}
    allowed |= {n.casefold() for n in ctx.request.named_places}
    allowed |= {b["place"].casefold() for b in ctx.plan.backups}
    names = sorted(ctx.holdout_names | ctx.excluded_names)
    return [n for n in names if n.casefold() not in allowed]


def check(draft_text: str, ctx: CriticContext) -> list[CriticFinding]:
    d = parse_draft(draft_text)
    f: list[CriticFinding] = []
    plan, req = ctx.plan, ctx.request
    # R1 ---------------------------------------------------------------------------------------
    if not plan.feasible:
        f.append(CriticFinding(rule="R1", message="the plan is infeasible: "
                               f"{plan.infeasible_reason}", fixable=False))
    for row in plan.days:
        if row.travel_hours > plan.limit_hours + 1e-9:
            f.append(CriticFinding(rule="R1", day=row.day, fixable=False, message=(
                f"day {row.day} travel {row.travel_hours:g} h exceeds the {plan.limit_hours:g} h "
                "daily limit")))
    nums = [x.number for x in d.days]
    if nums != list(range(1, len(plan.days) + 1)):
        f.append(CriticFinding(rule="R1", fixable=False, message=(
            f"day headings {nums} do not match the {len(plan.days)} planned days")))
    for x in d.days:
        if 1 <= x.number <= len(plan.days):
            want = plan.days[x.number - 1].place
            if want.casefold() not in x.place.casefold():
                f.append(CriticFinding(rule="R1", day=x.number, message=(
                    f"day {x.number} heading names {x.place!r} but the plan says {want!r}")))
    # R2 ---------------------------------------------------------------------------------------
    for s in plan.stops:
        if s.season_verdict == "poor" or s.access_risk == "high":
            text = " ".join(b.lower() for x in d.days if x.place.casefold().startswith(
                s.place.casefold()) for b in x.bullets)
            if not any(w in text for w in SEASON_WORDS):
                day = next((x.number for x in d.days if s.place.casefold() in x.place.casefold()),
                           None)
                f.append(CriticFinding(rule="R2", day=day, message=(
                    f"{s.place} is rated {s.season_verdict} / access risk {s.access_risk} for "
                    "the travel month but no bullet for it mentions season, rain or access")))
    # R3 ---------------------------------------------------------------------------------------
    if plan.budget.get("within_cap") is False:
        f.append(CriticFinding(rule="R3", fixable=False, message=(
            f"the estimate INR {plan.budget['total']:,} is above the cap INR "
            f"{plan.budget['cap_inr']:,} plus tolerance")))
    # R4 ---------------------------------------------------------------------------------------
    amap = ctx.packed.alias_map
    seen_text: dict[str, int] = {}
    for x in d.days:
        for b in x.bullets:
            key = " ".join(ALIAS.sub("", b).split()).casefold()
            if _is_exempt(b) or not key:
                continue
            if key in seen_text:
                f.append(CriticFinding(rule="R4", day=x.number, message=(
                    f"day {x.number}: repeats a bullet from day {seen_text[key]}: {b[:60]!r}")))
            else:
                seen_text[key] = x.number
    for x in d.days:
        if not x.bullets:
            f.append(CriticFinding(rule="R4", day=x.number, message=(
                f"day {x.number} has no bullets (write the no-details bullet)")))
        for b in x.bullets:
            if _is_exempt(b):
                continue
            tags = ALIAS.findall(b)
            if not tags:
                f.append(CriticFinding(rule="R4", day=x.number, message=(
                    f"day {x.number}: uncited bullet: {b[:80]!r}")))
            for t in tags:
                if t not in amap:
                    f.append(CriticFinding(rule="R4", day=x.number, message=(
                        f"day {x.number}: unknown source tag [{t}]")))
    # R5 ---------------------------------------------------------------------------------------
    full = "\n".join(render(d).splitlines())
    for name in blocked_names(ctx):
        if not _mentions(full, name):
            continue
        # D122: a finding carries the day whose section names the place, so the one repair pass can
        # run for it; a mention only in the backups or the preamble has no day.
        days = [x.number for x in d.days if _mentions("\n".join(x.lines), name)] or [None]
        for day in days:
            f.append(CriticFinding(rule="R5", day=day, message=(
                f"{name} is outside the plan (hold-out or excluded place) but is mentioned")))
    for s in plan.stops:
        if s.evidence_class == "limited":
            text = " ".join(b.lower() for x in d.days if s.place.casefold() in x.place.casefold()
                            for b in x.bullets)
            if "limited" not in text:
                day = next((x.number for x in d.days if s.place.casefold() in x.place.casefold()),
                           None)
                f.append(CriticFinding(rule="R5", day=day, message=(
                    f"{s.place} has limited evidence but no bullet says so")))
    # R6 ---------------------------------------------------------------------------------------
    body = render(d)
    if URL.search(body):
        f.append(CriticFinding(rule="R6", message="the draft contains a URL"))
    if INJECTION.search(body):
        f.append(CriticFinding(rule="R6", message="the draft contains instruction-like text"))
    for x in d.days:
        for b in x.bullets:
            tags = ALIAS.findall(b)
            if RECENCY.search(b) and not any(t.startswith("L") for t in tags):
                f.append(CriticFinding(rule="R6", day=x.number, message=(
                    f"day {x.number}: a recency word on a corpus-only claim: {b[:80]!r}")))
    # R7 ---------------------------------------------------------------------------------------
    if req.time_sensitive and ctx.live_status != "ok" and STATUS_LINE not in body:
        f.append(CriticFinding(rule="R7", message=(
            "a time-sensitive request without live evidence must carry the cannot-confirm line")))
    return f


def ensure_status_line(draft_text: str, request: ParsedRequest, live_status: str) -> str:
    """R7 is met by code, not by the model: add the cannot-confirm bullet before the backups."""
    if not request.time_sensitive or live_status == "ok" or STATUS_LINE in draft_text:
        return draft_text
    d = parse_draft(draft_text)
    if d.days:
        d.days[-1].lines.append(STATUS_LINE)
        d.days[-1].bullets.append(STATUS_LINE[2:])
    else:
        d.preamble.append(STATUS_LINE)
    return render(d)


def deterministic_fix(draft_text: str, findings: list[CriticFinding], packed: Packed,
                      blocked: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    """Strip uncited or wrongly tagged bullets, URLs and bullets that name a blocked place (D122);
    give an emptied day the no-details bullet. Returns (new text, actions)."""
    d = parse_draft(draft_text)
    amap = packed.alias_map
    actions: list[str] = []
    seen_text: set[str] = set()
    for x in d.days:
        keep_lines, keep_bullets = [], []
        for line in x.lines:
            b = BULLET.match(line)
            if b and any(_mentions(line, n) for n in blocked):
                actions.append(f"day {x.number}: removed a bullet that names a place outside the "
                               "plan")
                continue
            if b and not _is_exempt(b.group(1).strip()):
                key = " ".join(ALIAS.sub("", b.group(1)).split()).casefold()
                if key in seen_text:
                    actions.append(f"day {x.number}: removed a repeated bullet {b.group(1)[:60]!r}")
                    continue
                seen_text.add(key)
            if b and not _is_exempt(b.group(1).strip()):
                tags = ALIAS.findall(b.group(1))
                if not tags or any(t not in amap for t in tags):
                    actions.append(f"day {x.number}: removed bullet {b.group(1)[:60]!r}")
                    continue
            if URL.search(line):
                actions.append(f"day {x.number}: removed a line containing a URL")
                continue
            keep_lines.append(line)
            if b:
                keep_bullets.append(b.group(1).strip())
        x.lines, x.bullets = keep_lines, keep_bullets
        if not x.bullets:
            x.lines.append(NO_DETAILS)
            x.bullets.append(NO_DETAILS[2:])
            actions.append(f"day {x.number}: wrote the no-details bullet")
    kept = [b for b in d.backups if not any(_mentions(b, n) for n in blocked)]
    if len(kept) != len(d.backups):
        actions.append("removed a backup option that names a place outside the plan")
        d.backups = kept
    return render(d), actions


def _real(day: DaySection) -> list[str]:
    return [b for b in day.bullets if not _is_exempt(b)]


def fill_empty_days(draft_text: str, ctx: CriticContext) -> tuple[str, list[str]]:
    """No LLM: a day with no real bullet (nothing, or only the no-details line) is filled while its
    place still has packed sources the draft did not use. Order: the sources assigned to that day,
    then any unused source of the place (the opening sentences of the source, with its tag); if the
    place has none left, the last bullet of a day of the same place that has three or more moves
    here. The no-details line stays only when nothing at all can be given to the day."""
    d = parse_draft(draft_text)
    by_alias = {p.alias: p for p in ctx.packed.items if p.alias.startswith("S")}
    place_of = {row.day: row.place for row in ctx.plan.days}
    used = {t for x in d.days for b in x.bullets for t in ALIAS.findall(b)}
    actions: list[str] = []
    for x in d.days:
        if _real(x):
            continue
        place = place_of.get(x.number)
        cand = [a for a in ctx.packed.day_items.get(x.number, [])
                if a in by_alias and a not in used]
        cand += [a for a, p in by_alias.items() if place and p.item.place == place
                 and a not in used and a not in cand]
        new = [f"- {lead(by_alias[a].text, 220)} [{a}]" for a in cand[:2]]
        how = "from unused sources of the place"
        if not new and place:
            donors = sorted((y for y in d.days if y is not x and place_of.get(y.number) == place
                             and len(_real(y)) >= 3), key=lambda y: -len(_real(y)))
            if donors:
                moved = _real(donors[0])[-1]
                donors[0].bullets.remove(moved)
                idx = max(i for i, ln in enumerate(donors[0].lines)
                          if (m := BULLET.match(ln)) and m.group(1).strip() == moved)
                del donors[0].lines[idx]
                new, how = [f"- {moved}"], f"by moving a bullet from day {donors[0].number}"
        if not new:
            continue
        used.update(t for line in new for t in ALIAS.findall(line))
        keep = [ln for ln in x.lines[1:]
                if not ((m := BULLET.match(ln)) and is_no_details(m.group(1)))]
        x.lines = [x.lines[0], *new, *keep]
        x.bullets = [n[2:] for n in new] + [b for b in x.bullets if not is_no_details(b)]
        actions.append(f"day {x.number}: filled {how}")
    return render(d), actions


def rebalance_days(draft_text: str, ctx: CriticContext) -> tuple[str, list[str]]:
    """No LLM: a bullet whose tags are all listed for ANOTHER day of the same place goes back to
    that day (the writer was told to use only a day's own tags). Text is moved, never changed."""
    d = parse_draft(draft_text)
    owner: dict[str, int] = {}
    for day in sorted(ctx.packed.day_items):
        for a in ctx.packed.day_items[day]:
            owner.setdefault(a, day)
    place_of = {row.day: row.place for row in ctx.plan.days}
    by_num = {x.number: x for x in d.days}
    actions: list[str] = []
    moves = []
    for x in d.days:
        for b in _real(x):
            tags = [t for t in ALIAS.findall(b) if t.startswith("S")]
            homes = {owner.get(t) for t in tags}
            if not tags or x.number in homes or None in homes or len(homes) != 1:
                continue
            home = by_num.get(homes.pop())
            if home is not None and place_of.get(home.number) == place_of.get(x.number):
                moves.append((x, home, b))
    for x, home, b in moves:
        idx = next(i for i, ln in enumerate(x.lines)
                   if (m := BULLET.match(ln)) and m.group(1).strip() == b)
        line = x.lines.pop(idx)
        x.bullets.remove(b)
        home.lines = [ln for ln in home.lines
                      if not ((m := BULLET.match(ln)) and is_no_details(m.group(1)))]
        home.bullets = [t for t in home.bullets if not is_no_details(t)]
        last = max((i for i, ln in enumerate(home.lines)
                    if (m := BULLET.match(ln)) and not _is_exempt(m.group(1).strip())),
                   default=0)
        home.lines.insert(last + 1, line)
        home.bullets.append(b)
        actions.append(f"day {x.number}: moved a bullet to day {home.number}, which lists its tag")
    return render(d), actions


def repair_days(findings: list[CriticFinding]) -> list[int]:
    return sorted({f.day for f in findings if f.fixable and f.day is not None})


def merge_days(draft_text: str, new_text: str, days: list[int]) -> str:
    """Replace the given day sections of the draft by those of new_text (other days untouched)."""
    d, n = parse_draft(draft_text), parse_draft(new_text)
    fresh = {x.number: x for x in n.days}
    d.days = [fresh[x.number] if x.number in days and x.number in fresh else x for x in d.days]
    return render(d)


def run_cycle(draft_text: str, ctx: CriticContext,
              repair: Callable[[str, list[CriticFinding]], str] | None = None
              ) -> tuple[str, CriticReport, list[str]]:
    """critic -> deterministic fix -> critic -> ONE LLM repair of the offending days -> critic.
    Returns (final draft, report, actions). `repair(draft, findings)` returns replacement day
    sections (or '' to give up); it is called at most once."""
    text = ensure_status_line(normalize_output(draft_text), ctx.request, ctx.live_status)
    text, moved = rebalance_days(text, ctx)
    text, filled = fill_empty_days(text, ctx)
    filled = [*moved, *filled]
    first = check(text, ctx)
    if not first:
        return text, CriticReport(ok=True, status="passed"), filled
    text, actions = deterministic_fix(text, first, ctx.packed, tuple(blocked_names(ctx)))
    text, filled_again = fill_empty_days(text, ctx)
    actions = [*filled, *actions, *filled_again]
    findings = check(text, ctx)
    days = repair_days(findings)
    if findings and days and repair is not None:
        new = repair(text, [f for f in findings if f.day in days])
        if new:
            text = merge_days(text, new, days)
            actions.append(f"LLM repair of day(s) {days}")
            findings = check(text, ctx)
    ok = not findings
    return text, CriticReport(ok=ok, findings=findings,
                              status="repaired" if ok else "failed_rules"), actions


def repair_messages(ctx: CriticContext, draft_text: str, findings: list[CriticFinding],
                    cfg: dict | None = None) -> list[dict]:
    """The prompt of the single repair call: only the offending day sections, the findings and the
    SOURCES, cut to repair.prompt_chars_max by dropping the last sources."""
    from src.agents.evidence import load_pipeline_config
    from src.agents.writer import SYSTEM, evidence_block

    limit = (cfg or load_pipeline_config())["repair"]["prompt_chars_max"]
    days = {f.day for f in findings if f.day is not None}
    sections = [x for x in parse_draft(draft_text).days if x.number in days]
    body = "\n".join("\n".join(x.lines) for x in sections)
    issues = "\n".join(f"- {f.rule}: {f.message}" for f in findings)
    packed = Packed(list(ctx.packed.items), list(ctx.packed.dropped))
    tags = "\n".join(f"Day {n}: " + (", ".join(ctx.packed.day_items.get(n, [])) or "no sources")
                     for n in sorted(days))
    while True:
        user = (f"FIX ONLY these day sections and return them in the same format "
                f"('## Day N: Place' then bullets). Findings:\n{issues}\n\nDAYS TO FIX:\n{body}\n"
                f"\nTAGS PER DAY:\n{tags}\n\nSOURCES:\n{evidence_block(packed)}")
        if len(SYSTEM) + len(user) <= limit or not packed.items:
            break
        packed.items.pop()
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def make_repair(client, model: str, ctx: CriticContext, cfg: dict | None = None
                ) -> Callable[[str, list[CriticFinding]], str]:
    """The repair callable for run_cycle: ONE non-streamed call with the repair options, cached
    unless its prompt holds live text (D109, D115)."""
    from src.agents.evidence import load_pipeline_config
    from src.agents.writer import holds_live_text

    r = (cfg or load_pipeline_config())["repair"]

    def repair(draft_text: str, findings: list[CriticFinding]) -> str:
        msgs = repair_messages(ctx, draft_text, findings, cfg)
        extra = {"cache": False} if holds_live_text(msgs) else {}
        return normalize_output(client.chat(
            msgs, model, options={"max_tokens": r["max_tokens"],
                                  "reasoning_effort": r["reasoning_effort"]}, **extra))

    return repair
