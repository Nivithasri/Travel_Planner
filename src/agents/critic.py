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

from src.agents.evidence import INJECTION, Packed
from src.agents.state import CriticFinding, CriticReport, ParsedRequest, Plan
from src.agents.writer import ALIAS, NO_DETAILS

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
    for line in text.splitlines():
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


def _is_exempt(bullet: str) -> bool:
    return f"- {bullet}".strip() == NO_DETAILS or bullet.startswith("Current status")


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
    allowed = {s.place.casefold() for s in plan.stops} | {n.casefold() for n in req.named_places}
    allowed |= {b["place"].casefold() for b in plan.backups}
    full = "\n".join(render(d).splitlines())
    for name in sorted(ctx.holdout_names | ctx.excluded_names):
        if name.casefold() not in allowed and _mentions(full, name):
            f.append(CriticFinding(rule="R5", message=(
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


def deterministic_fix(draft_text: str, findings: list[CriticFinding], packed: Packed
                      ) -> tuple[str, list[str]]:
    """Strip uncited or wrongly tagged bullets and URLs; give an emptied day the no-details bullet.
    Returns (new text, actions)."""
    d = parse_draft(draft_text)
    amap = packed.alias_map
    actions: list[str] = []
    seen_text: set[str] = set()
    for x in d.days:
        keep_lines, keep_bullets = [], []
        for line in x.lines:
            b = BULLET.match(line)
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
    text = ensure_status_line(draft_text, ctx.request, ctx.live_status)
    first = check(text, ctx)
    if not first:
        return text, CriticReport(ok=True, status="passed"), []
    text, actions = deterministic_fix(text, first, ctx.packed)
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
    while True:
        user = (f"FIX ONLY these day sections and return them in the same format "
                f"('## Day N: Place' then bullets). Findings:\n{issues}\n\nDAYS TO FIX:\n{body}\n"
                f"\nSOURCES:\n{evidence_block(packed)}")
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
        return client.chat(msgs, model, options={"max_tokens": r["max_tokens"],
                                                 "reasoning_effort": r["reasoning_effort"]},
                           **extra)

    return repair
