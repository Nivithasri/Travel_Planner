"""The pipeline nodes (Phase 7, D99 to D104). Each node is a plain function `node(state, deps) ->
dict` returning ONLY the keys it changes, so the same functions run in LangGraph (with reducers for
warnings, timings and tokens) and in the local sequential fallback (graph.run_local).

parse -> (retrieve || live) -> merge_grade -> plan -> write -> critic -> finalize
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from src.agents import critic as C
from src.agents import evidence as E
from src.agents import writer as W
from src.agents.planner import Candidate, plan_trip
from src.agents.scope import foreign_place_matcher, out_of_scope_matcher
from src.agents.state import (
    CriticFinding,
    CriticReport,
    EvidenceItem,
    RunState,
    request_from_parse,
)
from src.live.search import NOTES as LIVE_NOTES
from src.places.places import normalize
from src.retrieval import grader
from src.tools.routing import UnknownPlaceError

DOC_NOTES = {
    "provisional": ("Travel times come from a provisional road model fitted on unverified "
                    "reference routes (D77): treat them as planning estimates, not timetables."),
    "freshness": ("The corpus has no page published in the last 18 months (wiki chunks carry "
                  "revision dates only): nothing taken from it is evidence of current conditions."),
    "budget": "The budget uses assumed tier figures, not corpus data (D101).",
}


@dataclass
class Deps:
    registry_entries: list
    excluded: frozenset[str]
    cards: dict[str, dict]
    provenance: dict[str, str]
    retrieve: Callable[[str, dict | None, int], list[dict]]
    hours: Callable[[str, str], float]
    season: Callable[[str, int], dict]
    planner_cfg: dict
    pipeline_cfg: dict
    client: Any = None  # chat_stream / chat (OpenAICompatClient or a fake)
    model: str = ""
    parse: Callable[[str], Any] | None = None  # text -> ValidatedQuery-like (LLM)
    live: Any = None  # LiveRetriever or None
    budget_cfg: dict | None = None
    on_piece: Callable[[str], None] | None = None
    tokens_used: Callable[[], int] = lambda: 0  # noqa: E731
    clock: Callable[[], float] = time.monotonic
    notes: dict = field(default_factory=dict)

    @property
    def place_names(self) -> list[str]:
        return [e.name for e in self.registry_entries if not e.holdout]

    @property
    def holdout_names(self) -> frozenset[str]:
        out = set()
        for e in self.registry_entries:
            if e.holdout:
                out.update([e.name, *e.aliases])
        return frozenset(out)

    @property
    def excluded_labels(self) -> frozenset[str]:
        out = set()
        for e in self.registry_entries:
            if e.name in self.excluded:
                out.update([e.name, *[a for a in e.aliases if len(a) >= 4]])
        return frozenset(out)


def apply_update(state: RunState, upd: dict) -> None:
    """Merge a node's update: dicts (timings, tokens) merge, warnings extend, the rest replace."""
    for k, v in upd.items():
        if k in ("timings", "tokens"):
            getattr(state, k).update(v)
        elif k == "warnings":
            state.warnings.extend(v)
        else:
            setattr(state, k, v)


def names_in(text: str, entries: list, *, holdout: bool) -> list[str]:
    hay = f" {normalize(text)} "
    out = []
    for e in entries:
        if bool(e.holdout) != holdout:
            continue
        for form in (e.name, *e.aliases):
            n = normalize(form)
            if len(n) >= 4 and f" {n} " in hay:
                out.append(e.name)
                break
    return out


# ---- parse ----------------------------------------------------------------------------------

def parse_node(state: RunState, deps: Deps) -> dict:
    text = state.input_text
    o = dict(state.overrides)
    warnings: list[str] = []
    held = names_in(text, deps.registry_entries, holdout=True)
    if held:
        warnings.append(f"{', '.join(held)} is not covered by this planner and was ignored")
    used = {}
    if deps.parse is not None and not o.get("no_llm"):
        before = deps.tokens_used()
        parsed = deps.parse(text)
        used = {"parse": deps.tokens_used() - before}
    else:
        named = names_in(text, deps.registry_entries, holdout=False)
        parsed = SimpleNamespace(text=text, places=named, states=[], month=None,
                                 duration_days=None, budget_tier=None, travel_style=None,
                                 interests=[], place_types=[], time_sensitive=False, notes=[])
    o.pop("no_llm", None)
    o.setdefault("text", text)
    req = request_from_parse(parsed, overrides=o)
    return {"request": req, "warnings": warnings, "tokens": used}


# ---- retrieve -------------------------------------------------------------------------------

def item_from_row(row: dict, provenance: dict[str, str]) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=row["chunk_id"], place=row["place"], source_type=row["source_type"],
        url=row["url"], text=row["text"], aspect=row.get("aspect"),
        place_type=row.get("place_type"), is_warning=bool(row.get("is_warning")),
        provenance=provenance.get(row["chunk_id"], "unknown"), date=row.get("date"),
        score=float(row.get("score") or 0.0))


def retrieve_node(state: RunState, deps: Deps) -> dict:
    req = state.request
    rcfg = deps.planner_cfg["retrieve"]
    filters = {"state": req.states} if req.states else None
    rows = deps.retrieve(req.text or " ".join(req.interests) or "travel in South India",
                         filters, rcfg["k_request"])
    named = set(req.named_places)
    eligible = {n for n, c in deps.cards.items() if c["evidence_class"] == "full"} | named
    hits: dict[str, list[dict]] = {}
    for r in rows:
        if r["place"] in eligible:
            hits.setdefault(r["place"], []).append(r)
    mass = {p: sum(float(r.get("score") or 0.0) for r in rs) for p, rs in hits.items()}
    top = max(mass.values(), default=0.0) or 1.0
    words = [w.casefold() for i in req.interests for w in re.findall(r"[A-Za-z]{4,}", i)]
    cands = []
    for p in sorted(eligible | named, key=lambda x: (-mass.get(x, 0.0), x)):
        rs = hits.get(p, [])
        match = sum(1 for r in rs if r.get("place_type") in req.place_types
                    or any(w in r["text"].casefold() for w in words))
        card = deps.cards.get(p)
        if card is None:
            continue
        cands.append({"name": p, "state": card["state"],
                      "relevance": round(mass.get(p, 0) / top, 4),
                      "interest": round(match / max(len(rs), 1), 4), "named": p in named})
    # every candidate the planner may pick needs evidence (they are few and the queries are local)
    shortlist = [c["name"] for c in cands]
    evidence: dict[str, list[EvidenceItem]] = {}
    for p in shortlist:
        seen: dict[str, EvidenceItem] = {}
        card = deps.cards.get(p, {})
        for w in card.get("warnings", []):
            it = item_from_row({**w, "place": p, "is_warning": True, "score": 0.9},
                               deps.provenance)
            seen.setdefault(it.chunk_id, it)
        sub = deps.retrieve(f"{p} {' '.join(req.interests)}".strip(), None, rcfg["k_place"])
        for r in [*hits.get(p, []), *[x for x in sub if x["place"] == p]]:
            seen.setdefault(r["chunk_id"], item_from_row(r, deps.provenance))
        for t in card.get("tips", []):
            seen.setdefault(t["chunk_id"], item_from_row({**t, "place": p, "score": 0.3},
                                                         deps.provenance))
        if req.interests and not any(E.interest_match(i, req.interests) for i in seen.values()):
            # the interests match nothing retrieved for this place: add its main sights (D131)
            for r in deps.retrieve(f"{p} main sights and attractions", None, rcfg["k_place"]):
                if r["place"] == p:
                    seen.setdefault(r["chunk_id"], item_from_row(r, deps.provenance))
        evidence[p] = list(seen.values())
    return {"candidates": cands, "corpus_evidence": evidence}


# ---- live -----------------------------------------------------------------------------------

def live_node(state: RunState, deps: Deps) -> dict:
    req = state.request
    if deps.live is None:
        return {"live_status": "disabled", "live_note": LIVE_NOTES["not_requested"]}
    res = deps.live.run(list(req.named_places), list(req.states), req.month)
    upd: dict = {"live_evidence": res.items, "live_status": res.status, "live_note": res.note,
                 "tokens": {"live_credits": res.credits}}
    if res.note:
        upd["warnings"] = [res.note]
    return upd


# ---- merge and grade ------------------------------------------------------------------------

def merge_grade_node(state: RunState, deps: Deps) -> dict:
    g = deps.planner_cfg["evidence_grade"]
    grades, warnings = {}, []
    evidence = {p: list(v) for p, v in state.corpus_evidence.items()}
    rcfg = deps.planner_cfg["retrieve"]
    for p, items in state.corpus_evidence.items():
        llm = [i for i in items if i.provenance == "llm"]
        urls = {i.url for i in items}
        thin = len(llm) < g["min_model_written"] or len(urls) < g["min_distinct_urls"]
        top5 = [{"place": i.place, "text": i.text, "url": i.url} for i in items[:5]]
        contradiction = grader.contradiction(top5)
        if thin and deps.cards.get(p, {}).get("evidence_class") == "full":
            seen = {i.chunk_id for i in items}
            extra = deps.retrieve(f"{p} travel", None, rcfg["k_thin"])
            for r in extra:
                if r["place"] == p and r["chunk_id"] not in seen:
                    evidence[p].append(item_from_row(r, deps.provenance))
                    seen.add(r["chunk_id"])
            llm = [i for i in evidence[p] if i.provenance == "llm"]
            urls = {i.url for i in evidence[p]}
            thin = len(llm) < g["min_model_written"] or len(urls) < g["min_distinct_urls"]
        grades[p] = {"thin": thin, "model_written": len(llm), "distinct_urls": len(urls),
                     "contradiction": contradiction}
        if thin:
            warnings.append(f"thin evidence for {p}: {len(llm)} model-written chunk(s), "
                            f"{len(urls)} source page(s)")
        if contradiction:
            warnings.append(f"sources for {p} may contradict each other on open/closed status")
    return {"corpus_evidence": evidence, "grades": grades, "warnings": warnings}


# ---- plan -----------------------------------------------------------------------------------

def plan_node(state: RunState, deps: Deps) -> dict:
    req = state.request.model_copy(deep=True)
    cands = [Candidate(c["name"], c["state"], c["relevance"], c["interest"], c["named"],
                       deps.cards.get(c["name"], {})) for c in state.candidates]
    warnings = []
    for field_name in ("start_city", "end_city"):
        city = getattr(req, field_name)
        if city:
            try:
                deps.hours(city, cands[0].name if cands else city)
            except UnknownPlaceError as exc:
                warnings.append(f"{field_name.replace('_', ' ')} ignored: {exc}")
                setattr(req, field_name, None)
    plan = plan_trip(req, cands, hours=deps.hours, season=deps.season, cfg=deps.planner_cfg,
                     budget_cfg=deps.budget_cfg)
    return {"plan": plan, "request": req, "warnings": warnings}


# ---- write ----------------------------------------------------------------------------------

def write_node(state: RunState, deps: Deps) -> dict:
    plan, req = state.plan, state.request
    if not plan.feasible:
        return {"draft": "", "packed": {}, "packed_obj": E.Packed()}
    stop_ev = {s.place: state.corpus_evidence.get(s.place, []) for s in plan.stops}
    allowed = {s.place for s in plan.stops} | set(req.named_places)
    packed = E.pack(plan, stop_ev, state.live_evidence, deps.pipeline_cfg,
                    interests=req.interests, named_places=req.named_places,
                    scope_check=out_of_scope_matcher(deps.registry_entries, deps.excluded, allowed),
                    constraints=req.constraints,
                    elsewhere=foreign_place_matcher(deps.registry_entries))
    messages = W.build_messages(req, plan, packed, deps.pipeline_cfg)
    before = deps.tokens_used()
    draft = W.write(deps.client, deps.model, messages, deps.pipeline_cfg, deps.on_piece)
    return {"draft": draft, "packed": packed.alias_map, "packed_obj": packed,
            "tokens": {"write": deps.tokens_used() - before,
                       "write_reservation": W.reservation_tokens(
                           messages, deps.pipeline_cfg["writer"]["max_tokens"])}}


# ---- critic ---------------------------------------------------------------------------------

def critic_node(state: RunState, deps: Deps) -> dict:
    plan = state.plan
    if not plan.feasible:
        rep = CriticReport(ok=False, status="failed_rules", findings=[CriticFinding(
            rule="R1", fixable=False, message=f"the plan is infeasible: {plan.infeasible_reason}")])
        return {"critic": rep}
    ctx = C.CriticContext(state.request, plan, state.packed_obj, state.live_status,
                          deps.excluded_labels, deps.holdout_names)
    before = deps.tokens_used()
    repair = C.make_repair(deps.client, deps.model, ctx, deps.pipeline_cfg) if deps.client else None
    text, report, actions = C.run_cycle(state.draft, ctx, repair)
    return {"draft": text, "critic": report, "repaired": report.status == "repaired",
            "warnings": [f"critic: {a}" for a in actions],
            "tokens": {"critic": deps.tokens_used() - before}}


# ---- finalize -------------------------------------------------------------------------------

def sources_used(draft: str, packed) -> list[dict]:
    cited = list(dict.fromkeys(W.citations(draft)))
    out = []
    for alias in cited:
        p = packed.by_alias(alias) if packed else None
        if p is None:
            continue
        it = p.item
        out.append({"alias": alias, "chunk_id": it.chunk_id, "place": it.place,
                    "source_type": it.source_type, "url": it.url, "live": it.is_live,
                    "fetch_date": it.fetch_date, "date": it.date, "provenance": it.provenance,
                    "limited": p.limited})
    return out


def finalize_node(state: RunState, deps: Deps) -> dict:
    plan, req = state.plan, state.request
    lines = ["# South India itinerary"]
    if not plan.feasible:
        lines += ["", f"No feasible itinerary: {plan.infeasible_reason}.",
                  "Try more days, a nearer start city or fewer constraints."]
        final = {"text": "\n".join(lines) + "\n", "sources": [], "plan": plan.model_dump(),
                 "critic": state.critic.model_dump() if state.critic else None,
                 "live_status": state.live_status, "warnings": state.warnings}
        return {"final": final}
    sources = sources_used(state.draft, state.packed_obj)
    body = W.expand_aliases(state.draft, state.packed_obj)
    lines += ["", f"{req.duration_days} days, {req.budget_tier} budget, {req.party}.", "",
              "## Route, season and budget (computed)", *[f"- {x}" for x in plan.computed_lines],
              "", body.strip(), "", "## Sources"]
    for s in sources:
        tag = f"[{s['chunk_id']}]" if s["live"] else f"[chunk:{s['chunk_id']}]"
        when = (f"live, fetched {s['fetch_date']}" if s["live"]
                else "no current date (wiki revision or undated)")
        lines.append(f"- {tag} {s['place'] or 'region'} ({s['source_type']}; {when}) {s['url']}")
    notes = [DOC_NOTES["provisional"], DOC_NOTES["freshness"], DOC_NOTES["budget"], *plan.notes,
             *state.warnings]
    if state.live_note:
        notes.append(state.live_note)
    if state.critic and state.critic.status != "passed":
        notes.append(f"critic status: {state.critic.status}")
        notes += [f"critic {f.rule}: {f.message}" for f in state.critic.findings]
    dropped = state.packed_obj.dropped if state.packed_obj else []
    if any(d["reason"] == "blog_excluded" for d in dropped):
        notes.append("blog sources were left out of the writer's context (no terms)")
    n_live_out = sum(1 for d in dropped if d["reason"] in ("live_out_of_plan", "live_unresolved"))
    n_live_scope = sum(1 for d in dropped if d["reason"] == "live_names_excluded")
    if n_live_scope:
        notes.append(f"{n_live_scope} live search result(s) were left out because they name a "
                     "place outside the project scope")
    if n_live_out:
        notes.append(f"{n_live_out} live search result(s) were left out because they are not about "
                     "a place in your plan")
    lines += ["", "## Notes and warnings", *[f"- {n}" for n in dict.fromkeys(notes)]]
    cited = {s["alias"] for s in sources}
    uncited_live = [
        {"alias": p.alias, "chunk_id": p.item.chunk_id, "place": p.item.place,
         "source_type": p.item.source_type, "url": p.item.url, "live": True,
         "fetch_date": p.item.fetch_date, "date": p.item.date, "provenance": p.item.provenance,
         "limited": False}
        for p in (state.packed_obj.items if state.packed_obj else [])
        if p.item.is_live and p.alias not in cited]
    final = {"text": "\n".join(lines) + "\n", "sources": sources, "plan": plan.model_dump(),
             "critic": state.critic.model_dump() if state.critic else None,
             "live_status": state.live_status, "warnings": state.warnings,
             "dropped": dropped, "request": req.model_dump(), "uncited_live": uncited_live}
    return {"final": final}


NODES = {"parse": parse_node, "retrieve": retrieve_node, "live": live_node,
         "merge_grade": merge_grade_node, "plan": plan_node, "write": write_node,
         "critic": critic_node, "finalize": finalize_node}
