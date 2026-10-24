"""The three Phase 9 systems and the one record they all return (D123).

llm_only    one write call, the request line and nothing else (no sources, no tags)
plain_rag   the frozen retrieve() on the request line, the first 12 non-blog chunks, one write call
            with chunk tags; no plan, parse, place cards, tools, critic or live
full        the Phase 7 pipeline (run_graph), live off or on

Every system uses the same writer call path (src.agents.writer.write: streamed, temperature 0,
seed, options from config/pipeline.json) and the same blog and injection rules as the full
system's evidence packer, so a difference between systems is not a difference in plumbing.
The record is plain JSON; it holds answer text and source excerpts and therefore stays under
.cache/phase9 (gitignored, O12, D100).
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from eval.phase9 import prompts
from src.agents import writer as W
from src.agents.evidence import (
    Packed,
    PackedItem,
    clean_text,
    injection_flags,
    load_pipeline_config,
)
from src.agents.planner import MONTHS
from src.agents.state import EvidenceItem
from src.common.schemas import REPO_ROOT

ARMS = ("llm_only", "plain_rag", "full_corpus", "full_live")
FETCH_K = 24  # rows asked from retrieve(); blog, injection and budget drops may remove some
MAX_PLAIN_ITEMS = 12
LINE_CHARS = 600


def request_line(req: dict) -> str:
    """The request as the two baselines see it: the text plus the structured fields the full
    system receives as overrides, so no system gets more of the user's intent than another."""
    o = req["overrides"]
    bits = [f"{o['duration_days']} days", f"{o['budget_tier']} budget", str(o["party"])]
    if o.get("month"):
        bits.append(f"travelling in {MONTHS[o['month'] - 1]}")
    if o.get("start_city"):
        bits.append(f"starting in {o['start_city']}")
    if o.get("end_city"):
        bits.append(f"ending in {o['end_city']}")
    if o.get("interests"):
        bits.append("interests: " + ", ".join(o["interests"]))
    if o.get("constraints"):
        bits.append("constraints: " + "; ".join(o["constraints"]))
    if o.get("max_daily_travel_hours"):
        bits.append(f"at most {o['max_daily_travel_hours']:g} hours of travel a day")
    if req.get("named_places"):
        bits.append("places to include: " + ", ".join(req["named_places"]))
    return f"{req['text']} ({'; '.join(bits)})"


def overrides_for(req: dict) -> dict:
    """The overrides handed to the full pipeline (the app form's fields)."""
    o = dict(req["overrides"])
    if req.get("named_places"):
        o["named_places"] = list(req["named_places"])
    return o


def git_state(root: Path = REPO_ROOT) -> dict:
    """The sha of the measured code and whether the tree has uncommitted changes."""
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_sha": sha, "dirty": bool(status) if status is not None else None}


def usage_total(client) -> int:
    usage = getattr(client, "usage", None) or {}
    return int(usage.get("total", 0))


def make_record(system: str, arm: str, req: dict, *, text: str, citations: list[dict],
                contexts: list[dict], tokens: dict, timings: dict, model: str,
                live: dict | None = None, critic: str | None = None,
                extra: dict | None = None, now: datetime | None = None,
                git: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    meta = {**(git if git is not None else git_state()), "model": model,
            "utc_date": now.strftime("%Y-%m-%d"), "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt_sha": prompts.prompt_sha(system) if system in prompts.PROMPTS else None}
    meta.update(extra or {})
    return {"system": system, "arm": arm, "request_id": req["id"], "text": text,
            "citations": citations, "contexts": contexts, "tokens": tokens,
            "timings": timings, "live": live, "critic": critic, "meta": meta}


def _context(p: PackedItem) -> dict:
    it = p.item
    return {"alias": p.alias, "chunk_id": it.chunk_id, "place": it.place,
            "source_type": it.source_type, "url": it.url, "text": p.text}


def _citation(p: PackedItem, cited: bool = True) -> dict:
    it = p.item
    return {"alias": p.alias, "chunk_id": it.chunk_id, "place": it.place,
            "source_type": it.source_type, "url": it.url, "live": it.is_live,
            "fetch_date": it.fetch_date, "cited": cited}


# ---- llm only ---------------------------------------------------------------------------------

def run_llm_only(req: dict, client, model: str, cfg: dict | None = None, *,
                 clock: Callable[[], float] = time.monotonic, git: dict | None = None) -> dict:
    cfg = cfg or load_pipeline_config()
    line = clean_text(request_line(req), LINE_CHARS)
    messages = [{"role": "system", "content": prompts.LLM_ONLY_SYSTEM},
                {"role": "user", "content": f"REQUEST: {line}"}]
    before, t0 = usage_total(client), clock()
    text = W.write(client, model, messages, cfg)
    timings = {"write": round(clock() - t0, 3)}
    tokens = {"total": usage_total(client) - before}
    return make_record("llm_only", "llm_only", req, text=text.strip() + "\n", citations=[],
                       contexts=[], tokens=tokens, timings=timings, model=model, git=git)


# ---- plain rag --------------------------------------------------------------------------------

def pack_plain(rows: list[dict], cfg: dict, max_items: int = MAX_PLAIN_ITEMS) -> Packed:
    """The first `max_items` usable rows: no blog (D103), no instruction-like text, cleaned and
    cut like the full system's packer, aliased S1.. in retrieval order."""
    w = cfg["writer"]
    out, seen = Packed(), set()
    for r in rows:
        if len(out.items) >= max_items:
            break
        cid = r["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        if r.get("source_type") == "blog" and not w["allow_blog"]:
            out.dropped.append({"chunk_id": cid, "reason": "blog_excluded"})
            continue
        flags = injection_flags(r.get("text", ""))
        if flags:
            out.dropped.append({"chunk_id": cid, "reason": "injection_pattern", "flags": flags})
            continue
        text = clean_text(r.get("text", ""), w["item_chars"])
        if not text:
            continue
        item = EvidenceItem(
            chunk_id=cid, place=r.get("place") or "", source_type=r.get("source_type") or "",
            url=r.get("url") or "", text=r.get("text", ""), aspect=r.get("aspect"),
            place_type=r.get("place_type"), is_warning=bool(r.get("is_warning")),
            provenance="unknown", date=r.get("date"), score=float(r.get("score", 0.0)))
        out.items.append(PackedItem(f"S{len(out.items) + 1}", item, text))
    return out


def plain_messages(line: str, packed: Packed, cfg: dict) -> list[dict]:
    """[system, user]; the last items are dropped (and recorded) until the prompt fits the same
    character and reservation limits the full system's writer obeys."""
    w = cfg["writer"]
    while True:
        user = (f"REQUEST: {clean_text(line, LINE_CHARS)}\n"
                f"SOURCES:\n{W.evidence_block(packed)}")
        messages = [{"role": "system", "content": prompts.PLAIN_RAG_SYSTEM},
                    {"role": "user", "content": user}]
        fits = (len(prompts.PLAIN_RAG_SYSTEM) + len(user) <= w["prompt_chars_max"]
                and W.reservation_tokens(messages, w["max_tokens"]) <= w["reservation_max_tokens"])
        if fits or not packed.items:
            return messages
        gone = packed.items.pop()
        packed.dropped.append({"chunk_id": gone.item.chunk_id, "reason": "prompt_budget"})


def run_plain_rag(req: dict, retrieve: Callable, client, model: str, cfg: dict | None = None, *,
                  clock: Callable[[], float] = time.monotonic, git: dict | None = None) -> dict:
    cfg = cfg or load_pipeline_config()
    line = request_line(req)
    t0 = clock()
    rows = retrieve(line, k=FETCH_K)
    t_retrieve = clock() - t0
    packed = pack_plain(list(rows), cfg)
    messages = plain_messages(line, packed, cfg)
    before, t1 = usage_total(client), clock()
    draft = W.write(client, model, messages, cfg)
    timings = {"retrieve": round(t_retrieve, 3), "write": round(clock() - t1, 3)}
    cited = list(dict.fromkeys(W.citations(draft)))
    citations = [_citation(p) for a in cited if (p := packed.by_alias(a)) is not None]
    text = W.expand_aliases(draft, packed).strip() + "\n"
    return make_record("plain_rag", "plain_rag", req, text=text, citations=citations,
                       contexts=[_context(p) for p in packed.items],
                       tokens={"total": usage_total(client) - before}, timings=timings,
                       model=model, extra={"dropped": len(packed.dropped)}, git=git)


# ---- full system ------------------------------------------------------------------------------

def run_full(req: dict, deps, runner: Callable, *, live: bool,
             clock: Callable[[], float] = time.monotonic, git: dict | None = None) -> dict:
    """One full-system run. `deps` was built with live on or off to match `live`; for a live run
    the retriever is wrapped so counts of searches, results and in-scope items exist."""
    from app import state as A

    trace = A.watch_live(deps) if live else None
    before, t0 = usage_total(deps.client), clock()
    state = runner(deps, req["text"], overrides_for(req))
    wall = clock() - t0
    view = A.view_from_state(state, wall, trace, use_live=live)
    final = state.final or {}
    cited_ids = {s["chunk_id"] for s in final.get("sources", [])}
    citations = [{"alias": s.get("alias"), "chunk_id": s["chunk_id"], "place": s.get("place"),
                  "source_type": s.get("source_type"), "url": s.get("url"),
                  "live": bool(s.get("live")), "fetch_date": s.get("fetch_date"), "cited": True}
                 for s in final.get("sources", [])]
    citations += [{**{k: u.get(k) for k in ("alias", "chunk_id", "place", "source_type", "url",
                                            "fetch_date")},
                   "live": True, "cited": False}
                  for u in final.get("uncited_live", []) if u["chunk_id"] not in cited_ids]
    packed = getattr(state, "packed_obj", None)
    contexts = [_context(p) for p in getattr(packed, "items", []) if not p.item.is_live]
    live_info = None
    if live:
        live_info = {"status": state.live_status, **(view.get("live_counts") or {}),
                     "note": state.live_note}
    tokens = {**{k: int(v) for k, v in state.tokens.items()},
              "total": usage_total(deps.client) - before}
    critic = (final.get("critic") or {}).get("status")
    arm = "full_live" if live else "full_corpus"
    return make_record("full", arm, req, text=final.get("text", ""), citations=citations,
                       contexts=contexts, tokens=tokens, timings=dict(state.timings),
                       model=getattr(deps, "model", ""), live=live_info, critic=critic,
                       extra={"wall_s": round(wall, 3), "repaired": bool(state.repaired)},
                       git=git)
