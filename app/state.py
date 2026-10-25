"""Logic of the Phase 8 page, kept free of Streamlit so it can be tested headless (D110).

Form -> text and overrides (free text never reaches the live search: only registry-resolved place
names do), the run lock and the streaming worker, the per-process Deps cache, redaction of
anything shown on the page, and the row builders for the sources, plan and timing panels.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.live.config import TERMS_NOTICE
from src.live.search import NOTES as LIVE_NOTES

ROOT = Path(__file__).resolve().parents[1]

BUDGET_TIERS = ("budget", "mid", "luxury")
PARTIES = ("solo", "couple", "family", "group")
INTEREST_CHOICES = ("temples", "heritage", "history", "culture", "beaches", "hill stations",
                    "wildlife", "backwaters", "waterfalls", "nature", "food")
MAX_FREE_CHARS = 200
LIVE_RAN = ("ok", "empty", "error")
BUSY_MESSAGE = "another run is in progress"
LIVE_CREDITS_HINT = "uses search credits (assumed 2 per search, not measured)"
PRIVACY_HINT = "do not enter personal information"
RECENCY_NOTE = ("no corpus item is dated within the last 18 months; live items show their fetch "
                "date, not a publish date")
SEASON_NOTE = "trip spans several months; season check uses the start month only"
NOT_REQUESTED_NOTE = LIVE_NOTES["not_requested"]  # the pipeline now says it itself (D115)
CARD_LABEL = "auto-generated, check before relying"
BUDGET_LABEL = "rough estimate, not verified"
TRAVEL_LABEL = "provisional"
CC_BY_SA = ("Wikipedia and Wikivoyage text: CC BY-SA (Wikimedia contributors; see each page's "
            "history for authors)")
FALLBACK_ATTRIBUTION = ("Climate: Weather data by Open-Meteo.com (CC BY 4.0). Airports: "
                        "OpenStreetMap contributors (ODbL).")


# ---- redaction ---------------------------------------------------------------------------------

_KEY_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bgsk_[A-Za-z0-9]+"),
    re.compile(r"\btvly-[A-Za-z0-9_-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(authorization|x-api-key|api-key)\s*[:=]\s*\S+"),
)
_ENV_SECRETS = ("LLM_API_KEY", "LIVE_SEARCH_API_KEY")
_ENV_URLS = ("LLM_API_BASE_URL", "LLM_API_URL", "OLLAMA_BASE_URL")
_env_loaded = False


def _load_env() -> None:
    global _env_loaded
    if not _env_loaded:
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env", override=False)
        except Exception:  # noqa: BLE001 - the page must work without python-dotenv
            pass
        _env_loaded = True


def secret_values() -> list[str]:
    """Key values and API hosts that must never be shown. Read from the environment only."""
    _load_env()
    out = [os.environ.get(n, "").strip() for n in _ENV_SECRETS]
    for n in _ENV_URLS:
        raw = os.environ.get(n, "").strip()
        if raw:
            out.append(raw)
            host = urlparse(raw).netloc
            if host:
                out.append(host)
    try:
        host = json.loads((ROOT / "config" / "live.json").read_text(encoding="utf-8")
                          ).get("endpoint_host", "")
        out.append(host)
    except (OSError, ValueError):
        pass
    out.append("api.tavily.com")
    return [s for s in dict.fromkeys(out) if s]


def redact(text: str, secrets: list[str] | None = None) -> str:
    """Replace key-like strings, key values and API hosts with <redacted>. Ordinary links to
    pages (Wikipedia, blogs, live results) are kept. Never raises."""
    try:
        text = str(text)
        for host in sorted(secrets if secrets is not None else secret_values(), key=len,
                           reverse=True):
            if host:
                text = re.sub(rf"https?://{re.escape(host)}[^\s)\]>\"']*", "<redacted>", text)
                text = text.replace(host, "<redacted>")
        for pat in _KEY_PATTERNS:
            text = pat.sub("<redacted>", text)
        return text
    except Exception:  # noqa: BLE001
        return "<redacted>"


def redact_deep(obj: Any, secrets: list[str] | None = None) -> Any:
    secrets = secrets if secrets is not None else secret_values()
    if isinstance(obj, str):
        return redact(obj, secrets)
    if isinstance(obj, dict):
        return {k: redact_deep(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_deep(v, secrets) for v in obj]
    return obj


def safe_error(exc: BaseException) -> str:
    """A short message for the page: redacted, no traceback, no host."""
    from src.common.llm import QuotaExhausted, scrub_text

    if isinstance(exc, QuotaExhausted):
        return "the language-model quota is used up for now; try again after the daily reset"
    body = scrub_text(str(exc), secret_values())[:300]
    return f"{type(exc).__name__}: {body}" if body else type(exc).__name__


# ---- registry, cards, attribution (read-only loaders) ------------------------------------------

def load_registry():
    from src.retrieval.query_understanding import load_core_registry

    return load_core_registry()


def load_excluded() -> frozenset[str]:
    try:
        doc = json.loads((ROOT / "data" / "manifests" / "d33_excluded.json"
                          ).read_text(encoding="utf-8"))
        return frozenset(doc["excluded"])
    except (OSError, ValueError, KeyError):
        return frozenset()


def load_cards() -> dict:
    try:
        from src.tools.place_cards import CARDS_PATH

        return json.loads(Path(CARDS_PATH).read_text(encoding="utf-8"))["places"]
    except (OSError, ValueError, KeyError, ImportError):
        return {}


def load_attribution() -> dict:
    try:
        return json.loads((ROOT / "data" / "processed" / "attribution.json"
                           ).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def footer_attribution() -> str:
    """The attribution strings of the routing and season files plus the wiki licence line."""
    parts = []
    for name in ("travel_matrix.json", "season_data.json"):
        try:
            meta = json.loads((ROOT / "index" / name).read_text(encoding="utf-8"))["meta"]
            if meta.get("attribution"):
                parts.append(meta["attribution"])
        except (OSError, ValueError, KeyError):
            continue
    body = " ".join(dict.fromkeys(parts)) or FALLBACK_ATTRIBUTION
    return f"{CC_BY_SA}. {body}"


def live_availability() -> dict:
    """Whether the live layer could make a call: key present (never its value), config flags."""
    info = {"key_present": False, "enabled": False, "terms_status": "unknown"}
    try:
        from src.common.schemas import load_live_settings
        from src.live.config import load_live_config

        cfg = load_live_config()
        info["key_present"] = bool(load_live_settings().api_key)
        info["enabled"] = bool(cfg.enabled)
        info["terms_status"] = cfg.terms_status
    except Exception:  # noqa: BLE001
        pass
    return info


# ---- form -> request -----------------------------------------------------------------------------

@dataclass
class PlanForm:
    duration_days: int = 4
    budget_tier: str = "mid"
    party: str = "group"
    travel_style: str = ""
    start_city: str = ""
    end_city: str = ""
    start_date: date | None = None
    end_date: date | None = None
    interests: list[str] = field(default_factory=list)
    interests_free: str = ""
    constraints: str = ""
    named_places: list[str] = field(default_factory=list)
    named_free: str = ""
    max_daily_travel_hours: float | None = None
    use_live: bool = False
    force_refresh: bool = False


@dataclass
class Prepared:
    text: str = ""
    overrides: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def split_free(value: str) -> list[str]:
    parts = re.split(r"[,;\n]", (value or "")[:MAX_FREE_CHARS * 2])
    return [p.strip()[:MAX_FREE_CHARS] for p in parts if p.strip()]


def trip_month(start: date | None, end: date | None) -> tuple[int | None, str | None]:
    """The start month drives check_season; a trip across months is noted (D110)."""
    if start is None:
        return None, None
    note = None
    if end is not None and (end.year, end.month) != (start.year, start.month):
        note = SEASON_NOTE
    return start.month, note


def resolve_place(text: str, registry) -> tuple[str, str | None]:
    """('ok', canonical name) | ('holdout', name) | ('unknown', None). Registry names and aliases
    only; this is the gate in front of the live query builder (D100)."""
    found = registry.find(text)
    if len(found) != 1:
        return "unknown", None
    entry = found[0]
    return ("holdout" if entry.holdout else "ok"), entry.name


def check_city(name: str, resolve=None) -> str | None:
    """None when the city is a destination or a hub of the routing table, else a message."""
    name = (name or "").strip()
    if not name:
        return None
    if resolve is None:
        from src.tools.travel_time import load_matrix

        resolve = load_matrix().resolve
    from src.tools.routing import UnknownPlaceError

    try:
        resolve(name)
    except UnknownPlaceError as exc:
        hint = f" Did you mean: {', '.join(exc.suggestions)}?" if exc.suggestions else ""
        return (f"'{name[:60]}' is not a destination or hub of this planner's routing table."
                f"{hint}")
    return None


def prepare(form: PlanForm, registry, resolve_city=None) -> Prepared:
    out = Prepared()
    for label, value in (("Start city", form.start_city), ("End city", form.end_city)):
        msg = check_city(value, resolve_city)
        if msg:
            out.errors.append(f"{label}: {msg}")
    month, note = trip_month(form.start_date, form.end_date)
    if note:
        out.notes.append(note)
    if form.start_date and form.end_date and form.end_date < form.start_date:
        out.errors.append("The end date is before the start date.")
    places: list[str] = []
    for raw in [*form.named_places, *split_free(form.named_free)]:
        status, name = resolve_place(raw, registry)
        if status == "ok":
            if name not in places:
                places.append(name)
        elif status == "holdout":
            out.notes.append(f"'{raw[:60]}' is not covered by this planner and was ignored")
        else:
            out.notes.append(f"'{raw[:60]}' is not a known place; it was ignored")
    interests = [*form.interests, *split_free(form.interests_free)]
    constraints = split_free(form.constraints)
    style = (form.travel_style or "").strip()[:MAX_FREE_CHARS]
    sentence = [f"{form.duration_days} day South India trip, {form.budget_tier} budget."]
    if style:
        sentence.append(f"Style: {style}.")
    if interests:
        sentence.append(f"Interests: {', '.join(interests)}.")
    if places:
        sentence.append(f"Places: {', '.join(places)}.")
    if constraints:
        sentence.append(f"Constraints: {', '.join(constraints)}.")
    out.text = " ".join(sentence)
    o: dict = {"duration_days": int(form.duration_days), "budget_tier": form.budget_tier,
               "party": form.party, "interests": interests, "constraints": constraints,
               "named_places": places}
    if month is not None:
        o["month"] = month
    if form.start_city.strip():
        o["start_city"] = form.start_city.strip()
    if form.end_city.strip():
        o["end_city"] = form.end_city.strip()
    if form.max_daily_travel_hours:
        o["max_daily_travel_hours"] = float(form.max_daily_travel_hours)
    out.overrides = o
    return out


# ---- Deps cache ----------------------------------------------------------------------------------

_DEPS: dict[bool, Any] = {}
_DEPS_LOCK = threading.Lock()


def default_factory(**kwargs):
    from src.agents.deps import build_default_deps

    return build_default_deps(**kwargs)


def get_deps(use_live: bool, factory=None):
    """One Deps per live setting for the whole process (the local model is loaded once, on the
    first retrieve). on_piece is set per run, not here."""
    k = bool(use_live)
    with _DEPS_LOCK:
        if k not in _DEPS:
            _DEPS[k] = (factory or default_factory)(live=k, parse=True, on_piece=None)
        return _DEPS[k]


def reset_deps() -> None:
    with _DEPS_LOCK:
        _DEPS.clear()


# ---- run lock and worker -------------------------------------------------------------------------

class RunBusy(RuntimeError):
    pass


_RUN_LOCK = threading.Lock()


def run_lock_held() -> bool:
    return _RUN_LOCK.locked()


@dataclass
class RunHandle:
    pieces: queue.Queue = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    state: Any = None
    error: str | None = None
    quota: bool = False
    wall_s: float = 0.0
    thread: threading.Thread | None = None
    trace: Any = None

    def drain(self) -> str:
        out = []
        while True:
            try:
                out.append(self.pieces.get_nowait())
            except queue.Empty:
                return "".join(out)


def default_runner():
    """run_graph (the CLI default, LangGraph); run_local when langgraph cannot be imported."""
    try:
        import langgraph  # noqa: F401
    except ImportError:
        from src.agents.graph import run_local

        return run_local
    from src.agents.graph import run_graph

    return run_graph


def _worker(handle: RunHandle, deps, text: str, overrides: dict, runner) -> None:
    t0 = time.monotonic()
    try:
        handle.state = runner(deps, text, dict(overrides))
    except BaseException as exc:  # noqa: BLE001 - shown on the page, redacted
        from src.common.llm import QuotaExhausted

        handle.quota = isinstance(exc, QuotaExhausted)
        handle.error = safe_error(exc)
    finally:
        handle.wall_s = round(time.monotonic() - t0, 2)
        try:
            deps.on_piece = None
        finally:
            _RUN_LOCK.release()
            handle.done.set()


def start_run(deps, text: str, overrides: dict, runner=None) -> RunHandle:
    """Start the pipeline in a worker thread. Only one run at a time (the LLM rate limiter is
    sequential); the lock is released by the worker in a finally, so an exception or an
    interrupted page rerun cannot leave it held."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise RunBusy(BUSY_MESSAGE)
    try:
        handle = RunHandle()
        handle.trace = watch_live(deps)
        deps.on_piece = handle.pieces.put
        handle.thread = threading.Thread(
            target=_worker, args=(handle, deps, text, overrides, runner or default_runner()),
            daemon=True)
        handle.thread.start()
    except BaseException:
        deps.on_piece = None
        _RUN_LOCK.release()
        raise
    return handle


# ---- views and panel rows ------------------------------------------------------------------------

@dataclass
class LiveTrace:
    """Counts of one live step, collected without touching src/: the retriever's own search and
    run methods are wrapped on the instance. Counts only, never result text."""

    searches: int = 0
    results: int = 0
    in_scope: int = 0
    ran: bool = False

    def reset(self) -> None:
        self.searches = self.results = self.in_scope = 0
        self.ran = False


def watch_live(deps) -> LiveTrace | None:
    """Wrap deps.live (once) so each run leaves a LiveTrace; None when there is no live step."""
    live = getattr(deps, "live", None)
    if live is None or not callable(getattr(live, "run", None)):
        return None
    existing = getattr(live, "_p8_trace", None)
    if existing is not None:
        return existing
    trace = LiveTrace()
    run, search = live.run, getattr(live, "search", None)

    def counted_run(*args, **kwargs):
        trace.reset()
        res = run(*args, **kwargs)
        trace.searches = int(getattr(res, "searches", 0) or 0)
        trace.in_scope = len(getattr(res, "items", []) or [])
        trace.ran = True
        return res

    live.run = counted_run
    if search is not None:
        def counted_search(*args, **kwargs):
            out = search(*args, **kwargs)
            trace.results += len(out[0])
            return out

        live.search = counted_search
    live._p8_trace = trace
    return trace


def live_counts(state, trace: LiveTrace | None) -> dict:
    packed = getattr(state, "packed_obj", None)
    merged = sum(1 for p in getattr(packed, "items", []) if p.item.is_live)
    cited = sum(1 for s in (state.final or {}).get("sources", []) if s.get("live"))
    t = trace or LiveTrace()
    left_out = sum(1 for d in (state.final or {}).get("dropped", [])
                   if d.get("reason") in ("live_out_of_plan", "live_unresolved"))
    names_excluded = sum(1 for d in (state.final or {}).get("dropped", [])
                         if d.get("reason") == "live_names_excluded")
    return {"searches": t.searches, "results": t.results, "in_scope": t.in_scope,
            "in_plan": max(t.in_scope - left_out, 0), "dropped_excluded": names_excluded,
            "merged": merged, "cited": cited, "not_cited": max(merged - cited, 0)}


def view_from_state(state, wall_s: float = 0.0, trace: LiveTrace | None = None,
                    use_live: bool = False) -> dict:
    final = state.final or {}
    view = {"final": final, "timings": dict(state.timings), "tokens": dict(state.tokens),
            "live_status": state.live_status, "live_note": state.live_note,
            "repaired": bool(state.repaired), "wall_s": wall_s, "use_live": bool(use_live)}
    if use_live:  # counts exist only for a live-on answer, which is never written to disk
        view["live_counts"] = live_counts(state, trace)
    return view


def live_status_line(view: dict) -> str | None:
    """One counts-only line about the live step; None when live search was not requested."""
    if not view.get("use_live"):
        return None
    status = view.get("live_status")
    c = view.get("live_counts") or {}
    if status in ("ok", "empty"):
        return (f"live search: {status}, {c.get('searches', 0)} searches, "
                f"{c.get('results', 0)} results, {c.get('in_scope', 0)} in scope, "
                f"{c.get('in_plan', 0)} in plan, {c.get('dropped_excluded', 0)} dropped, names "
                f"an excluded place, {c.get('merged', 0)} merged, "
                f"{c.get('cited', 0)} cited, {c.get('not_cited', 0)} not cited")
    if status == "error":
        reason = (view.get("live_note") or "").replace(TERMS_NOTICE, "").strip(" ;")
        return f"live search: error: {redact(reason or 'the search failed')}"
    if status == "no_key":
        return "live search: no key"
    return f"live search: not used ({status})"


def _replace_deep(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, dict):
        return {k: _replace_deep(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_deep(v, old, new) for v in obj]
    return obj


def display_view(view: dict) -> dict:
    """The view as shown: when live search was not requested, the pipeline's 'switched off
    (config/live.json)' note is reworded, because nothing was switched off."""
    if view.get("use_live") or view.get("live_status") not in ("disabled", "not_run"):
        return view
    from src.live.search import NOTES

    return _replace_deep(view, NOTES["disabled"], NOT_REQUESTED_NOTE)


def live_ran(view: dict) -> bool:
    return view.get("live_status") in LIVE_RAN


def terms_notice_visible(view: dict) -> bool:
    return live_ran(view)


CITATION = re.compile(r"\[((?:chunk|live):[0-9A-Za-z]+)\]")
MARKER_NOTE = ("Markers [S#] / [L#] stand for the cited passages; the Chunk id column of the "
               "sources table shows the exact passage")


def source_index(final: dict) -> dict[str, dict]:
    """Every source of an answer by the id the itinerary text prints: 'chunk:<id>' for corpus
    pages, 'live:<hash>' for live items (cited sources first, then retrieved-not-cited ones)."""
    idx: dict[str, dict] = {}
    for s in [*final.get("sources", []), *final.get("uncited_live", [])]:
        cid = str(s.get("chunk_id", ""))
        idx.setdefault(cid if s.get("live") else f"chunk:{cid}", s)
    return idx


def assign_aliases(final: dict) -> dict[str, str]:
    """Short aliases S1, S2, ... (corpus) and L1, L2, ... (live) for the sources of an answer:
    numbered in order of first appearance in the text, no gaps; sources that the text does not
    cite follow in listing order. The stored answer is not touched."""
    idx = source_index(final)
    aliases: dict[str, str] = {}
    counts = {"S": 0, "L": 0}

    def give(key: str) -> None:
        letter = "L" if key.startswith("live:") else "S"
        counts[letter] += 1
        aliases[key] = f"{letter}{counts[letter]}"

    for m in CITATION.finditer(str(final.get("text", ""))):
        if m.group(1) in idx and m.group(1) not in aliases:
            give(m.group(1))
    for key in idx:
        if key not in aliases:
            give(key)
    return aliases


def _link(url: str) -> str | None:
    url = str(url or "")
    if not url.startswith(("http://", "https://")):
        return None
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def cited_text(final: dict) -> tuple[str, dict[str, str]]:
    """The itinerary as shown: each [chunk:<id>] / [live:<id>] marker becomes a short marker
    [S1] / [L1] linked to the source page. A marker whose id is not among the sources stays as it
    is. Returns (text, aliases); `final` is not modified."""
    text = str(final.get("text", ""))
    idx = source_index(final)
    aliases = assign_aliases(final)

    def replace(m: re.Match) -> str:
        s = idx.get(m.group(1))
        if s is None:
            return m.group(0)
        marker = f"[{aliases[m.group(1)]}]"
        link = _link(s.get("url"))
        return f"[{marker}]({link})" if link else marker

    return CITATION.sub(replace, text), aliases


SOURCE_FIELDS = ("Alias", "Chunk id", "Title", "Place", "Source", "Date", "Fetch date", "Link",
                 "Licence", "Flags")
SOURCE_FLAGS = ("card warning", "limited evidence", "retrieved, not cited")


def source_rows(final: dict, attribution: dict, card_warning_ids: set[str]) -> list[dict]:
    """Rows for the sources panel: only the fields in SOURCE_FIELDS, built one by one from short
    labels, links and dates. No chunk text or excerpt can reach a row, whatever the source dict
    carries."""
    rows = []
    aliases = assign_aliases(final)
    entries = [(s, False) for s in final.get("sources", [])]
    entries += [(s, True) for s in final.get("uncited_live", [])]
    for s, uncited in entries:
        info = attribution.get(s.get("url", ""), {})
        kind = s.get("source_type", "")
        if s.get("live"):
            fetched = s.get("fetch_date") or "unknown date"
            title, licence = "live web result", "see the linked page"
            source, date_label = f"live, fetched {fetched}", "undated"
        else:
            fetched = ""
            title = info.get("title") or s.get("place") or "corpus page"
            if kind in ("wikipedia", "wikivoyage", "reference"):
                licence = CC_BY_SA
            elif kind == "blog":
                licence = "link and title only; no licence terms found"
            else:
                licence = info.get("license") or "see the linked page"
            source = kind
            date_label = s["date"] if kind == "blog" and s.get("date") else "undated"
        flags = []
        if s.get("chunk_id") in card_warning_ids:
            flags.append(SOURCE_FLAGS[0])
        if s.get("limited"):
            flags.append(SOURCE_FLAGS[1])
        if uncited:
            flags.append(SOURCE_FLAGS[2])
        cid = str(s.get("chunk_id", ""))
        key = cid if s.get("live") else f"chunk:{cid}"
        rows.append({"Alias": aliases.get(key, s.get("alias", "")),
                     "Chunk id": key,
                     "Title": title, "Place": s.get("place") or "",
                     "Source": source, "Date": date_label, "Fetch date": fetched,
                     "Link": s.get("url", ""), "Licence": licence, "Flags": ", ".join(flags)})
    return rows


def card_warning_chunk_ids(cards: dict) -> set[str]:
    return {w.get("chunk_id") for c in cards.values() for w in c.get("warnings", [])}


def timing_rows(timings: dict) -> tuple[list[dict], float]:
    rows = [{"Node": k, "Seconds": v} for k, v in timings.items()]
    return rows, round(sum(timings.values()), 3)


def writer_line(view: dict) -> str | None:
    """Which writer produced the text. There is no template writer: the itinerary is always the
    hosted language model's reply (a live call, or a replay of an earlier reply from the local LLM
    cache when the prompt was identical). None when no write step ran (an infeasible plan)."""
    tokens = view.get("tokens") or {}
    if "write_reservation" not in tokens and not tokens.get("write"):
        return None
    if int(tokens.get("write") or 0) > 0:
        return f"write step: language model, live call ({int(tokens['write'])} tokens)"
    return "write step: language model reply replayed from the local cache (0 tokens sent)"


def token_rows(tokens: dict) -> list[dict]:
    """Tokens the hosted model reported per step; 0 means the call was served from the local LLM
    cache (nothing was sent). live_credits are assumed, not measured."""
    names = (("parse", "parse (tokens)"), ("write", "write (tokens)"),
             ("critic", "repair (tokens)"), ("live_credits", "live search credits (assumed)"))
    return [{"Step": label, "Value": int(tokens[k])} for k, label in names if k in tokens]


def leg_rows(plan: dict) -> list[dict]:
    return [{"From": leg["origin"], "To": leg["destination"], "Hours": leg["hours"],
             "Travel days": leg["days"], "Kind": leg.get("kind", "between")}
            for leg in plan.get("legs", [])]


# ---- quiet display: notes and caveats in one collapsed expander (D131) -------------------------

NOTES_HEADING = "## Notes and warnings"
BUDGET_SHORT = "estimate"
DETAILS_TITLE = "Details and sources notes"
REPAIRED_NOTE = "The rule checker repaired the draft once; the text above is the repaired one."
CORPUS_ONLY_NOTE = "corpus only: no live search was used for this answer"


def split_notes(text: str) -> tuple[str, list[str]]:
    """(the itinerary without its 'Notes and warnings' block, the lines of that block)."""
    lines = str(text).splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == NOTES_HEADING), None)
    if idx is None:
        return str(text), []
    notes = [ln.strip()[2:].strip() if ln.strip().startswith("- ") else ln.strip()
             for ln in lines[idx + 1:] if ln.strip()]
    return "\n".join(lines[:idx]).rstrip() + "\n", notes


def short_labels(text: str) -> str:
    """The computed lines with short labels: 'provisional' beside travel times, 'estimate' beside
    the budget; the long caveat sentences are shown in the details instead."""
    from src.agents.budget import LABEL

    long_budget = f". {LABEL[0].upper()}{LABEL[1:]}. (computed: budget table)"
    t = str(text).replace(long_budget, f" ({BUDGET_SHORT})")
    t = t.replace("(computed: travel_time, provisional)", f"({TRAVEL_LABEL})")
    return t.replace(" (computed: check_season)", "").replace(" (computed: budget table)", "")


def detail_lines(view: dict, final: dict, prep_notes=()) -> list[str]:
    """Everything the page used to print as notices and caveats, in one list: the notes block of
    the itinerary, warnings, the live status, the recency, marker and label lines. The terms
    notice of a live answer is NOT here (it stays on screen, D109)."""
    from src.agents.budget import LABEL

    _, note_lines = split_notes(str(final.get("text", "")))
    lines: list[str] = []
    if view.get("repaired"):
        lines.append(REPAIRED_NOTE)
    lines += note_lines
    lines += [*(final.get("warnings") or []), *(view.get("warnings") or []), *prep_notes]
    status = view.get("live_status")
    if live_ran(view):
        if status != "ok":
            lines.append(view.get("live_note") or "live search returned nothing usable; "
                         "answers use the corpus only")
    elif status in ("no_key", "quota", "terms_not_verified"):
        lines.append(view.get("live_note")
                     or "live search unavailable; answers use the corpus only")
    else:
        lines.append(CORPUS_ONLY_NOTE)
    count_line = live_status_line(view)
    if count_line:
        lines.append(count_line)
    lines += [RECENCY_NOTE, MARKER_NOTE,
              f"Travel times are {TRAVEL_LABEL}. Budget figures are a {BUDGET_LABEL}. "
              f"Place-card items are {CARD_LABEL}.",
              f"Budget: {LABEL[0].upper()}{LABEL[1:]}."]
    return list(dict.fromkeys(ln.strip() for ln in lines if ln and ln.strip()))
