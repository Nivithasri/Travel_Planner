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

ROOT = Path(__file__).resolve().parents[1]

BUDGET_TIERS = ("budget", "mid", "luxury")
PARTIES = ("solo", "couple", "family", "group")
INTEREST_CHOICES = ("temples", "heritage", "beaches", "hill stations", "wildlife", "backwaters",
                    "food", "nature", "culture", "history")
MAX_FREE_CHARS = 200
LIVE_RAN = ("ok", "empty", "error")
BUSY_MESSAGE = "another run is in progress"
LIVE_CREDITS_HINT = "uses search credits (assumed 2 per search, not measured)"
PRIVACY_HINT = "do not enter personal information"
RECENCY_NOTE = ("no corpus item is dated within the last 18 months; live items show their fetch "
                "date, not a publish date")
SEASON_NOTE = "trip spans several months; season check uses the start month only"
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

def view_from_state(state, wall_s: float = 0.0) -> dict:
    final = state.final or {}
    return {"final": final, "timings": dict(state.timings), "tokens": dict(state.tokens),
            "live_status": state.live_status, "live_note": state.live_note,
            "repaired": bool(state.repaired), "wall_s": wall_s}


def live_ran(view: dict) -> bool:
    return view.get("live_status") in LIVE_RAN


def terms_notice_visible(view: dict) -> bool:
    return live_ran(view)


def source_rows(final: dict, attribution: dict, card_warning_ids: set[str]) -> list[dict]:
    """Rows for the sources panel. Only links, titles and labels: no chunk text, ever."""
    rows = []
    for s in final.get("sources", []):
        info = attribution.get(s.get("url", ""), {})
        kind = s.get("source_type", "")
        if s.get("live"):
            title, licence = "live web result", "see the linked page"
            when = f"live, fetched {s.get('fetch_date') or 'unknown date'}"
        else:
            title = info.get("title") or s.get("place") or "corpus page"
            if kind in ("wikipedia", "wikivoyage", "reference"):
                licence = CC_BY_SA
            elif kind == "blog":
                licence = "link and title only; no licence terms found"
            else:
                licence = info.get("license") or "see the linked page"
            when = (f"published {s['date']}" if kind == "blog" and s.get("date")
                    else "undated or wiki revision (not a current date)")
        flags = []
        if s.get("chunk_id") in card_warning_ids:
            flags.append("card warning")
        if s.get("limited"):
            flags.append("limited evidence")
        rows.append({"Alias": s.get("alias", ""), "Title": title, "Place": s.get("place") or "",
                     "Source": "live" if s.get("live") else kind, "When": when,
                     "Link": s.get("url", ""), "Licence": licence, "Flags": ", ".join(flags)})
    return rows


def card_warning_chunk_ids(cards: dict) -> set[str]:
    return {w.get("chunk_id") for c in cards.values() for w in c.get("warnings", [])}


def timing_rows(timings: dict) -> tuple[list[dict], float]:
    rows = [{"Node": k, "Seconds": v} for k, v in timings.items()]
    return rows, round(sum(timings.values()), 3)


def leg_rows(plan: dict) -> list[dict]:
    return [{"From": leg["origin"], "To": leg["destination"], "Hours": leg["hours"],
             "Travel days": leg["days"], "Kind": leg.get("kind", "between")}
            for leg in plan.get("legs", [])]
