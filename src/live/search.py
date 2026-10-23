"""Live retrieval (Phase 7, D100): `live_retrieve` and the retriever behind it.

Gates, in order: layer enabled; same-day cache; terms verified by the owner; API key present;
monthly credit stop; per-request search cap; latency budget. Any failure returns a status and a
visible note and the pipeline continues with the corpus alone. Live chunks are tagged
source_type "live", carry the fetch date, are held in memory for the run only and are never
embedded, indexed, written to data/ or eval/, or committed.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from src.agents.evidence import clean_text, injection_flags
from src.agents.state import EvidenceItem
from src.common.http import FetchError, HttpClient
from src.live.config import LiveConfig, load_live_config
from src.live.queries import STATE_NAMES, build_queries, is_safe_query, vocabulary
from src.live.store import DayCache, Ledger, minimal

NOTES = {
    "disabled": "live search is switched off (config/live.json); answers use the corpus only",
    "terms_not_verified": ("live search was not used: the provider's terms are not verified "
                           "(config/live.json); answers use the corpus only"),
    "no_key": "live search was not used: no LIVE_SEARCH_API_KEY is set; corpus only",
    "quota": "live search was not used: the monthly credit stop was reached; corpus only",
    "empty": "live search returned nothing usable for this request; answers use the corpus only",
    "error": "live search failed; answers use the corpus only",
    "ok": "",
}


class UnsafeLiveQuery(ValueError):
    """A live query contained words outside the closed vocabulary (D100 privacy rule)."""


@dataclass
class LiveResult:
    items: list[EvidenceItem] = field(default_factory=list)
    status: str = "not_run"
    note: str = ""
    searches: int = 0
    credits: int = 0
    cache_hits: int = 0
    seconds: float = 0.0


def live_chunk_id(url: str, fetch_date: str) -> str:
    return "live:" + hashlib.sha256(f"{url}\x1f{fetch_date}".encode()).hexdigest()[:12]


def _valid_date(v) -> str | None:
    if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v[:10] or ""):
        return v[:10]
    return None


def to_items(results: Iterable[dict], query: str, scope_terms: Iterable[str], fetch_date: str,
             max_chars: int, place_names: Iterable[str]) -> list[EvidenceItem]:
    """Cleaned live items for results that name a registry place, state or alias (South India
    scope). Items with instruction-like text are dropped later by the packer too."""
    scope = [t.casefold() for t in scope_terms]
    names = list(place_names)
    qwords = {w.casefold() for w in re.findall(r"[A-Za-z]{4,}", query)}
    out = []
    for r in results:
        text = clean_text(f"{r.get('title') or ''}. {r.get('content') or ''}", max_chars)
        hay = text.casefold()
        if not r.get("url") or not any(re.search(rf"\b{re.escape(t)}\b", hay) for t in scope):
            continue
        place = next((n for n in names if re.search(rf"\b{re.escape(n)}\b", hay, re.I)), "")
        overlap = len({w for w in re.findall(r"[a-z]{4,}", hay)} & qwords) / max(len(qwords), 1)
        out.append(EvidenceItem(
            chunk_id=live_chunk_id(r["url"], fetch_date), place=place, source_type="live",
            url=r["url"], text=text, provenance="live", date=_valid_date(r.get("published_date")),
            fetch_date=fetch_date, score=round(overlap, 4)))
    return out


class TavilyClient:
    """POST /search through the project HttpClient (allowlist, throttle, cap, request log). The
    request/response shape is ASSUMED until the first approved smoke test (R1 in the plan)."""

    def __init__(self, http: HttpClient, api_key: str, cfg: LiveConfig):
        self.http, self._key, self.cfg = http, api_key, cfg

    def search(self, query: str) -> list[dict]:
        url = f"https://{self.cfg.endpoint_host}{self.cfg.endpoint_path}"
        body = {"query": query, "max_results": self.cfg.max_results_per_search,
                "search_depth": "basic", "include_answer": False, "include_raw_content": False}
        resp = self.http.post_json(
            url, body, secret_headers={"Authorization": f"Bearer {self._key}"})
        try:
            doc = json.loads(resp.text)
        except ValueError as exc:
            raise FetchError("the search reply was not JSON") from exc
        results = doc.get("results") if isinstance(doc, dict) else None
        if not isinstance(results, list):
            raise FetchError("the search reply has no results list")
        return [r for r in results if isinstance(r, dict)]


class LiveRetriever:
    def __init__(self, cfg: LiveConfig | None = None, *, api_key: str | None = None,
                 client: TavilyClient | None = None, cache: DayCache | None = None,
                 ledger: Ledger | None = None, place_labels: Iterable[str] = (),
                 place_names: Iterable[str] = (), clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], date] = lambda: datetime.now(timezone.utc).date()):
        self.cfg = cfg or load_live_config()
        self._key, self.client = api_key, client
        self.cache = cache or DayCache(self.cfg.cache_path, self.cfg.cache_enabled)
        self.ledger = ledger or Ledger(self.cfg.cache_path / "ledger.json",
                                       self.cfg.monthly_stop_at, self.cfg.monthly_credits)
        self.place_names = list(place_names)
        self.vocab = vocabulary([*place_labels, *self.place_names])
        self._clock, self._today = clock, today
        self.searches_this_request = 0

    def new_request(self) -> None:
        self.searches_this_request = 0

    def _gate(self) -> str | None:
        if not self.cfg.enabled:
            return "disabled"
        return None

    def search(self, query: str, started: float | None = None) -> tuple[list[dict], str, bool]:
        """(raw minimal results, status, from_cache). Raises UnsafeLiveQuery."""
        if not is_safe_query(query, self.vocab):
            raise UnsafeLiveQuery("a live query may only contain registry place names, state "
                                  "names, a month, a year and fixed keywords")
        if (g := self._gate()):
            return [], g, False
        hit = self.cache.get(query)
        if hit is not None:
            return hit, "ok" if hit else "empty", True
        gate = self.cfg.real_calls_status()
        if gate != "ok":
            return [], gate, False
        if not self._key:
            return [], "no_key", False
        if self.searches_this_request >= self.cfg.max_searches_per_request:
            return [], "quota", False
        if not self.ledger.can_spend(self.cfg.credits_per_search):
            return [], "quota", False
        if started is not None and self._clock() - started > self.cfg.latency_budget_s:
            return [], "error", False
        if self.client is None:
            raise RuntimeError("no search client was given")
        self.searches_this_request += 1
        # charged before the call: a failed call may still have cost credits
        self.ledger.spend(self.cfg.credits_per_search)
        try:
            raw = self.client.search(query)
        except (FetchError, OSError):
            return [], "error", False
        results = [minimal(r, self.cfg.snippet_max_chars) for r in raw]
        self.cache.put(query, results)
        return results, "ok" if results else "empty", False

    def run(self, places: list[str], states: list[str], month: int | None) -> LiveResult:
        """The live node: build the safe queries, search, filter to scope, rank, cap."""
        t0 = self._clock()
        self.new_request()
        today = self._today()
        queries = build_queries(places, states, month, today, self.place_names,
                                self.cfg.default_searches_per_request)
        scope = [*self.place_names, *STATE_NAMES.values()]
        res = LiveResult()
        statuses = []
        seen: set[str] = set()
        for q in queries:
            results, status, cached = self.search(q, started=t0)
            statuses.append(status)
            res.cache_hits += int(cached)
            for it in to_items(results, q, scope, today.isoformat(), self.cfg.snippet_max_chars,
                               self.place_names):
                if it.chunk_id not in seen and not injection_flags(it.text):
                    seen.add(it.chunk_id)
                    res.items.append(it)
            if status in ("disabled", "terms_not_verified", "no_key", "quota"):
                break
        res.searches = self.searches_this_request  # real calls only; each cost credits
        res.credits = res.searches * self.cfg.credits_per_search
        res.items.sort(key=lambda i: (-i.score, i.chunk_id))
        res.items = res.items[:self.cfg.max_live_chunks]
        if res.items:
            res.status = "ok"
        else:
            bad = next((s for s in statuses if s not in ("ok", "empty")), None)
            res.status = bad or "empty"
        res.note = NOTES.get(res.status, "")
        res.seconds = round(self._clock() - t0, 3)
        return res


def live_retrieve(query: str, places: list[str] | None = None, k: int = 5, *,
                  retriever: LiveRetriever | None = None) -> list[dict]:
    """The contract function (docs/contracts.md): live chunks as dicts with the chunk fields plus
    "score". The query must pass the closed-vocabulary check (UnsafeLiveQuery otherwise)."""
    if retriever is None:
        from src.retrieval.query_understanding import load_core_registry

        reg = load_core_registry()
        names = [e.name for e in reg.entries if not e.holdout]
        labels = [a for e in reg.entries if not e.holdout for a in e.aliases]
        retriever = LiveRetriever(place_labels=labels, place_names=names)
    results, status, _ = retriever.search(query)
    today = retriever._today().isoformat()
    scope = [*retriever.place_names, *STATE_NAMES.values()]
    items = to_items(results, query, scope, today, retriever.cfg.snippet_max_chars,
                     retriever.place_names)
    if places:
        want = {p.casefold() for p in places}
        items = [i for i in items if i.place.casefold() in want or not i.place]
    items.sort(key=lambda i: (-i.score, i.chunk_id))
    return [{**i.model_dump(), "score": i.score} for i in items[:k]]
