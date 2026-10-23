"""Place cards (Phase 7, D102): a precomputed, extractive summary per destination, with citations.

`build_cards` is pure and deterministic (no LLM, no network, no clock); scripts/build_place_cards.py
feeds it the enriched corpus and writes index/place_cards.json (gitignored: it holds short excerpts
of CC BY-SA text, O12). `get_place_card(place)` reads that file.

Rules (D69, D102):
* only model-written warnings (provenance `llm` AND is_warning) are shown as warnings; rule-based
  flags are only counted;
* tips and highlights of a FULL-evidence place come from model-written rows; a LIMITED place (a
  D33-excluded place) has only rule-based rows, which are used but flagged;
* blog chunks (no terms) are excluded unless include_blog is set;
* every item cites a chunk of that place; nothing is generated, so nothing can be invented;
* the card never claims recency: the corpus has 0 published chunks in the 18-month window (D32).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.common.schemas import REPO_ROOT
from src.tools.routing import UnknownPlaceError

CARDS_PATH = REPO_ROOT / "index" / "place_cards.json"
EXCERPT_CHARS = 240
MAX_TIPS, MAX_WARNINGS, MAX_HIGHLIGHTS = 5, 5, 6
RECENCY = ("No published chunk falls in the 18 months before the reference date; wiki chunks carry "
           "revision dates only. Nothing on this card is evidence of current conditions.")
TIP_ASPECTS = ("timing", "price", "closure", "safety", "food", "other")
HIGHLIGHT_ASPECTS = ("scenery", "history")
HIGHLIGHT_TYPES = ("temple", "heritage", "beach", "hill_station", "wildlife", "backwater",
                   "waterfall")
ASPECT_RANK = {a: i for i, a in enumerate(
    ("timing", "price", "closure", "safety", "food", "scenery", "history", "other"))}
SOURCE_RANK = {"wikivoyage": 0, "reference": 1}
_SENT = re.compile(r"(?<=[.!?])\s+")


class CardItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    chunk_id: str
    source_type: str
    aspect: str | None = None
    provenance: str  # llm | fallback | rule_only
    url: str


class PlaceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place: str
    aliases: list[str] = Field(default_factory=list)
    state: str
    popularity: str
    visibility_tercile: int | None = None  # 1 = most visited .. 3 = least (full-evidence places)
    evidence_class: str  # full | limited
    hill: bool = False
    nearest_airport: dict | None = None
    best_months: dict = Field(default_factory=dict)
    tips: list[CardItem] = Field(default_factory=list)
    warnings: list[CardItem] = Field(default_factory=list)
    highlights: list[CardItem] = Field(default_factory=list)
    counts: dict = Field(default_factory=dict)
    recency: str = RECENCY
    caveats: list[str] = Field(default_factory=list)


def excerpt(text: str, max_chars: int = EXCERPT_CHARS) -> str:
    """The first full sentence that starts with a capital letter and is long enough, cut at a word
    boundary to max_chars. Falls back to the start of the text. Deterministic."""
    clean = " ".join(str(text).split())
    for sent in _SENT.split(clean):
        if len(sent) >= 40 and sent[:1].isupper():
            clean = sent
            break
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ")
    return cut + "..."


def _item(chunk: dict, prov: str) -> CardItem:
    return CardItem(text=excerpt(chunk["text"]), chunk_id=chunk["chunk_id"],
                    source_type=chunk["source_type"], aspect=chunk.get("aspect"),
                    provenance=prov, url=chunk["url"])


def _tercile(rank: int, n: int) -> int:
    return 1 + (rank * 3) // n


def visibility_terciles(entries: Iterable, full_places: set[str]) -> dict[str, int | None]:
    """Tercile 1 (most visited) to 3 (least) by popularity_signal among the full-evidence places;
    a missing signal counts as the least visible (D19: an unavailable signal means hidden)."""
    rows = [(e.name, getattr(e, "popularity_signal", None)) for e in entries
            if e.name in full_places]
    rows.sort(key=lambda r: (-(r[1] if r[1] is not None else -1.0), r[0]))
    n = len(rows)
    return {name: _tercile(i, n) for i, (name, _) in enumerate(rows)}


def best_months(season_place: dict | None) -> dict:
    if not season_place:
        return {}
    good, fair, poor, risk = [], [], [], []
    for m in sorted(season_place["months"], key=int):
        d = season_place["months"][m]
        {"good": good, "fair": fair, "poor": poor}[d["verdict"]].append(int(m))
        if d.get("access_risk"):
            risk.append(int(m))
    return {"good": good, "fair": fair, "poor": poor, "access_risk": risk,
            "confidence": season_place.get("confidence"),
            "basis": "ERA5 monthly climate normals (check_season); climate only, not road status"}


def build_cards(chunks: Iterable[dict], provenance: dict[str, str], entries: list,
                season: dict, excluded: set[str], *, holdout=None,
                include_blog: bool = False) -> dict[str, dict]:
    """{place name: card dict} for every non-hold-out registry place that has season data."""
    by_place: dict[str, list[dict]] = {}
    for c in chunks:
        by_place.setdefault(c["place"], []).append(c)
    live_entries = [e for e in entries if not getattr(e, "holdout", False)]
    full_places = {e.name for e in live_entries if e.name not in excluded
                   and e.name in season["places"]}
    terciles = visibility_terciles(live_entries, full_places)
    cards: dict[str, dict] = {}
    for e in sorted(live_entries, key=lambda x: x.name):
        sp = season["places"].get(e.name)
        if sp is None:
            continue
        limited = e.name in excluded
        rows = by_place.get(e.name, [])
        kinds = Counter()
        pool: list[tuple[dict, str]] = []
        rule_flags = 0
        for c in rows:
            prov = provenance.get(c["chunk_id"], "unknown")
            kinds[(prov, c["source_type"])] += 1
            if prov != "llm" and c.get("is_warning"):
                rule_flags += 1
            if c["source_type"] == "blog" and not include_blog:
                continue
            if holdout is not None and holdout.text_match(c["text"]):
                continue
            usable = prov == "llm" or (limited and prov in ("rule_only", "fallback"))
            if usable:
                pool.append((c, prov))
        warn_pool = [(c, p) for c, p in pool if p == "llm" and c.get("is_warning")]
        warn_pool.sort(key=lambda cp: (ASPECT_RANK.get(cp[0].get("aspect"), 9), cp[0]["chunk_id"]))
        used = {c["chunk_id"] for c, _ in warn_pool[:MAX_WARNINGS]}
        tip_pool = [(c, p) for c, p in pool if not c.get("is_warning")
                    and c.get("aspect") in TIP_ASPECTS and c["chunk_id"] not in used]
        tip_pool.sort(key=lambda cp: (ASPECT_RANK.get(cp[0].get("aspect"), 9),
                                      SOURCE_RANK.get(cp[0]["source_type"], 2),
                                      cp[0]["chunk_id"]))
        tips = _pick(tip_pool, MAX_TIPS, used)
        hl_pool = [(c, p) for c, p in pool if not c.get("is_warning")
                   and c["chunk_id"] not in used
                   and (c.get("aspect") in HIGHLIGHT_ASPECTS
                        or c.get("place_type") in HIGHLIGHT_TYPES)]
        hl_pool.sort(key=lambda cp: (SOURCE_RANK.get(cp[0]["source_type"], 2),
                                     str(cp[0].get("place_type")), cp[0]["chunk_id"]))
        highlights = _pick(hl_pool, MAX_HIGHLIGHTS, used, distinct_types=True)
        warnings = [_item(c, p) for c, p in warn_pool[:MAX_WARNINGS]]
        caveats = []
        if limited:
            caveats.append("limited evidence: this place is excluded from the evaluated corpus "
                           "(D33); its chunks carry rule-based labels (D69) and no warnings can "
                           "be shown")
        if sp.get("confidence") == "low":
            caveats.append("climate confidence is low for this place (D42)")
        if not warnings and not limited:
            caveats.append("no model-written warning exists for this place in the corpus")
        card = PlaceCard(
            place=e.name, aliases=list(e.aliases), state=getattr(e.state, "value", str(e.state)),
            popularity=e.popularity, visibility_tercile=None if limited else terciles.get(e.name),
            evidence_class="limited" if limited else "full", hill=bool(sp.get("hill")),
            nearest_airport=sp.get("nearest_airport"), best_months=best_months(sp),
            tips=[_item(c, p) for c, p in tips], warnings=warnings,
            highlights=[_item(c, p) for c, p in highlights],
            counts={"chunks_total": len(rows),
                    "by_provenance_source": {f"{p}:{s}": n for (p, s), n in sorted(kinds.items())},
                    "blog_excluded": 0 if include_blog else sum(
                        n for (_, s), n in kinds.items() if s == "blog"),
                    "rule_based_warning_flags": rule_flags,
                    "model_written_warnings": len(warn_pool)},
            caveats=caveats)
        cards[e.name] = card.model_dump()
    return cards


def _pick(pool: list[tuple[dict, str]], limit: int, used: set[str], *,
          distinct_types: bool = False) -> list[tuple[dict, str]]:
    out: list[tuple[dict, str]] = []
    seen_text: set[str] = set()
    seen_types: set[str] = set()
    for c, p in pool:
        if c["chunk_id"] in used or len(out) >= limit:
            continue
        key = excerpt(c["text"])
        if key in seen_text:
            continue
        if distinct_types and len(out) < 3 and str(c.get("place_type")) in seen_types:
            continue
        seen_text.add(key)
        seen_types.add(str(c.get("place_type")))
        used.add(c["chunk_id"])
        out.append((c, p))
    return out


# ---- reading the built file ---------------------------------------------------------------

_cache: dict[Path, tuple[tuple[int, int], dict]] = {}


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _load(path: Path) -> dict:
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    doc = json.loads(path.read_text(encoding="utf-8"))
    _cache[path] = (stamp, doc)
    return doc


def get_place_card(place: str, path: Path = CARDS_PATH) -> dict:
    """The card of a destination (name or registry alias, any case). Unknown names, hold-outs and
    hubs raise UnknownPlaceError (a ValueError) like the Phase 5 tools (D81)."""
    doc = _load(Path(path))
    names: dict[str, str] = {}
    for name, card in doc["places"].items():
        for label in [name, *card.get("aliases", [])]:
            names.setdefault(_norm(label), name)
    key = _norm(place) if isinstance(place, str) else ""
    if key not in names:
        raise UnknownPlaceError(place, list(doc["places"]))
    return doc["places"][names[key]]
