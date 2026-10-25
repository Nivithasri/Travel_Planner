"""Evidence packing for the writer (Phase 7, D103, D104; O7): chooses a bounded set of corpus and
live items, aliases them S1.. and L1.., and neutralises untrusted text.

Corpus and live text is DATA. Items that carry an instruction-like pattern are dropped (and
recorded), tag-like and delimiter-like strings are removed, URLs are replaced, and every item is
cut to a fixed length, so the prompt size is bounded and a source cannot forge a citation tag.
Blog chunks (no terms) are excluded unless the config allows them (owner decision, D103).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.agents.state import EvidenceItem, Plan
from src.common.schemas import REPO_ROOT

PIPELINE_CONFIG = REPO_ROOT / "config" / "pipeline.json"
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
MOJIBAKE = re.compile("â[^\\x00-\\x7f]{1,2}")  # UTF-8 bytes read as cp1252 (the rupee sign)
TAGLIKE = re.compile(r"\[(?:S|L|P|T)\d+\]|<<[^>]*>>|<<|>>", re.I)


def _repair(m: re.Match) -> str:
    try:
        return m.group(0).encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return " "


INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above)|disregard (the )?(previous|above|rules)|"
    r"system prompt|you (must|should) now|new instructions?|act as (a|an)\b|"
    r"(^|\n)\s*(assistant|system|user)\s*:|do not follow|override (the )?(rules|instructions)|"
    r"reveal (your|the) (prompt|instructions)|jailbreak", re.I)
# The closed interest list of the form mapped to the chunk fields it can match (D53 enums). Fixed
# by hand and not tuned on any query (D116). A chunk matches an interest when its place_type or
# its aspect is listed AND its fields are model-written (provenance llm; rule-based fields are
# exempt, D69). Interests outside the list match nothing.
INTEREST_MAP = {
    "temples": {"place_types": ("temple",), "aspects": ("history",)},
    "heritage": {"place_types": ("heritage", "temple"), "aspects": ("history",)},
    "history": {"place_types": ("heritage",), "aspects": ("history",)},
    "culture": {"place_types": ("temple", "heritage"), "aspects": ("history",)},
    "beaches": {"place_types": ("beach",), "aspects": ("scenery",)},
    "hill stations": {"place_types": ("hill_station",), "aspects": ("scenery",)},
    "wildlife": {"place_types": ("wildlife",), "aspects": ("scenery",)},
    "backwaters": {"place_types": ("backwater",), "aspects": ("scenery",)},
    "waterfalls": {"place_types": ("waterfall",), "aspects": ("scenery",)},
    "nature": {"place_types": ("wildlife", "waterfall", "hill_station", "backwater"),
               "aspects": ("scenery",)},
    "food": {"place_types": (), "aspects": ("food",)},
}
PRIORITY = {"warning": 0, "timing": 1, "closure": 1, "price": 2, "safety": 2, "food": 3,
            "scenery": 4, "history": 4, "other": 5}


def load_pipeline_config(path: Path = PIPELINE_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def injection_flags(text: str) -> list[str]:
    return sorted({m.group(0).strip().lower() for m in INJECTION.finditer(text)})


MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec")
# Wiki editor notes (D131). Only a parenthetical that is nothing but "updated <month> <year>" goes;
# every "as of 2017" stays, so a dated figure never looks current.
EDITOR_NOTE = re.compile(
    r"\(\s*(?:last\s+)?(?:updated|revised|edited)\s+(?:in\s+)?"
    rf"(?:(?:{MONTHS})[a-z]*\.?\s+)?(?:\d{{1,2}}\s+)?\d{{4}}\s*\)|"
    r"\[\s*edit\s*\]|\(?\bcitation needed\b\)?", re.I)
UPDATED_LINE = re.compile(
    rf"(?m)^[ \t]*(?:last\s+)?updated:?\s+(?:in\s+)?(?:(?:{MONTHS})[a-z]*\.?\s+)?\d{{4}}\.?[ \t]*$",
    re.I)
# "3 Hebbal Lake." / "1. Cave temples": a list number at the start of a line before a capitalised
# word (not a month, which would be a date).
LIST_PREFIX = re.compile(
    rf"(?m)^[ \t]*\d{{1,2}}[.)]?[ \t]+(?=(?!(?i:{MONTHS})[a-z]*\b)[A-Z][a-z])")


def strip_editor_notes(text: str) -> str:
    """Remove wiki editor notes and list-number prefixes from raw chunk text (before cleaning, so
    line starts are still visible)."""
    t = UPDATED_LINE.sub("", str(text))
    t = EDITOR_NOTE.sub(" ", t)
    return LIST_PREFIX.sub("", t)


def clean_text(text: str, max_chars: int, sentences: bool = False) -> str:
    """Whitespace collapsed, URLs and tag/delimiter-like strings removed, cut at a word boundary.
    With sentences=True the cut is at the last full sentence that fits (when one of at least 40
    characters does) and no ellipsis is added there."""
    t = MOJIBAKE.sub(_repair, str(text)).replace("\u202f", " ").replace("\u00a0", " ")
    t = URL.sub("[link removed]", t)
    t = TAGLIKE.sub(" ", t).replace("[", "(").replace("]", ")")
    t = " ".join(t.split())
    if len(t) <= max_chars:
        return t
    if sentences:
        cut = _sentence_cut(t, max_chars)
        if cut:
            return cut
    return t[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ") + "..."


def _sentence_cut(t: str, max_chars: int, min_chars: int = 40) -> str | None:
    out = ""
    for s in split_sentences(t):
        nxt = f"{out} {s}".strip()
        if len(nxt) > max_chars:
            break
        out = nxt
    return out if len(out) >= min_chars else None


SENTENCE_END = re.compile(r"(?<=[.!?])[\"')]?\s+(?=[\"'(]?[A-Z0-9])")
ABBREV = re.compile(r"(?:\b(?:St|Dr|Mr|Mrs|Ms|No|Rs|approx|vs|etc|Sr|Jr|Mt|Fr)|\b[A-Z])\.$")


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends, not after an abbreviation or a single capital letter."""
    out: list[str] = []
    start = 0
    for m in SENTENCE_END.finditer(text):
        piece = text[start:m.start()]
        if ABBREV.search(piece.rstrip("\"') ")):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def lead(text: str, max_chars: int) -> str:
    """The first sentence(s) of a cleaned text, up to max_chars, never cut mid-sentence unless the
    first sentence alone is longer (then cut at a word and marked with '...')."""
    parts = split_sentences(text)
    if not parts:
        return ""
    out = parts[0]
    if len(out) > max_chars:
        return out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ") + "..."
    for nxt in parts[1:]:
        if len(out) + 1 + len(nxt) > max_chars or len(out) >= max_chars * 0.6:
            break
        out += " " + nxt
    return out


@dataclass
class PackedItem:
    alias: str
    item: EvidenceItem
    text: str
    limited: bool = False


@dataclass
class Packed:
    items: list[PackedItem] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    # plan day -> the S aliases that day may use (D131); place -> its number of plan days
    day_items: dict[int, list[str]] = field(default_factory=dict)
    place_days: dict[str, int] = field(default_factory=dict)

    @property
    def alias_map(self) -> dict[str, str]:
        return {p.alias: p.item.chunk_id for p in self.items}

    def by_alias(self, alias: str) -> PackedItem | None:
        return next((p for p in self.items if p.alias == alias), None)


def interest_match(item: EvidenceItem, interests) -> bool:
    """True when a model-written chunk matches one of the closed-list interests (D116)."""
    if item.provenance != "llm":
        return False
    for name in interests or ():
        m = INTEREST_MAP.get(str(name).strip().casefold())
        if m and (item.place_type in m["place_types"] or item.aspect in m["aspects"]):
            return True
    return False


def _is_warning(item: EvidenceItem) -> bool:
    return bool(item.is_warning and item.provenance == "llm")


# Text about getting there, where to sleep and road conditions (D131): useful sometimes, but a
# sight, a dish or a timing is what a day should be built from. Ranked below them, not dropped.
LOGISTICS_STRONG = re.compile(
    r"\b(?:by (?:bus|train|road|air|car|taxi|flight|rail)|take an? (?:bus|train|taxi|cab|auto)|"
    r"(?:bus|railway|train) (?:stand|station|stop|junction)|airports?|taxis?|cabs?|"
    r"auto[- ]?rickshaws?|accommodation|guest ?houses?|lodges?|dormitor(?:y|ies)|check[- ]?in|"
    r"public transport(?:ation)?|national highway|NH ?\d+|ghat road|"
    r"\d+(?:\.\d+)? ?km (?:from|away|south|north|east|west))\b", re.I)
LOGISTICS_WEAK = re.compile(
    r"\b(?:hotels?|resorts?|rooms?|fares?|highway|roads?|buses|trains?)\b", re.I)
LOGISTICS_WORDS = ("road", "train", "bus", "airport", "taxi", "hotel", "stay", "accommodation",
                   "transport", "drive", "driving", "highway", "ghat", "rail", "flight")


def is_logistics(text: str) -> bool:
    return bool(LOGISTICS_STRONG.search(text) or len(LOGISTICS_WEAK.findall(text)) >= 2)


def logistics_wanted(constraints) -> bool:
    """True when a constraint of the request is about roads, transport or where to stay."""
    joined = " ".join(str(c) for c in constraints or ()).casefold()
    return any(word in joined for word in LOGISTICS_WORDS)


def low_value_ids(items: list[EvidenceItem], constraints=(), elsewhere=None) -> set[str]:
    """Chunk ids of a stop that rank below its sights: logistics text (unless a constraint asks
    for it) and text about other places (`elsewhere(text, place)` -> bool)."""
    keep = logistics_wanted(constraints)
    return {i.chunk_id for i in items
            if (not keep and is_logistics(i.text)) or (elsewhere and elsewhere(i.text, i.place))}


def is_low_value(item: EvidenceItem, elsewhere=None) -> bool:
    return bool(low_value_ids([item], (), elsewhere))


def _rank(item: EvidenceItem, interests=(), demoted=frozenset(), low=frozenset()) -> tuple:
    """Model-written warnings first (except the ones beyond the per-stop cap, D129, and logistics
    or off-place text), then sights before logistics, then chunks that match an interest, then the
    aspect priority; with no interest, no cap and no logistics the order is the old one."""
    is_low = item.chunk_id in low
    is_warn = _is_warning(item) and item.chunk_id not in demoted and not is_low
    key = "warning" if is_warn else (item.aspect or "other")
    matched = 0 if interest_match(item, interests) else 1
    return (0 if is_warn else 1, 1 if is_low else 0, matched, PRIORITY.get(key, 5), -item.score,
            item.chunk_id)


def _demoted(items: list[EvidenceItem], w: dict, interests, cap: int,
             low=frozenset()) -> set[str]:
    """The model-written warnings of one stop beyond the best `cap`, among the items that can be
    packed at all (no live, no blog, no instruction-like text, no logistics). They are not
    dropped: they rank like any other chunk (D129; the warning flag is only about 61% right,
    D106)."""
    usable = [i for i in items if _is_warning(i) and not i.is_live and i.chunk_id not in low
              and not (i.source_type == "blog" and not w["allow_blog"])
              and not injection_flags(i.text)]
    warns = sorted(usable, key=lambda i: _rank(i, interests, low=low))
    return {i.chunk_id for i in warns[cap:]}


def drop_one(packed: Packed) -> PackedItem:
    """Remove and return the corpus item that costs a day the least: the lowest-ranked item of the
    place with the most items per day (the later place on a tie), so the last stop does not lose
    all its evidence first. Falls back to the last item when only live items are left."""
    s_idx = [i for i, p in enumerate(packed.items) if p.alias.startswith("S")]
    if not s_idx:
        return packed.items.pop()
    by_place: dict[str, list[int]] = {}
    for i in s_idx:
        by_place.setdefault(packed.items[i].item.place, []).append(i)
    place = max(by_place, key=lambda pl: (len(by_place[pl]) / packed.place_days.get(pl, 1),
                                          max(by_place[pl])))
    gone = packed.items.pop(max(by_place[place]))
    for aliases in packed.day_items.values():
        if gone.alias in aliases:
            aliases.remove(gone.alias)
    return gone


def pack(plan: Plan, corpus: dict[str, list[EvidenceItem]], live: list[EvidenceItem],
         cfg: dict | None = None, interests=(), named_places=(), scope_check=None,
         constraints=(), elsewhere=None) -> Packed:
    """Per planned stop at most `items_per_night` corpus items for each night, never more than
    `items_per_stop_max`, with at most `warnings_per_stop` model-written warnings ranked first
    (D129), and `live_items` live items. Order is the plan's stop order, then live; aliases follow
    that order."""
    w = (cfg or load_pipeline_config())["writer"]
    out = Packed()
    n_s = n_l = 0
    seen: set[str] = set()
    for stop in plan.stops:
        taken = 0
        days = [d.day for d in plan.days if d.place == stop.place]
        span = max(stop.nights, len(days), 1)
        limit = min(w["items_per_night"] * span, w["items_per_stop_max"])
        first = len(out.items)
        out.place_days[stop.place] = max(len(days), 1)
        items = corpus.get(stop.place, [])
        low = low_value_ids(items, constraints, elsewhere)
        demoted = _demoted(items, w, interests, w["warnings_per_stop"], low)
        for item in sorted(items, key=lambda i: _rank(i, interests, demoted, low)):
            if taken >= limit:
                break
            if item.chunk_id in seen or item.is_live:
                continue
            if item.source_type == "blog" and not w["allow_blog"]:
                out.dropped.append({"chunk_id": item.chunk_id, "reason": "blog_excluded"})
                continue
            flags = injection_flags(item.text)
            if flags:
                out.dropped.append({"chunk_id": item.chunk_id, "reason": "injection_pattern",
                                    "flags": flags})
                continue
            text = clean_text(strip_editor_notes(item.text), w["item_chars"], sentences=True)
            if not text:
                continue
            n_s += 1
            seen.add(item.chunk_id)
            out.items.append(PackedItem(f"S{n_s}", item, text, stop.evidence_class == "limited"))
            taken += 1
        # the stop's items, best first, are dealt over its days: every day gets its own share
        for i, packed_item in enumerate(out.items[first:]):
            if days:
                out.day_items.setdefault(days[i % len(days)], []).append(packed_item.alias)
    taken = 0
    # D119: a live item reaches the writer only if its resolved place is a stop of the plan or a
    # place the request named; items about other places, or about no resolved place, are dropped.
    in_plan = {s.place for s in plan.stops} | set(named_places)
    for item in sorted(live, key=lambda i: (-i.score, i.chunk_id)):
        if not item.place:
            out.dropped.append({"chunk_id": item.chunk_id, "reason": "live_unresolved"})
            continue
        if item.place not in in_plan:
            out.dropped.append({"chunk_id": item.chunk_id, "reason": "live_out_of_plan"})
            continue
        if scope_check is not None and scope_check(item.text):  # D120
            out.dropped.append({"chunk_id": item.chunk_id, "reason": "live_names_excluded"})
            continue
        if taken >= w["live_items"]:
            break
        flags = injection_flags(item.text) + injection_flags(item.url)
        if flags:
            out.dropped.append({"chunk_id": item.chunk_id, "reason": "injection_pattern",
                                "flags": flags})
            continue
        text = clean_text(item.text, w["live_item_chars"])
        if not text:
            continue
        n_l += 1
        out.items.append(PackedItem(f"L{n_l}", item, text))
        taken += 1
    return out
