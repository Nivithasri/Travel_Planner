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


def clean_text(text: str, max_chars: int) -> str:
    """Whitespace collapsed, URLs and tag/delimiter-like strings removed, cut at a word boundary."""
    t = MOJIBAKE.sub(_repair, str(text)).replace("\u202f", " ").replace("\u00a0", " ")
    t = URL.sub("[link removed]", t)
    t = TAGLIKE.sub(" ", t).replace("[", "(").replace("]", ")")
    t = " ".join(t.split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ") + "..."


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


def _rank(item: EvidenceItem, interests=()) -> tuple:
    """Model-written warnings first, then chunks that match an interest, then the aspect
    priority; with no interest the order is the old one."""
    is_warn = item.is_warning and item.provenance == "llm"
    key = "warning" if is_warn else (item.aspect or "other")
    matched = 0 if interest_match(item, interests) else 1
    return (0 if is_warn else 1, matched, PRIORITY.get(key, 5), -item.score, item.chunk_id)


def pack(plan: Plan, corpus: dict[str, list[EvidenceItem]], live: list[EvidenceItem],
         cfg: dict | None = None, interests=(), named_places=(), scope_check=None) -> Packed:
    """At most `items_per_stop` corpus items per planned stop (warnings first) and `live_items`
    live items. Order is the plan's stop order, then live; aliases follow that order."""
    w = (cfg or load_pipeline_config())["writer"]
    out = Packed()
    n_s = n_l = 0
    seen: set[str] = set()
    for stop in plan.stops:
        taken = 0
        for item in sorted(corpus.get(stop.place, []), key=lambda i: _rank(i, interests)):
            if taken >= w["items_per_stop"]:
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
            text = clean_text(item.text, w["item_chars"])
            if not text:
                continue
            n_s += 1
            seen.add(item.chunk_id)
            out.items.append(PackedItem(f"S{n_s}", item, text, stop.evidence_class == "limited"))
            taken += 1
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
