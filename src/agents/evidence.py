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
TAGLIKE = re.compile(r"\[(?:S|L|P|T)\d+\]|<<[^>]*>>|<<|>>", re.I)
INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above)|disregard (the )?(previous|above|rules)|"
    r"system prompt|you (must|should) now|new instructions?|act as (a|an)\b|"
    r"(^|\n)\s*(assistant|system|user)\s*:|do not follow|override (the )?(rules|instructions)|"
    r"reveal (your|the) (prompt|instructions)|jailbreak", re.I)
PRIORITY = {"warning": 0, "timing": 1, "closure": 1, "price": 2, "safety": 2, "food": 3,
            "scenery": 4, "history": 4, "other": 5}


def load_pipeline_config(path: Path = PIPELINE_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def injection_flags(text: str) -> list[str]:
    return sorted({m.group(0).strip().lower() for m in INJECTION.finditer(text)})


def clean_text(text: str, max_chars: int) -> str:
    """Whitespace collapsed, URLs and tag/delimiter-like strings removed, cut at a word boundary."""
    t = URL.sub("[link removed]", str(text))
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


def _rank(item: EvidenceItem) -> tuple:
    key = "warning" if (item.is_warning and item.provenance == "llm") else (item.aspect or "other")
    return (PRIORITY.get(key, 5), -item.score, item.chunk_id)


def pack(plan: Plan, corpus: dict[str, list[EvidenceItem]], live: list[EvidenceItem],
         cfg: dict | None = None) -> Packed:
    """At most `items_per_stop` corpus items per planned stop (warnings first) and `live_items`
    live items. Order is the plan's stop order, then live; aliases follow that order."""
    w = (cfg or load_pipeline_config())["writer"]
    out = Packed()
    n_s = n_l = 0
    seen: set[str] = set()
    for stop in plan.stops:
        taken = 0
        for item in sorted(corpus.get(stop.place, []), key=_rank):
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
    for item in sorted(live, key=lambda i: (-i.score, i.chunk_id)):
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
