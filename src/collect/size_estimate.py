"""ESTIMATE ONLY. This is NOT the Phase 2 cleaner and must not be imported by it.

It exists to answer one question before the full run: roughly how many chunks will the raw text
give? Phase 2 will clean, de-duplicate, drop off-topic text and apply O8 (hold-out mentions), so
every figure computed here is an UPPER BOUND on the final corpus size.

Two levels, both estimate-only:
- estimate_clean: drop wiki heading lines "== x ==", drop the share-button lines that start
  inditales posts, drop caption tags such as [Music], turn non-breaking spaces into spaces and
  collapse whitespace. The stored wiki bodies are already plain text (MediaWiki TextExtracts), so
  there is no wiki markup left to strip and this level removes very little.
- estimate_strict: additionally drops the back-matter sections listed in BACK_MATTER (with their
  sub-sections), which carry no travel information.
No de-duplication, no relevance filtering in either.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.collect.types import SOURCE_TYPE

CHARS_PER_TOKEN = 4.0  # assumption for English prose
OVERLAP = 0.15
# exact section headings dropped by the strict estimate (matched case-insensitively)
BACK_MATTER = ("references", "external links", "see also", "notes", "bibliography", "gallery",
               "further reading", "sources", "footnotes", "citations")
_HEADING = re.compile(r"^\s*(=+)\s*(.*?)\s*=+\s*$")
_SHARE = {"facebook", "x", "pinterest", "whatsapp", "linkedin", "telegram", "twitter", "email"}
_CAPTION_TAG = re.compile(r"\[(?:music|applause|laughter|singing|inaudible)\]", re.IGNORECASE)
_WS = re.compile(r"\s+")


def estimate_clean(text: str, source: str) -> str:
    lines = text.replace("\xa0", " ").splitlines()
    out = []
    leading = True
    for line in lines:
        s = line.strip()
        if source in ("wikipedia", "wikivoyage") and _HEADING.match(s):
            continue
        if leading and s.lower() in _SHARE:
            continue
        if s:
            leading = False
        out.append(s)
    return _WS.sub(" ", _CAPTION_TAG.sub(" ", " ".join(out))).strip()


def estimate_strict(text: str, source: str) -> tuple[str, list[str]]:
    """(text without back-matter sections, the exact headings that were dropped)."""
    if source not in ("wikipedia", "wikivoyage"):
        return estimate_clean(text, source), []
    kept, dropped = [], []
    skip_level = 0  # heading level of the section being dropped; 0 = not dropping
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if skip_level and level <= skip_level:
                skip_level = 0
            if not skip_level and title.lower() in BACK_MATTER:
                skip_level = level
                dropped.append(title)
        if not skip_level:
            kept.append(line)
    return estimate_clean("\n".join(kept), source), dropped


def chunks_for(chars: int, chunk_tokens: int) -> float:
    """Chunks of chunk_tokens with OVERLAP: stride = (1 - OVERLAP) * chunk_tokens tokens."""
    return chars / ((1 - OVERLAP) * chunk_tokens * CHARS_PER_TOKEN)


# ---- merged from src/collect/matrix.py: ESTIMATE ONLY. Place x source-type and place x aspect
# coverage of data/raw (no network).

ASPECTS = ("timing", "closure", "price", "scenery", "food", "safety", "history",
           "transport", "season")
KEYWORDS = {
    "timing": ("opening hours", "opens at", "open from", "closes at", "timings", "darshan",
               "o'clock", "a.m.", "p.m."),
    "closure": ("closed", "closure", "shut", "not allowed", "banned", "restricted",
                "under renovation"),
    "price": ("rs.", "rs ", "₹", "inr", "entry fee", "ticket", "admission", "per person", "cost"),
    "scenery": ("view", "scenic", "waterfall", "sunrise", "sunset", "beach", "valley",
                "landscape", "viewpoint"),
    "food": ("food", "restaurant", "cuisine", "dosa", "biryani", "meal", "cafe", "thali"),
    "safety": ("safety", "scam", "warning", "avoid", "danger", "landslide", "leech", "caution",
               "pickpocket"),
    "history": ("history", "century", "dynasty", "built by", "temple was", "king", "colonial",
                "ruins", "empire"),
    "transport": ("bus", "train", "railway", "airport", "taxi", "auto-rickshaw", "ferry",
                  "metro", "km from"),
    "season": ("monsoon", "winter", "summer", "best time", "season", "rainy", "festival", "peak"),
}
_PATTERNS = {a: re.compile("|".join(rf"(?<!\w){re.escape(k)}(?!\w)" for k in ks), re.IGNORECASE)
             for a, ks in KEYWORDS.items()}


def tag_aspects(text: str) -> set[str]:
    return {a for a, pat in _PATTERNS.items() if pat.search(text)}


def build_matrix(records: list[dict]) -> dict:
    """records: raw record dicts (source, place, body, publish_date). Returns per-place figures."""
    out: dict[str, dict] = defaultdict(lambda: {
        "by_type": defaultdict(lambda: {"records": 0, "chars": 0}),
        "aspects": defaultdict(int), "records": 0, "dated": 0, "newest": None})
    for r in records:
        p = out[r["place"]]
        t = SOURCE_TYPE[r["source"]]
        cleaned, _ = estimate_strict(r["body"], r["source"])
        cell = p["by_type"][t]
        cell["records"] += 1
        cell["chars"] += len(cleaned)
        p["records"] += 1
        for a in tag_aspects(cleaned):
            p["aspects"][a] += 1
        d = r.get("publish_date")  # publish date only; a revision date never counts (O9)
        if d:
            p["dated"] += 1
            if p["newest"] is None or d > p["newest"]:
                p["newest"] = d
    return out
