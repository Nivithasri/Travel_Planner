"""Walk the Wikivoyage hierarchy (state -> region -> city) and read its destination lists (D17).

Each state article has a "Regions" section ({{Regionlist}} with regionNname=[[Region]]); each region
article has "Cities" and "Other destinations" lists, and may have sub-regions. Entries in the lists
are marker templates: name=[[Target|Display]], wikidata=Q.., lat=.., long=.. . A name that is plain
text (no link) has no Wikivoyage article of its own.

`count_see_do` counts the See/Do listings of an article: the tourism filter (an administrative
city with fewer than MIN_SEE_DO listings is not a destination). Articles name these sections "See",
"Do" or "See and do", so the count is over the {{see}}, {{do}} and {{listing|type=see/do}} templates
anywhere in the article; Wikivoyage uses those templates only for that purpose.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.common.http import HttpClient

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
LIST_SECTIONS = ("Cities", "Other destinations")
STATE_ARTICLES = ("Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana")
MIN_SEE_DO = 3
MIN_SEE_DO_WORDS = 120
SEE_DO_SECTIONS = ("See", "Do", "See and do")
MAX_DEPTH = 3
MAX_ARTICLES = 80

_MARKER = re.compile(r"\{\{\s*marker\b(.*?)\}\}", re.S | re.I)
_LINK = re.compile(r"\[\[([^\]|#]+)(?:\|([^\]]*))?\]\]")
_REGION_NAME = re.compile(r"region\d*name\s*=\s*\[\[([^\]|#]+)", re.I)
_LISTING = re.compile(
    r"\{\{\s*(?:see|do)\s*\||\{\{\s*listing\b[^}]*?\btype\s*=\s*(?:see|do)\b", re.I | re.S
)
_PIPE = "␟"  # stands in for the "|" inside [[Target|Display]] while splitting parameters


@dataclass(frozen=True)
class Marker:
    article: str  # the article whose list it came from
    section: str
    title: str | None  # linked Wikivoyage article, None when the name is plain text
    display: str
    wikidata: str | None  # the marker's own wikidata= value
    lat: float | None = None
    lon: float | None = None


def section_text(wikitext: str, name: str) -> str:
    """A level-2 section including its sub-sections."""
    m = re.search(
        r"^==\s*" + re.escape(name) + r"\s*==\s*$(.*?)(?=^==[^=]|\Z)", wikitext,
        re.M | re.S | re.I,
    )
    return m.group(1) if m else ""


def _float(v: str | None) -> float | None:
    try:
        return float(v) if v else None
    except ValueError:
        return None


def parse_markers(text: str, article: str, section: str) -> list[Marker]:
    out = []
    for m in _MARKER.finditer(text):
        params = {}
        body = _LINK.sub(lambda lm: lm.group(0).replace("|", _PIPE), m.group(1))
        for part in body.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip().lower()] = v.strip()
        raw = params.get("name", "").replace(_PIPE, "|")
        link = _LINK.search(raw)
        if link:
            title = link.group(1).strip()
            display = (link.group(2) or link.group(1)).strip()
        else:
            title, display = None, raw
        qid = params.get("wikidata", "")
        if display:
            out.append(Marker(
                article, section, title, display, qid if qid.startswith("Q") else None,
                _float(params.get("lat")), _float(params.get("long") or params.get("lon")),
            ))
    return out


def region_links(wikitext: str) -> list[str]:
    return [t.strip() for t in _REGION_NAME.findall(section_text(wikitext, "Regions"))]


def count_see_do(wikitext: str) -> int:
    return len(_LISTING.findall(wikitext))


_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b.*?</ref>", re.S | re.I)
_FILE = re.compile(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
_INNER_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TAG = re.compile(r"<[^>]+>")
_HEADING_LINE = re.compile(r"^=+.*=+\s*$", re.M)
_WIKILINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_QUOTES = re.compile(r"'{2,}")


def see_do_words(wikitext: str) -> int:
    """Words of prose in the See / Do / See and do sections (all sub-sections included).
    Comments, references, images, templates (listings are counted by count_see_do) and
    markup are removed before counting."""
    text = "\n".join(section_text(wikitext, s) for s in SEE_DO_SECTIONS)
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    text = _FILE.sub(" ", text)
    prev = None
    while prev != text:  # strip templates innermost-first so nested ones go too
        prev, text = text, _INNER_TEMPLATE.sub(" ", text)
    text = _HEADING_LINE.sub(" ", text)
    text = _WIKILINK.sub(r"\1", text)
    text = _QUOTES.sub("", _TAG.sub(" ", text))
    return len(re.findall(r"[^\W_]+", text))


def passes_tourism_filter(wikitext: str) -> tuple[bool, int, int]:
    """(passes, listings, words): >= MIN_SEE_DO listings OR >= MIN_SEE_DO_WORDS words."""
    n, w = count_see_do(wikitext), see_do_words(wikitext)
    return (n >= MIN_SEE_DO or w >= MIN_SEE_DO_WORDS), n, w


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def fetch_wikitext(http: HttpClient, title: str) -> tuple[str, str] | None:
    """(canonical title, wikitext) following redirects, or None if the article does not exist."""
    res = http.fetch(
        WIKIVOYAGE_API,
        {"action": "parse", "page": title, "prop": "wikitext", "format": "json",
         "formatversion": 2, "redirects": 1},
        check_robots=False, validate=_is_json,
    )
    body = res.json()
    if "error" in body:
        return None
    return body["parse"]["title"], body["parse"]["wikitext"]


def crawl(
    http: HttpClient,
    states: tuple[str, ...] = STATE_ARTICLES,
    max_depth: int = MAX_DEPTH,
    max_articles: int = MAX_ARTICLES,
) -> tuple[list[Marker], list[tuple[str, int]]]:
    """Breadth-first walk state -> regions -> sub-regions. Returns (markers, [(article, depth)])."""
    queue = [(s, 0) for s in states]
    seen: set[str] = set()
    visited: list[tuple[str, int]] = []
    markers: list[Marker] = []
    while queue and len(visited) < max_articles:
        title, depth = queue.pop(0)
        got = fetch_wikitext(http, title)
        if got is None:
            continue
        canonical, wikitext = got
        if canonical in seen:
            continue
        seen.add(canonical)
        visited.append((canonical, depth))
        for section in LIST_SECTIONS:
            markers += parse_markers(section_text(wikitext, section), canonical, section)
        if depth < max_depth:
            queue += [(r, depth + 1) for r in region_links(wikitext)]
    return markers, visited
