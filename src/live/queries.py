"""Privacy-safe live queries (D100): a live query is built ONLY from registry place names, state
names, the month, the year and fixed keywords. The user's free text, start city, party,
constraints or anything else never reaches the search provider.

`is_safe_query` is the boundary check: every word of a query must come from that closed
vocabulary (or be a four-digit year), otherwise `live_retrieve` refuses it.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
STATE_NAMES = {"TN": "Tamil Nadu", "KL": "Kerala", "KA": "Karnataka", "AP": "Andhra Pradesh",
               "TS": "Telangana", "PY": "Puducherry"}
KEYWORD_SETS = (
    ("closures", "timings", "opening", "hours"),
    ("ticket", "price", "entry", "fee", "travel", "advisory"),
)
WORD = re.compile(r"[A-Za-z0-9À-ɏ']+")
MAX_PLACES_PER_QUERY = 3


def _tokens(text: str) -> list[str]:
    return [w.casefold() for w in WORD.findall(text)]


def vocabulary(place_labels: Iterable[str]) -> frozenset[str]:
    """Every word allowed in a live query: place names and aliases, state names, months and
    the fixed keywords."""
    words: set[str] = set()
    for label in place_labels:
        words.update(_tokens(label))
    for name in STATE_NAMES.values():
        words.update(_tokens(name))
    for m in MONTHS:
        words.add(m.casefold())
    words.update(("south", "india"))
    for ks in KEYWORD_SETS:
        words.update(ks)
    return frozenset(words)


def is_safe_query(query: str, vocab: frozenset[str]) -> bool:
    toks = _tokens(query)
    if not toks or len(query) > 160:
        return False
    return all(t in vocab or (t.isdigit() and len(t) == 4) for t in toks)


def build_queries(places: list[str], states: list[str], month: int | None, today: date,
                  place_names: Iterable[str], n: int) -> list[str]:
    """Up to `n` queries. `places` are filtered through the registry names (case-insensitive) so
    nothing else can enter; with no usable place the state names are used."""
    known = {p.casefold(): p for p in place_names}
    chosen = []
    for p in places:
        canon = known.get(str(p).casefold())
        if canon and canon not in chosen:
            chosen.append(canon)
    chosen = chosen[:MAX_PLACES_PER_QUERY]
    if not chosen:
        chosen = [STATE_NAMES[s] for s in states if s in STATE_NAMES][:MAX_PLACES_PER_QUERY]
    if not chosen:
        chosen = ["South India"]
    when = f"{MONTHS[month - 1]} {today.year}" if month else str(today.year)
    where = ", ".join(chosen)
    out = [f"{where} {' '.join(ks)} {when}" for ks in KEYWORD_SETS]
    return out[:max(n, 0)]
