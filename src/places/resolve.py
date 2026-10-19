"""Free text -> canonical place. Registry and aliases first (no network), then Wikidata.

Outcomes: resolved (an entry, existing or newly built), ambiguous (candidates, nothing is picked),
rejected (a real place outside TN/KL/KA/AP/TS/PY), not_found. A new entry is only returned here; the
caller (ensure_place) decides to store it in the live registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from src.common.schemas import PlaceRegistryEntry
from src.places.registry import Registry, normalize
from src.places.wikidata import WikidataClient, district_of, state_from_ancestors

Status = Literal["resolved", "ambiguous", "rejected", "not_found"]
MAX_WIKIDATA_CANDIDATES = 7
POPULARITY_SIGNAL = "wikipedia_pageviews_12m"


@dataclass
class Resolution:
    status: Status
    entry: PlaceRegistryEntry | None = None
    candidates: list[dict] = field(default_factory=list)
    reason: str = ""
    new: bool = False  # True when `entry` was just built from Wikidata (not yet in any registry)
    network: bool = False  # True when the resolution consulted Wikidata (may still be cached)


def _candidate_dict(e: PlaceRegistryEntry) -> dict:
    return {"name": e.name, "state": e.state.value, "wikidata_id": e.wikidata_id}


def resolve_place(
    text: str,
    registries: list[Registry],
    wd: WikidataClient,
    as_of: date,
    popular_cutoff: float | None = None,
) -> Resolution:
    text = text.strip()
    if not normalize(text):
        return Resolution("not_found", reason="empty place text")

    # 1. registries: no network
    hits: list[PlaceRegistryEntry] = []
    for reg in registries:
        for e in reg.find(text):
            if e not in hits:
                hits.append(e)
    if len(hits) == 1:
        return Resolution("resolved", hits[0])
    if len(hits) > 1:
        return Resolution(
            "ambiguous", candidates=[_candidate_dict(e) for e in hits],
            reason=f"{text!r} matches several known places",
        )

    # 2. Wikidata
    in_scope: list[tuple] = []
    out_of_scope: list[str] = []
    for cand in wd.search(text, limit=MAX_WIKIDATA_CANDIDATES):
        known = next((r.by_wikidata_id(cand.qid) for r in registries if r.by_wikidata_id(cand.qid)),
                     None)
        if known:
            in_scope.append((cand, None, None, known))
            continue
        ent = wd.entity(cand.qid)
        if ent is None or ent.lat is None or ent.lon is None:
            continue  # not a located thing (films, people, ...)
        ancestors = wd.ancestors(cand.qid)
        if not ancestors:
            continue
        state = state_from_ancestors(ancestors)
        if state is None:
            out_of_scope.append(f"{ent.label} ({cand.description})")
            continue
        in_scope.append((cand, ent, (state, district_of(ancestors)), None))

    if not in_scope:
        if out_of_scope:
            return Resolution(
                "rejected", network=True,
                reason="outside TN/KL/KA/AP/TS/PY: " + "; ".join(out_of_scope[:3]),
            )
        return Resolution("not_found", network=True, reason=f"no located place found for {text!r}")

    key = normalize(text)
    exact = [
        t for t in in_scope
        if normalize(t[0].label) == key
        or any(normalize(a) == key for a in (t[1].aliases if t[1] else []))
    ]
    pick = exact or in_scope
    if len(pick) > 1:
        cands = []
        for cand, ent, geo, known in pick:
            cands.append(_candidate_dict(known) if known else {
                "name": cand.label, "state": geo[0], "wikidata_id": cand.qid,
                "description": cand.description,
            })
        return Resolution(
            "ambiguous", candidates=cands, network=True,
            reason=f"{text!r} could be several places; nothing was picked",
        )

    cand, ent, geo, known = pick[0]
    if known:
        return Resolution("resolved", known, network=True)
    assert ent is not None and geo is not None
    state, district = geo
    views = wd.pageviews_12m(ent.wikipedia_title, as_of) if ent.wikipedia_title else None
    if views is not None and popular_cutoff is not None:
        popularity = "popular" if views >= popular_cutoff else "hidden_gem"
        pop_source = POPULARITY_SIGNAL
    else:
        popularity, pop_source = "hidden_gem", "unavailable"
    name = ent.label or cand.label
    if any(r.get(name) for r in registries):  # name taken by a different place
        name = f"{name} ({district or state})"
    aliases = sorted({a for a in [*ent.aliases, text, ent.wikipedia_title or ""]
                      if a and normalize(a) != normalize(name)})
    entry = PlaceRegistryEntry(
        name=name, aliases=aliases, state=state,
        region_cluster=f"{state}-{district}" if district else f"{state}-unclustered",
        popularity=popularity, lat=ent.lat, lon=ent.lon, hub=None,
        wikidata_id=ent.qid, wikipedia_title=ent.wikipedia_title,
        wikivoyage_title=ent.wikivoyage_title,
        coord_source="wikidata:P625",
        region_cluster_source="wikidata:P131 district" if district else "state only",
        popularity_source=pop_source,
        popularity_signal=float(views) if views is not None else None,
    )
    return Resolution("resolved", entry, new=True, network=True)
