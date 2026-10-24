"""Out-of-scope place names in live text (D120).

A live snippet may name a place the project does not cover (a D33-excluded place the user did not
name, or a hold-out): the writer would copy it and the critic's R5 would then fail the answer. The
matcher says whether a text names such a place. Word-bounded, case-insensitive, over registry
names and aliases (aliases of four letters or more, as the critic does), plus the hold-out filter
and the independent hold-out scan used for enrichment (D63). Counts only leave this module.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable


def _word_re(labels: Iterable[str]) -> re.Pattern | None:
    labels = sorted({x.strip() for x in labels if x and x.strip()}, key=len, reverse=True)
    if not labels:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in labels) + r")(?!\w)",
                      re.IGNORECASE)


def out_of_scope_matcher(registry_entries: Iterable, excluded: Iterable[str],
                         allowed_places: Iterable[str]) -> Callable[[str], bool]:
    """A function text -> True when the text names an excluded place that is not allowed (not a
    stop of the plan and not named in the request) or any hold-out place."""
    from src.facts.facts import HoldoutFilter
    from src.process.enrich import leak_scan

    entries = list(registry_entries)
    excluded = set(excluded)
    allowed = set(allowed_places)
    labels: set[str] = set()
    held: dict[str, str] = {}
    for e in entries:
        if getattr(e, "holdout", False):
            for t in [e.name, *e.aliases]:
                labels.add(t)
                if len(t) > 3:
                    held[t] = e.name
        elif e.name in excluded and e.name not in allowed:
            labels.add(e.name)
            labels.update(a for a in e.aliases if len(a) >= 4)
    pattern = _word_re(labels)
    holdout = HoldoutFilter((), held)

    def names_out_of_scope(text: str) -> bool:
        if pattern is not None and pattern.search(text):
            return True
        return leak_scan(text, holdout) is not None

    return names_out_of_scope
