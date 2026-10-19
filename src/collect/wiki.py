"""Wikivoyage and Wikipedia collectors: one plain-text extract per place via the MediaWiki API.

Both are CC BY-SA wikis read through their documented API with the project's contact User-Agent
(Wikimedia policy). `publish_date` stays null: a wiki page has no single publish date, and the
revision timestamp goes to `lastmod`, which is never used as a publish date.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from src.collect.raw import RawStore, make_record
from src.collect.types import Budget, SourceResult, check_block
from src.common.http import FetchError, HttpClient
from src.common.schemas import PlaceRegistryEntry

WIKIS = {
    "wikivoyage": {"api": "https://en.wikivoyage.org/w/api.php",
                   "page": "https://en.wikivoyage.org/wiki/", "title_field": "wikivoyage_title",
                   "license": "CC BY-SA (Wikivoyage; see page history for authors)"},
    "wikipedia": {"api": "https://en.wikipedia.org/w/api.php",
                  "page": "https://en.wikipedia.org/wiki/", "title_field": "wikipedia_title",
                  "license": "CC BY-SA (Wikipedia; see page history for authors)"},
}


def _is_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


class WikiCollector:
    def __init__(self, source: str, http: HttpClient, manifest: str | Path | None = None):
        if source not in WIKIS:
            raise ValueError(f"unknown wiki source {source!r}")
        self.source, self.http, self.manifest = source, http, manifest
        self.cfg = WIKIS[source]

    def collect(self, entry: PlaceRegistryEntry, budget: Budget, now: Callable[[], float],
                raw_root: Path) -> SourceResult:
        title = getattr(entry, self.cfg["title_field"])
        if not title:
            return SourceResult(skipped=f"no {self.source} article for {entry.name}")
        if budget.cap_for(self.source, 1) < 1:
            return SourceResult(skipped="request cap is 0")
        before = self.http.requests_made
        try:
            res = self.http.fetch(
                self.cfg["api"],
                {"action": "query", "prop": "extracts|revisions", "titles": title,
                 "explaintext": 1, "exsectionformat": "wiki", "rvprop": "ids|timestamp",
                 "redirects": 1, "format": "json", "formatversion": 2},
                check_robots=False, validate=_is_json, deadline=budget.deadline)
        except FetchError as exc:
            check_block(exc, self.source)
            return SourceResult(error=str(exc)[:200],
                                n_requests=self.http.requests_made - before)
        n_req = self.http.requests_made - before
        if res.status in (403, 429):
            check_block(FetchError("blocked", res.status), self.source)
        if not res.ok:
            return SourceResult(n_requests=n_req, error=f"HTTP {res.status} from {self.source}")
        pages = res.json().get("query", {}).get("pages", [])
        page = pages[0] if pages else {}
        text = (page.get("extract") or "").strip()
        if page.get("missing") or not text:
            return SourceResult(n_requests=n_req, error=f"no {self.source} page text for {title!r}")
        rev = (page.get("revisions") or [{}])[0]
        stamp = rev.get("timestamp") or ""
        canonical = page.get("title", title)
        rec = make_record(
            self.source, entry.name, self.cfg["page"] + canonical.replace(" ", "_"), text,
            res.retrieved_at, title=canonical, license=self.cfg["license"],
            lastmod=stamp[:10] or None, extra={"revid": rev.get("revid")})
        wrote = RawStore(raw_root, self.manifest).write(rec)
        return SourceResult(n_raw=1 if wrote else 0, n_requests=n_req)
