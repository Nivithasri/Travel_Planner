"""Web collector for the fixed domain allowlist in config/sources.json (D17).

Only domains listed under "web_allowlist" are ever fetched; a domain under "ask_first" or anything
else is refused. Pages are found through the site's own sitemap (sitemap.xml, wp-sitemap.xml), never
guessed, and only those whose URL names the place. robots.txt is checked by HttpClient for every
page and a per-domain delay is applied. The page date comes from the page itself (article
published_time, JSON-LD datePublished, <time>); the sitemap lastmod is stored as `lastmod` only.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from src.collect.raw import RawStore, iso_utc, make_record, sha256_text
from src.collect.types import Budget, SourceResult, check_block
from src.common.cache import append_jsonl
from src.common.config import REPO_ROOT
from src.common.http import FetchError, HttpClient
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import normalize

CONFIG_PATH = REPO_ROOT / "config" / "sources.json"
SITEMAP_PATHS = ("/sitemap.xml", "/wp-sitemap.xml")
MAX_SITEMAP_FILES = 12
MIN_TEXT_CHARS = 400
_CC = re.compile(r"creativecommons\.org/licenses/([\w-]+)/([\d.]+)", re.I)
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


Pages = list[tuple[str, str | None]]  # (url, sitemap lastmod)


class SourcesConfigError(RuntimeError):
    pass


def load_allowlist(path: Path = CONFIG_PATH) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    allow = {d["domain"].lower(): d for d in doc.get("web_allowlist", [])}
    clash = set(allow) & {d.lower() for d in doc.get("ask_first", [])}
    if clash:
        raise SourcesConfigError(f"domain both allowed and ask-first: {sorted(clash)}")
    for d in allow.values():
        for key in ("source_type", "raw_source", "min_delay_s", "max_pages_per_place", "license"):
            if key not in d:
                raise SourcesConfigError(f"{d['domain']}: missing {key}")
    return allow


def host_key(host: str) -> str:
    h = host.lower()
    return h[4:] if h.startswith("www.") else h


def applies_to(cfg: dict, entry: PlaceRegistryEntry) -> bool:
    """A domain may be scoped to states and/or place names in config/sources.json."""
    states, places = cfg.get("states"), cfg.get("places")
    if states and entry.state.value not in states:
        return False
    return not places or entry.name in places


def place_keys(entry: PlaceRegistryEntry) -> list[str]:
    keys = {normalize(entry.name)}
    keys |= {normalize(a) for a in entry.aliases if len(normalize(a)) >= 5}
    return sorted(k for k in keys if k)


def url_names_place(url: str, keys: list[str]) -> bool:
    path = " " + normalize(urlsplit(url).path) + " "
    return any(f" {k} " in path for k in keys)


def parse_sitemap(xml: str) -> tuple[Pages, list[str]]:
    """(page urls with lastmod, nested sitemap urls)"""
    soup = BeautifulSoup(xml, "xml")
    pages, nested = [], []
    for u in soup.find_all("url"):
        loc = u.find("loc")
        if loc and loc.text.strip():
            lm = u.find("lastmod")
            m = _DATE.search(lm.text) if lm else None
            pages.append((loc.text.strip(), m.group(0) if m else None))
    for s in soup.find_all("sitemap"):
        loc = s.find("loc")
        if loc and loc.text.strip():
            nested.append(loc.text.strip())
    return pages, nested


def extract_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.find("meta", property="og:title") or {}).get("content") or (
        soup.title.string.strip() if soup.title and soup.title.string else "")
    published = None
    meta = soup.find("meta", property="article:published_time")
    candidates = [meta.get("content") if meta else None]
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(tag.string or "")
        except ValueError:
            continue
        for node in (blob if isinstance(blob, list) else [blob]):
            if isinstance(node, dict):
                candidates.append(node.get("datePublished"))
    t = soup.find("time", attrs={"datetime": True})
    candidates.append(t.get("datetime") if t else None)
    for c in candidates:
        m = _DATE.match(c or "")
        if m:
            published = m.group(0)
            break
    licenses = sorted({f"{a}/{b}" for a, b in _CC.findall(html)})
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
                     "svg"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = re.sub(r"[ \t]+", " ", root.get_text("\n"))
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return {"title": str(title).strip(), "text": text, "publish_date": published,
            "cc_licenses": licenses}


class WebCollector:
    source = "web"

    def __init__(self, http: HttpClient, allowlist: dict[str, dict] | None = None,
                 manifest: str | Path | None = None):
        self.http, self.manifest = http, manifest
        self.allow = load_allowlist() if allowlist is None else allowlist
        for dom, cfg in self.allow.items():
            for h in (dom, host_key(dom), "www." + host_key(dom)):
                http.host_delays[h] = float(cfg["min_delay_s"])

    def _discover(
        self, domain: str, keys: list[str], deadline: float | None
    ) -> tuple[Pages, str | None]:
        found: Pages = []
        queue = [f"https://{domain}{p}" for p in SITEMAP_PATHS]
        try:  # sitemaps announced in robots.txt come first (cached, same fetch the check uses)
            rob = self.http.fetch(f"https://{domain}/robots.txt", check_robots=False,
                                  max_age=86400, deadline=deadline)
            if rob.ok:
                queue = [m.group(1) for m in re.finditer(r"(?im)^\s*sitemap:\s*(\S+)", rob.text)
                         if host_key(urlsplit(m.group(1)).netloc) == host_key(domain)] + queue
        except FetchError as exc:
            check_block(exc, domain)
        seen, errors = set(), []
        while queue and len(seen) < MAX_SITEMAP_FILES:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                res = self.http.fetch(url, max_age=7 * 86400, deadline=deadline)
            except FetchError as exc:
                check_block(exc, url)
                errors.append(str(exc)[:80])
                continue
            if not res.ok:
                continue
            pages, nested = parse_sitemap(res.text)
            found += [p for p in pages if url_names_place(p[0], keys)]
            queue += [n for n in nested if host_key(urlsplit(n).netloc) == host_key(domain)]
            if len(found) >= 15 or (found and not nested):
                break
        return found, ("; ".join(errors) if errors and not found else None)

    def _record_skip(self, raw_source: str, place: str, url: str, text: str,
                     retrieved_at: float) -> str:
        """A page with too little text is skipped, but never silently: a status-file note and a
        manifest line record url, chars and reason (no content)."""
        chars = len(text)
        if self.manifest is not None:
            append_jsonl(self.manifest, [{
                "url": url, "source": raw_source, "place": place,
                "retrieved_at": iso_utc(retrieved_at), "sha256": sha256_text(text), "bytes": 0,
                "skipped": "too_short", "chars": chars, "min_chars": MIN_TEXT_CHARS}])
        return f"{url}: skipped, {chars} chars of text < {MIN_TEXT_CHARS} (too_short)"

    def collect(self, entry: PlaceRegistryEntry, budget: Budget, now: Callable[[], float],
                raw_root: Path) -> SourceResult:
        if not self.allow:
            return SourceResult(skipped="no domain is on the allowlist")
        keys = place_keys(entry)
        store = RawStore(raw_root, self.manifest)
        before = self.http.requests_made
        n_raw, errors, complete = 0, [], True
        notes: list[str] = []
        skip_notes: list[str] = []
        for domain, cfg in self.allow.items():
            if not applies_to(cfg, entry):
                continue
            cap = budget.cap_for("web", int(cfg["max_pages_per_place"]))
            pages, err = self._discover(domain, keys, budget.deadline)
            if not pages:
                notes.append(f"{domain}: no sitemap URL names {entry.name}")
            if err:
                errors.append(f"{domain}: {err}")
            for url, lastmod in pages[: min(cap, int(cfg["max_pages_per_place"]))]:
                if budget.expired(now()):
                    complete = False
                    break
                try:
                    res = self.http.fetch(url, deadline=budget.deadline)
                except FetchError as exc:
                    check_block(exc, url)
                    errors.append(f"{url}: {str(exc)[:80]}")
                    continue
                if not res.ok:
                    continue
                page = extract_page(res.text)
                if len(page["text"]) < MIN_TEXT_CHARS:
                    skip_notes.append(self._record_skip(
                        cfg["raw_source"], entry.name, url, page["text"], res.retrieved_at))
                    continue
                lic = cfg["license"]
                if cfg["license"].upper().startswith("CC") and page["cc_licenses"]:
                    want = "by-nc-sa/4.0"
                    if want not in page["cc_licenses"]:  # a page under another licence is skipped
                        errors.append(f"{url}: page licence {page['cc_licenses']} differs")
                        continue
                rec = make_record(
                    cfg["raw_source"], entry.name, url, page["text"], res.retrieved_at,
                    title=page["title"], publish_date=page["publish_date"], lastmod=lastmod,
                    license=lic, extra={"domain": domain, "cc_markers": page["cc_licenses"]})
                n_raw += store.write(rec)
        return SourceResult(n_raw=n_raw, n_requests=self.http.requests_made - before,
                            error="; ".join(errors)[:300] if errors and not n_raw else None,
                            complete=complete,
                            notes=[e[:200] for e in errors] + skip_notes + notes)
