"""Collectors for the raw snapshot and on-demand places: Wikivoyage and Wikipedia, the allowlisted
web sites, Reddit."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from src.collect.run import (
    Budget,
    RawStore,
    SourceResult,
    check_block,
    iso_utc,
    make_record,
    sha256_text,
)
from src.common.cache import append_jsonl
from src.common.http import FetchError, HttpClient
from src.common.schemas import REPO_ROOT, PlaceRegistryEntry
from src.places.places import normalize

# ==== wiki =======================================================================
# Wikivoyage and Wikipedia collectors: one plain-text extract per place via the MediaWiki API.
#
# Both are CC BY-SA wikis read through their documented API with the project's contact User-Agent
# (Wikimedia policy). `publish_date` stays null: a wiki page has no single publish date, and the
# revision timestamp goes to `lastmod`, which is never used as a publish date.

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


# ==== web ========================================================================
# Web collector for the fixed domain allowlist in config/sources.json (D17).
#
# Only domains listed under "web_allowlist" are ever fetched; a domain under "ask_first" or anything
# else is refused. Pages are found through the site's own sitemap (sitemap.xml, wp-sitemap.xml),
# never
# guessed, and only those whose URL names the place. robots.txt is checked by HttpClient for every
# page and a per-domain delay is applied. The page date comes from the page itself (article
# published_time, JSON-LD datePublished, <time>); the sitemap lastmod is stored as `lastmod` only.

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


# ==== reddit =====================================================================
# Reddit collector (official OAuth API, application-only). Site-wide search with a place filter.
#
# No subreddit list (D19): the query is "<place> travel", and a post is kept only when its title or
# text names the place as a whole word. Top-level comments of the best few posts are stored as
# their own records because closures, prices and timings are mostly in comments.
#
# Needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET from .env. Without them the collector is
# skipped, never faked. The bearer token is a secret: it is sent with `secret_headers`, so it is in
# no cache key, cache file, manifest or message. The token request is a POST and is the one network
# call that does not go through the disk cache (a cached token would be wrong and a secret).
# Reddit's API terms apply to use of this data and must be reviewed before the corpus is shared.

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
DEFAULT_REQUESTS = 12  # 1 search + up to 11 comment fetches
MAX_COMMENT_POSTS = 6
MAX_COMMENTS_PER_POST = 12
MIN_COMMENT_SCORE = 2
REDDIT_MIN_TEXT_CHARS = 80
_REMOVED = {"[removed]", "[deleted]", ""}


def fetch_token(client_id: str, client_secret: str, user_agent: str) -> str:
    r = requests.post(TOKEN_URL, auth=(client_id, client_secret),
                      data={"grant_type": "client_credentials"},
                      headers={"User-Agent": user_agent}, timeout=(10, 30))
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("Reddit returned no access_token")
    return token


def _date(epoch) -> str | None:
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def names_place(text: str, keys: list[str]) -> bool:
    flat = " " + re.sub(r"[^a-z0-9]+", " ", text.casefold()) + " "
    return any(f" {k} " in flat for k in keys)


class RedditCollector:
    source = "reddit"

    def __init__(self, http: HttpClient, client_id: str | None, client_secret: str | None,
                 manifest: str | Path | None = None,
                 token_fn: Callable[[str, str, str], str] = fetch_token):
        self.http, self.manifest = http, manifest
        self.client_id, self.client_secret, self.token_fn = client_id, client_secret, token_fn
        self._token: str | None = None
        http.host_delays.setdefault("oauth.reddit.com", 1.0)

    @property
    def available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get(self, path: str, params: dict, deadline: float | None):
        assert self._token
        return self.http.fetch(
            API + path, params, check_robots=False, deadline=deadline,
            secret_headers={"Authorization": f"bearer {self._token}"})

    def collect(self, entry: PlaceRegistryEntry, budget: Budget, now: Callable[[], float],
                raw_root: Path) -> SourceResult:
        if not self.available:
            return SourceResult(skipped="no Reddit credentials in .env (access requested)")
        cap = budget.cap_for("reddit", DEFAULT_REQUESTS)
        before = self.http.requests_made
        n_extra = 0  # the token POST is not counted by HttpClient
        try:
            if self._token is None:
                self._token = self.token_fn(self.client_id, self.client_secret,
                                            self.http.user_agent)
                n_extra += 1
        except Exception as exc:  # noqa: BLE001 - any token failure is reported, not raised
            return SourceResult(error=f"Reddit token request failed: {type(exc).__name__}",
                                n_requests=1)
        keys = place_keys(entry)
        store = RawStore(raw_root, self.manifest)
        n_raw, complete = 0, True

        def used() -> int:
            return self.http.requests_made - before

        try:
            res = self._get("/search", {"q": f"{entry.name} travel", "type": "link",
                                        "sort": "relevance", "t": "all", "limit": 25,
                                        "raw_json": 1}, budget.deadline)
            if res.status == 401:
                self._token = None
                return SourceResult(error="Reddit rejected the credentials (HTTP 401)",
                                    n_requests=used() + n_extra)
            if res.status in (403, 429):
                check_block(FetchError("blocked", res.status), "reddit")
            if not res.ok:
                return SourceResult(error=f"Reddit search HTTP {res.status}",
                                    n_requests=used() + n_extra)
            posts = [c["data"] for c in res.json().get("data", {}).get("children", [])
                     if c.get("kind") == "t3"]
            kept = []
            for p in posts:
                text = f"{p.get('title', '')}\n\n{p.get('selftext', '')}".strip()
                if p.get("over_18") or not names_place(text, keys):
                    continue
                kept.append(p)
                body = p.get("selftext", "")
                if body in _REMOVED:
                    body = ""
                full = f"{p.get('title', '')}\n\n{body}".strip()
                if len(full) < REDDIT_MIN_TEXT_CHARS:
                    continue
                n_raw += store.write(make_record(
                    "reddit", entry.name, "https://www.reddit.com" + p.get("permalink", ""),
                    full, res.retrieved_at, title=p.get("title", ""),
                    publish_date=_date(p.get("created_utc")), license="Reddit API terms",
                    extra={"id": p.get("id"), "subreddit": p.get("subreddit"),
                           "score": p.get("score"), "num_comments": p.get("num_comments"),
                           "kind": "post"}))
            kept.sort(key=lambda p: -(p.get("num_comments") or 0))
            for p in [p for p in kept if (p.get("num_comments") or 0) >= 3][:MAX_COMMENT_POSTS]:
                if used() >= cap or budget.expired(now()):
                    complete = False
                    break
                cres = self._get(f"/comments/{p['id']}", {"limit": 40, "depth": 1,
                                                          "sort": "top", "raw_json": 1},
                                 budget.deadline)
                if not cres.ok:
                    continue
                listing = cres.json()
                children = listing[1]["data"]["children"] if len(listing) > 1 else []
                good = [c["data"] for c in children if c.get("kind") == "t1"
                        and (c["data"].get("score") or 0) >= MIN_COMMENT_SCORE
                        and c["data"].get("body", "") not in _REMOVED
                        and len(c["data"].get("body", "")) >= REDDIT_MIN_TEXT_CHARS]
                for c in good[:MAX_COMMENTS_PER_POST]:
                    n_raw += store.write(make_record(
                        "reddit", entry.name, "https://www.reddit.com" + c.get("permalink", ""),
                        c["body"], cres.retrieved_at, title=p.get("title", ""),
                        publish_date=_date(c.get("created_utc")), license="Reddit API terms",
                        extra={"id": c.get("id"), "parent_post": p.get("id"),
                               "subreddit": p.get("subreddit"), "score": c.get("score"),
                               "kind": "comment"}))
        except FetchError as exc:
            check_block(exc, "reddit")
            return SourceResult(n_raw=n_raw, n_requests=used() + n_extra,
                                error=str(exc)[:200])
        return SourceResult(n_raw=n_raw, n_requests=used() + n_extra, complete=complete)
