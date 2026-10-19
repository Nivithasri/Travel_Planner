"""Reddit collector (official OAuth API, application-only). Site-wide search with a place filter.

No subreddit list (D19): the query is "<place> travel", and a post is kept only when its title or
text names the place as a whole word. Top-level comments of the best few posts are stored as
their own records because closures, prices and timings are mostly in comments.

Needs REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET from .env. Without them the collector is
skipped, never faked. The bearer token is a secret: it is sent with `secret_headers`, so it is in
no cache key, cache file, manifest or message. The token request is a POST and is the one network
call that does not go through the disk cache (a cached token would be wrong and a secret).
Reddit's API terms apply to use of this data and must be reviewed before the corpus is shared.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.collect.raw import RawStore, make_record
from src.collect.types import Budget, SourceResult, check_block
from src.collect.web import place_keys
from src.common.http import FetchError, HttpClient
from src.common.schemas import PlaceRegistryEntry

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
DEFAULT_REQUESTS = 12  # 1 search + up to 11 comment fetches
MAX_COMMENT_POSTS = 6
MAX_COMMENTS_PER_POST = 12
MIN_COMMENT_SCORE = 2
MIN_TEXT_CHARS = 80
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
                if len(full) < MIN_TEXT_CHARS:
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
                        and len(c["data"].get("body", "")) >= MIN_TEXT_CHARS]
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
