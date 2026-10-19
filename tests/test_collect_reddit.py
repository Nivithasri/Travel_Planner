from __future__ import annotations

import json

from src.collect.raw import RawStore
from src.collect.reddit import RedditCollector, names_place
from src.collect.types import Budget
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry

UA = "Test/0.1 (x@example.org)"
LONG = "The Munnar road was closed for two days after the landslide in July, ask locally. " * 2
BIRYANI = "Try the dum biryani near the station, it is great. " * 3


def entry():
    return PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0)


def post(i, title, text, comments=5, over18=False):
    return {"kind": "t3", "data": {"id": i, "title": title, "selftext": text, "over_18": over18,
                                   "permalink": f"/r/kerala/comments/{i}/x/", "score": 10,
                                   "num_comments": comments, "created_utc": 1_717_200_000,
                                   "subreddit": "kerala"}}


SEARCH = {"data": {"children": [
    post("a1", "Munnar in monsoon", LONG),
    post("a2", "Best biryani in Chennai", BIRYANI),  # does not name the place
    post("a3", "Munnar nsfw", LONG, over18=True),
    post("a4", "Munnar", "[removed]", comments=0),
]}}
COMMENTS = [{"data": {"children": []}}, {"data": {"children": [
    {"kind": "t1", "data": {"id": "c1", "body": LONG, "score": 9,
                            "permalink": "/r/kerala/comments/a1/x/c1/",
                            "created_utc": 1_717_300_000}},
    {"kind": "t1", "data": {"id": "c2", "body": LONG, "score": 0, "permalink": "/p/c2/"}},
]}}]


def build(tmp_path, creds=("id", "secret"), routes=None):
    seen = []

    def transport(method, url, headers, timeout):
        seen.append((url, dict(headers)))
        for key, resp in (routes or {}).items():
            if url.startswith(key):
                return resp
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0)
    tokens = []
    col = RedditCollector(http, *creds, token_fn=lambda i, s, ua: tokens.append(i) or "TOK123")
    return col, http, seen, tokens


ROUTES = {"https://oauth.reddit.com/search": TransportResponse(200, {}, json.dumps(SEARCH)),
          "https://oauth.reddit.com/comments/a1": TransportResponse(200, {}, json.dumps(COMMENTS))}


def test_without_credentials_it_is_skipped_and_makes_no_request(tmp_path):
    col, http, seen, tokens = build(tmp_path, creds=(None, None), routes=ROUTES)
    r = col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.skipped and "credentials" in r.skipped
    assert seen == [] and tokens == [] and http.requests_made == 0


def test_collects_posts_and_good_comments_with_dates_and_filters(tmp_path):
    col, http, seen, tokens = build(tmp_path, routes=ROUTES)
    r = col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    recs = RawStore(tmp_path / "raw").read("reddit", "Munnar")
    kinds = sorted(x["extra"]["kind"] for x in recs)
    assert kinds == ["comment", "post"]  # off-topic, over_18, removed and low-score dropped
    post_rec = next(x for x in recs if x["extra"]["kind"] == "post")
    assert post_rec["publish_date"] == "2024-06-01" and post_rec["extra"]["score"] == 10
    assert r.error is None and r.n_raw == 2 and r.n_requests == 3  # token + search + comments
    assert all(h.get("Authorization") == "bearer TOK123" for u, h in seen if "oauth" in u)


def test_token_is_never_stored_in_cache_or_raw_files(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)
    col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    blob = "".join(p.read_text("utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert "TOK123" not in blob and "secret" not in blob


def test_second_run_uses_the_cache_for_searches(tmp_path):
    col, http, seen, _ = build(tmp_path, routes=ROUTES)
    col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    n = http.requests_made
    col2 = RedditCollector(http, "id", "secret", token_fn=lambda *a: "OTHER-TOKEN")
    r = col2.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert http.requests_made == n and r.n_raw == 0  # all cached, nothing new written


def test_request_cap_stops_comment_fetching_and_marks_incomplete(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)
    r = col.collect(entry(), Budget(caps={"reddit": 1}), lambda: 0, tmp_path / "raw")
    assert r.complete is False and r.n_raw == 1


def test_401_is_reported_and_token_dropped(tmp_path):
    col, *_ = build(tmp_path, routes={"https://oauth.reddit.com/search":
                                      TransportResponse(401, {}, "")})
    col.http.max_retries = 0
    r = col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error is not None
    assert col._token is None or "401" in r.error


def test_token_failure_is_an_error_without_leaking_the_secret(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)

    def boom(*a):
        raise RuntimeError("bad secret")

    col.token_fn = boom
    r = col.collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error == "Reddit token request failed: RuntimeError" and "secret" not in r.error


def test_place_name_filter_uses_whole_words():
    assert names_place("Trip to Munnar, Kerala", ["munnar"])
    assert not names_place("Munnarville", ["munnar"])
