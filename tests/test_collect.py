"""Tests for src/collect: wiki, web and Reddit collectors, blocks, coverage, ensure_place,
the YouTube refusal (D27) (D47)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.collect.run import (
    SOURCES,
    BlockDetected,
    Budget,
    HoldoutError,
    PlaceIngestor,
    PlaceStore,
    RawRecord,
    RawStore,
    SourceResult,
    build_collectors,
    core_only_resolver,
    load_run_config,
    make_http,
    make_record,
    passes_rule,
    place_coverage,
    render,
)
from src.collect.sources import (
    RedditCollector,
    SourcesConfigError,
    WebCollector,
    WikiCollector,
    extract_page,
    load_allowlist,
    names_place,
    parse_sitemap,
    place_keys,
    url_names_place,
)
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import REPO_ROOT, CollectSettings, PlaceRegistryEntry
from src.places.places import Registry, Resolution

# ==== from test_collect_wiki.py


WIKI_UA = "Test/0.1 (x@example.org)"


def wiki_entry(**kw):
    base = dict(name="Munnar", aliases=[], state="KL", region_cluster="KL-Idukki",
                popularity="popular", lat=10.0, lon=77.0, wikipedia_title="Munnar",
                wikivoyage_title="Munnar")
    return PlaceRegistryEntry(**{**base, **kw})


def api(page):
    return TransportResponse(200, {}, json.dumps({"query": {"pages": [page]}}))


def client(tmp_path, response):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        return response

    http = HttpClient(DiskCache(tmp_path / "c"), WIKI_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0)
    return http, calls


GOOD = {"title": "Munnar", "extract": "Munnar is a hill station.\n\n== See ==\nTea museum.",
        "revisions": [{"revid": 7, "timestamp": "2026-05-04T10:00:00Z"}]}


def test_collects_one_record_with_lastmod_not_publish_date(tmp_path):
    http, calls = client(tmp_path, api(GOOD))
    col = WikiCollector("wikipedia", http, tmp_path / "m.jsonl")
    r = col.collect(wiki_entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    assert (r.n_raw, r.error, r.n_requests) == (1, None, 1)
    [rec] = RawStore(tmp_path / "raw").read("wikipedia", "Munnar")
    RawRecord(**rec)
    assert rec["publish_date"] is None and rec["lastmod"] == "2026-05-04"
    assert rec["url"] == "https://en.wikipedia.org/wiki/Munnar" and rec["extra"]["revid"] == 7
    assert json.loads((tmp_path / "m.jsonl").read_text("utf-8"))["sha256"] == rec["content_sha256"]
    assert calls and all("wikipedia.org" in c for c in calls)


def test_same_content_twice_is_not_duplicated(tmp_path):
    http, _ = client(tmp_path, api(GOOD))
    col = WikiCollector("wikivoyage", http)
    col.collect(wiki_entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    r2 = col.collect(wiki_entry(), Budget(), lambda: 1.7e9, tmp_path / "raw")
    assert r2.n_raw == 0 and len(RawStore(tmp_path / "raw").read("wikivoyage", "Munnar")) == 1


def test_no_article_is_skipped_without_network(tmp_path):
    http, calls = client(tmp_path, api(GOOD))
    r = WikiCollector("wikivoyage", http).collect(
        wiki_entry(wikivoyage_title=None), Budget(), lambda: 0, tmp_path / "raw")
    assert r.skipped and calls == [] and http.requests_made == 0


def test_missing_page_is_an_error_not_an_empty_record(tmp_path):
    http, _ = client(tmp_path, api({"title": "X", "missing": True}))
    r = WikiCollector("wikipedia", http).collect(wiki_entry(), Budget(), lambda: 0,
                                                 tmp_path / "raw")
    assert r.error and r.n_raw == 0
    assert RawStore(tmp_path / "raw").read("wikipedia", "Munnar") == []


def test_http_error_is_reported(tmp_path):
    http, _ = client(tmp_path, TransportResponse(404, {}, ""))
    http.max_retries = 0
    r = WikiCollector("wikipedia", http).collect(wiki_entry(), Budget(), lambda: 0,
                                                 tmp_path / "raw")
    assert r.error == "HTTP 404 from wikipedia" and r.n_raw == 0


def test_record_validation_rejects_bad_dates_empty_body_and_unknown_source():
    with pytest.raises(ValueError):
        make_record("wikipedia", "P", "u", "text", 0.0, publish_date="12/05/2024")
    with pytest.raises(ValueError):
        make_record("wikipedia", "P", "u", "", 0.0)
    with pytest.raises(ValueError):
        make_record("twitter", "P", "u", "t", 0.0)


@pytest.mark.parametrize("status", [403, 429])
def test_block_status_aborts_the_run(tmp_path, status):
    from src.collect.run import BlockDetected
    http, calls = client(tmp_path, TransportResponse(status, {}, ""))
    http.max_retries = 0
    with pytest.raises(BlockDetected):
        WikiCollector("wikipedia", http).collect(wiki_entry(), Budget(), lambda: 0,
                                                 tmp_path / "raw")
    assert len(calls) == 1  # no retry


# ==== from test_collect_web.py


WEB_UA = "Test/0.1 (x@example.org)"


WEB_ALLOW = {"example.org": {"domain": "example.org", "source_type": "reference",
                         "raw_source": "sahapedia", "license": "CC BY-NC-SA 4.0",
                         "min_delay_s": 0.0, "max_pages_per_place": 2}}


WEB_LONG = "Munnar tea estates and the Eravikulam park. " * 20
PAGE = ('<html><head><title>Munnar story</title>'
        '<meta property="article:published_time" content="2024-03-02T08:00:00Z"></head>'
        f'<body><nav>MENU</nav><main><p>{WEB_LONG}</p></main><footer>FOOT</footer>'
        '<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">cc</a>'
        '</body></html>')


SITEMAP = """<urlset><url><loc>https://example.org/places/munnar-tea</loc><lastmod>2025-01-09</lastmod></url>
<url><loc>https://example.org/places/hampi</loc></url>
<url><loc>https://example.org/blog/munnar</loc></url>
<url><loc>https://example.org/blog/munnar-3</loc></url></urlset>"""


def web_entry(name="Munnar", aliases=()):
    return PlaceRegistryEntry(name=name, aliases=list(aliases), state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0)


def web_make(tmp_path, routes):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        r = routes.get(url)
        return r if r is not None else TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), WEB_UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    return http, calls


def test_place_matching_uses_whole_words_only():
    keys = place_keys(web_entry("Munnar"))
    assert url_names_place("https://x.org/places/munnar-tea", keys)
    assert not url_names_place("https://x.org/places/munnarville", keys)
    assert place_keys(web_entry("Kodagu", ["Coorg", "Hyd"])) == ["coorg", "kodagu"]


def test_sitemap_parsing_pages_and_nested():
    pages, nested = parse_sitemap(SITEMAP)
    assert ("https://example.org/places/munnar-tea", "2025-01-09") in pages and nested == []
    _, nested = parse_sitemap("<sitemapindex><sitemap><loc>https://example.org/s2.xml</loc>"
                              "</sitemap></sitemapindex>")
    assert nested == ["https://example.org/s2.xml"]


def test_extract_page_drops_boilerplate_and_reads_the_page_date():
    p = extract_page(PAGE)
    assert "MENU" not in p["text"] and "FOOT" not in p["text"] and "Eravikulam" in p["text"]
    assert p["publish_date"] == "2024-03-02" and p["cc_licenses"] == ["by-nc-sa/4.0"]


def test_collects_matching_pages_up_to_the_cap_with_lastmod_separate(tmp_path):
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(200, {}, SITEMAP),
              **{f"https://example.org/{p}": TransportResponse(200, {}, PAGE)
                 for p in ("places/munnar-tea", "blog/munnar", "blog/munnar-3")}}
    http, calls = web_make(tmp_path, routes)
    r = WebCollector(http, WEB_ALLOW).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    recs = RawStore(tmp_path / "raw").read("sahapedia", "Munnar")
    assert r.n_raw == 2 == len(recs)  # cap of 2 pages
    first = recs[0]
    assert first["publish_date"] == "2024-03-02" and first["lastmod"] == "2025-01-09"
    assert first["license"] == "CC BY-NC-SA 4.0"
    assert not any("hampi" in c for c in calls)


def test_page_under_another_cc_licence_is_skipped(tmp_path):
    other = PAGE.replace("by-nc-sa/4.0", "by-nd/4.0")
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(
                  200, {}, "<urlset><url><loc>https://example.org/munnar</loc></url></urlset>"),
              "https://example.org/munnar": TransportResponse(200, {}, other)}
    http, _ = web_make(tmp_path, routes)
    r = WebCollector(http, WEB_ALLOW).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 0 and "differs" in (r.error or "")


def test_robots_disallow_means_nothing_is_fetched(tmp_path):
    routes = {"https://example.org/robots.txt": TransportResponse(
        200, {}, "User-agent: *\nDisallow: /\n"),
        "https://example.org/sitemap.xml": TransportResponse(200, {}, SITEMAP)}
    http, calls = web_make(tmp_path, routes)
    r = WebCollector(http, WEB_ALLOW).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 0 and calls == ["https://example.org/robots.txt"]


def test_shipped_config_allowlist_and_ask_first_are_disjoint_and_scoped():
    allow = load_allowlist()
    assert set(allow) == {"sahapedia.org", "aptourism.gov.in", "www.keralatourism.org",
                          "www.inditales.com", "tirumala.org", "hrce.tn.gov.in"}
    assert {d["raw_source"] for d in allow.values()} == {"sahapedia", "official", "blog"}
    assert allow["aptourism.gov.in"]["states"] == ["AP"] and allow["tirumala.org"]["places"]
    accepted = [d for d in allow.values() if "accepted_by_owner" in d]
    assert len(accepted) == 4
    assert all("user accepted risk on 2026-10-18" in d["basis"] for d in accepted)


def test_a_domain_both_allowed_and_ask_first_is_refused(tmp_path):
    p = tmp_path / "s.json"
    dom = {"domain": "a.org", "source_type": "official", "raw_source": "official",
           "license": "x", "min_delay_s": 1, "max_pages_per_place": 1}
    p.write_text(json.dumps({"web_allowlist": [dom], "ask_first": ["a.org"]}))
    with pytest.raises(SourcesConfigError):
        load_allowlist(p)


def test_domain_scoping_by_state_and_place(tmp_path):
    from src.collect.sources import applies_to
    assert applies_to({"states": ["KL"]}, web_entry())
    assert not applies_to({"states": ["AP"]}, web_entry())
    assert applies_to({"places": ["Munnar"]}, web_entry())
    assert not applies_to({"places": ["X"]}, web_entry())
    assert applies_to({}, web_entry())


def test_out_of_scope_domain_is_never_contacted(tmp_path):
    allow = {"example.org": {**WEB_ALLOW["example.org"], "states": ["AP"]}}
    http, calls = web_make(tmp_path, {})
    r = WebCollector(http, allow).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert calls == [] and r.n_raw == 0


def test_sitemap_listed_in_robots_on_the_bare_host_is_used(tmp_path):
    allow = {"www.example.org": {**WEB_ALLOW["example.org"], "domain": "www.example.org"}}
    routes = {
        "https://www.example.org/robots.txt": TransportResponse(
            200, {}, "User-agent: *\nDisallow: /admin\nSitemap: https://example.org/idx.xml\n"),
        "https://example.org/idx.xml": TransportResponse(
            200, {}, "<sitemapindex><sitemap><loc>https://example.org/p1.xml</loc></sitemap>"
                     "</sitemapindex>"),
        "https://example.org/p1.xml": TransportResponse(
            200, {}, "<urlset><url><loc>https://www.example.org/munnar</loc></url></urlset>"),
        "https://www.example.org/munnar": TransportResponse(200, {}, PAGE)}
    http, _ = web_make(tmp_path, routes)
    r = WebCollector(http, allow).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 1


def test_403_on_any_fetch_aborts_with_block_detected(tmp_path):
    from src.collect.run import BlockDetected
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(403, {}, "")}
    http, calls = web_make(tmp_path, routes)
    with pytest.raises(BlockDetected):
        WebCollector(http, WEB_ALLOW).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert calls[-1].endswith("/sitemap.xml") and len(calls) == 2  # nothing after the 403


def test_a_page_with_too_little_text_is_skipped_but_recorded(tmp_path):
    shell = "<html><head><title>App</title></head><body><div id='root'>Loading</div></body></html>"
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(
                  200, {}, "<urlset><url><loc>https://example.org/munnar</loc></url></urlset>"),
              "https://example.org/munnar": TransportResponse(200, {}, shell)}
    http, _ = web_make(tmp_path, routes)
    manifest = tmp_path / "manifest.jsonl"
    r = WebCollector(http, WEB_ALLOW, manifest=manifest).collect(
        web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 0 and not RawStore(tmp_path / "raw").read("sahapedia", "Munnar")
    assert any("munnar" in n and "too_short" in n and "chars" in n for n in r.notes)
    line = json.loads(manifest.read_text("utf-8").splitlines()[0])
    assert line["url"] == "https://example.org/munnar" and line["skipped"] == "too_short"
    assert line["chars"] < line["min_chars"] == 400 and line["place"] == "Munnar"
    assert "body" not in line and "text" not in line  # no content in the manifest


def test_discovery_gaps_are_kept_as_notes_even_when_a_page_was_collected(tmp_path):
    allow = {**WEB_ALLOW, "other.org": {**WEB_ALLOW["example.org"], "domain": "other.org"}}
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(
                  200, {}, "<urlset><url><loc>https://example.org/munnar</loc></url></urlset>"),
              "https://example.org/munnar": TransportResponse(200, {}, PAGE),
              "https://other.org/robots.txt": TransportResponse(404, {}, ""),
              "https://other.org/sitemap.xml": TransportResponse(404, {}, "")}
    http, _ = web_make(tmp_path, routes)
    r = WebCollector(http, allow).collect(web_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 1 and r.error is None
    assert any("other.org" in n and "no sitemap URL names Munnar" in n for n in r.notes)


# ==== from test_collect_reddit.py


REDDIT_UA = "Test/0.1 (x@example.org)"


REDDIT_LONG = ("The Munnar road was closed for two days after the landslide in July, "
               "ask locally. ") * 2


BIRYANI = "Try the dum biryani near the station, it is great. " * 3


def reddit_entry():
    return PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0)


def post(i, title, text, comments=5, over18=False):
    return {"kind": "t3", "data": {"id": i, "title": title, "selftext": text, "over_18": over18,
                                   "permalink": f"/r/kerala/comments/{i}/x/", "score": 10,
                                   "num_comments": comments, "created_utc": 1_717_200_000,
                                   "subreddit": "kerala"}}


SEARCH = {"data": {"children": [
    post("a1", "Munnar in monsoon", REDDIT_LONG),
    post("a2", "Best biryani in Chennai", BIRYANI),  # does not name the place
    post("a3", "Munnar nsfw", REDDIT_LONG, over18=True),
    post("a4", "Munnar", "[removed]", comments=0),
]}}


COMMENTS = [{"data": {"children": []}}, {"data": {"children": [
    {"kind": "t1", "data": {"id": "c1", "body": REDDIT_LONG, "score": 9,
                            "permalink": "/r/kerala/comments/a1/x/c1/",
                            "created_utc": 1_717_300_000}},
    {"kind": "t1", "data": {"id": "c2", "body": REDDIT_LONG, "score": 0, "permalink": "/p/c2/"}},
]}}]


def build(tmp_path, creds=("id", "secret"), routes=None):
    seen = []

    def transport(method, url, headers, timeout):
        seen.append((url, dict(headers)))
        for key, resp in (routes or {}).items():
            if url.startswith(key):
                return resp
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), REDDIT_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0)
    tokens = []
    col = RedditCollector(http, *creds, token_fn=lambda i, s, ua: tokens.append(i) or "TOK123")
    return col, http, seen, tokens


ROUTES = {"https://oauth.reddit.com/search": TransportResponse(200, {}, json.dumps(SEARCH)),
          "https://oauth.reddit.com/comments/a1": TransportResponse(200, {}, json.dumps(COMMENTS))}


def test_without_credentials_it_is_skipped_and_makes_no_request(tmp_path):
    col, http, seen, tokens = build(tmp_path, creds=(None, None), routes=ROUTES)
    r = col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.skipped and "credentials" in r.skipped
    assert seen == [] and tokens == [] and http.requests_made == 0


def test_collects_posts_and_good_comments_with_dates_and_filters(tmp_path):
    col, http, seen, tokens = build(tmp_path, routes=ROUTES)
    r = col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    recs = RawStore(tmp_path / "raw").read("reddit", "Munnar")
    kinds = sorted(x["extra"]["kind"] for x in recs)
    assert kinds == ["comment", "post"]  # off-topic, over_18, removed and low-score dropped
    post_rec = next(x for x in recs if x["extra"]["kind"] == "post")
    assert post_rec["publish_date"] == "2024-06-01" and post_rec["extra"]["score"] == 10
    assert r.error is None and r.n_raw == 2 and r.n_requests == 3  # token + search + comments
    assert all(h.get("Authorization") == "bearer TOK123" for u, h in seen if "oauth" in u)


def test_token_is_never_stored_in_cache_or_raw_files(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)
    col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    blob = "".join(p.read_text("utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert "TOK123" not in blob and "secret" not in blob


def test_second_run_uses_the_cache_for_searches(tmp_path):
    col, http, seen, _ = build(tmp_path, routes=ROUTES)
    col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    n = http.requests_made
    col2 = RedditCollector(http, "id", "secret", token_fn=lambda *a: "OTHER-TOKEN")
    r = col2.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert http.requests_made == n and r.n_raw == 0  # all cached, nothing new written


def test_request_cap_stops_comment_fetching_and_marks_incomplete(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)
    r = col.collect(reddit_entry(), Budget(caps={"reddit": 1}), lambda: 0, tmp_path / "raw")
    assert r.complete is False and r.n_raw == 1


def test_401_is_reported_and_token_dropped(tmp_path):
    col, *_ = build(tmp_path, routes={"https://oauth.reddit.com/search":
                                      TransportResponse(401, {}, "")})
    col.http.max_retries = 0
    r = col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error is not None
    assert col._token is None or "401" in r.error


def test_token_failure_is_an_error_without_leaking_the_secret(tmp_path):
    col, *_ = build(tmp_path, routes=ROUTES)

    def boom(*a):
        raise RuntimeError("bad secret")

    col.token_fn = boom
    r = col.collect(reddit_entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.error == "Reddit token request failed: RuntimeError" and "secret" not in r.error


def test_place_name_filter_uses_whole_words():
    assert names_place("Trip to Munnar, Kerala", ["munnar"])
    assert not names_place("Munnarville", ["munnar"])


# ==== from test_collect_blocks.py: A block (HTTP 403/429, transcript block) in ANY source aborts
# the whole run, even when other


BLOCKS_UA = "Test/0.1 (x@example.org)"


BLOCKS_ENTRY = PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=10.0, lon=77.0, wikivoyage_title="Munnar")


WIKI_OK = json.dumps({"query": {"pages": [{"title": "Munnar", "extract": "Munnar text " * 30,
                                           "revisions": [{"revid": 1,
                                                          "timestamp": "2026-01-02T00:00:00Z"}]}]}})


BLOCKS_ALLOW = {"example.org": {"domain": "example.org", "source_type": "official",
                         "raw_source": "official", "license": "x", "min_delay_s": 0.0,
                         "max_pages_per_place": 2}}


class Third:
    source = "reddit"
    calls = 0

    def collect(self, entry, budget, now, raw_root):
        Third.calls += 1
        return SourceResult()


def run(tmp_path, sitemap_status):
    def transport(method, url, headers, timeout):
        if "wikivoyage" in url:
            return TransportResponse(200, {}, WIKI_OK)
        if url.endswith("/robots.txt"):
            return TransportResponse(404, {}, "")
        if url.endswith("/sitemap.xml"):
            return TransportResponse(sitemap_status, {}, "")
        return TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), BLOCKS_UA, transport=transport,
                      sleep=lambda s: None, min_delay_s=0, max_retries=0)
    Third.calls = 0
    cols = [WikiCollector("wikivoyage", http), WebCollector(http, BLOCKS_ALLOW), Third()]
    store = PlaceStore(tmp_path / "live")
    ing = PlaceIngestor(store, Registry(), cols, lambda t, r: Resolution("resolved", BLOCKS_ENTRY),
                        clock=lambda: 1_000_000.0, request_counter=lambda: http.requests_made)
    return ing, store


@pytest.mark.parametrize("status", [403, 429])
def test_block_in_one_source_aborts_the_run_after_another_source_succeeded(tmp_path, status):
    ing, store = run(tmp_path, status)
    with pytest.raises(BlockDetected):
        ing.ensure_place("munnar")
    assert Third.calls == 0  # nothing after the blocked source was tried
    # the earlier success is kept on disk and in the status file
    assert len(RawStore(store.raw_root).read("wikivoyage", "Munnar")) == 1
    assert store.load("Munnar")["sources"]["wikivoyage"]["n_raw"] == 1
    assert "web" not in store.load("Munnar")["sources"]  # the blocked source is not recorded done


def test_without_a_block_the_same_run_completes(tmp_path):
    ing, _ = run(tmp_path, 404)
    st = ing.ensure_place("munnar")
    assert st.status == "collected" and Third.calls == 1


# ==== from test_coverage_orchestrator.py


def cov_entry(name, holdout=None):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0, holdout=holdout)


def put(root, source, place, n=1):
    for i in range(n):
        RawStore(root).write(make_record(source, place, f"https://x/{source}/{i}", "text", 0.0))


def test_rule_needs_two_types_and_one_outside_the_reference_family():
    assert not passes_rule({"wikivoyage", "reference"})  # two types, both reference family
    assert not passes_rule({"reddit"})
    assert passes_rule({"wikivoyage", "reddit"})
    assert passes_rule({"reference", "official"})
    assert not passes_rule(set())


def test_place_coverage_maps_raw_sources_to_types(tmp_path):
    put(tmp_path, "wikipedia", "Munnar")
    put(tmp_path, "sahapedia", "Munnar")
    c = place_coverage(cov_entry("Munnar"), tmp_path)
    assert c.types == ["reference"] and not c.passes and c.reason == "only 1 source type(s)"
    put(tmp_path, "wikivoyage", "Munnar")
    two = place_coverage(cov_entry("Munnar"), tmp_path)
    assert not two.passes and "reference-wiki" in two.reason
    put(tmp_path, "youtube", "Munnar")
    assert place_coverage(cov_entry("Munnar"), tmp_path).passes


def test_holdout_is_listed_but_never_counted(tmp_path):
    put(tmp_path, "reddit", "Thekkady")
    put(tmp_path, "youtube", "Thekkady")
    text, data = render([cov_entry("Munnar"), cov_entry("Thekkady", holdout=True)], tmp_path,
                        tmp_path / "status", {"reddit": "skipped: no credentials"})
    assert data["counted"] == 1 and data["passing"] == 0
    assert "0 of 1" in text and "hold-out" in text
    assert "**reddit**: skipped: no credentials" in text


def test_skipped_collectors_come_from_status_files_too(tmp_path):
    status = tmp_path / "status" / "places"
    status.mkdir(parents=True)
    (status / "munnar.json").write_text(json.dumps(
        {"sources": {"youtube": {"skipped": "no YOUTUBE_API_KEY in .env"},
                     "web": {"error": "sahapedia.org: no sitemap"}}}))
    text, _ = render([cov_entry("Munnar")], tmp_path, tmp_path / "status")
    assert "skipped: no YOUTUBE_API_KEY" in text and "error: sahapedia.org" in text


def test_snapshot_resolver_refuses_holdouts_and_unknowns():
    reg = Registry([cov_entry("Munnar"), cov_entry("Thekkady", holdout=True)])
    assert core_only_resolver("munnar", [reg]).status == "resolved"
    assert core_only_resolver("Atlantis", [reg]).status == "not_found"
    with pytest.raises(HoldoutError):
        core_only_resolver("Thekkady", [reg])


def test_build_collectors_in_fixed_order_and_unknown_source_is_refused(tmp_path):
    cfg = CollectSettings("a@example.org", "UA a@example.org", tmp_path)
    from src.collect.run import make_http
    cols = build_collectors(cfg, make_http(cfg), None, ["youtube", "wikivoyage"])
    assert [c.source for c in cols] == ["wikivoyage", "youtube"]
    with pytest.raises(ValueError):
        build_collectors(cfg, make_http(cfg), None, ["twitter"])


def test_nothing_collected_is_reported_as_not_collected_yet(tmp_path):
    assert place_coverage(cov_entry("Munnar"), tmp_path).reason == "nothing collected yet"


# ==== from test_ensure_place.py


DAY = 86400.0


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def ensure_entry(name="Munnar"):
    return PlaceRegistryEntry(name=name, aliases=[], state="KL", region_cluster="KL-Idukki",
                              popularity="popular", lat=10.0, lon=77.0)


class FakeCollector:
    def __init__(self, source, clock, counter, cost=1, result=None, tick=0.0):
        self.source, self.clock, self.counter = source, clock, counter
        self.cost, self.result, self.tick, self.calls = cost, result, tick, 0

    def collect(self, ensure_entry, budget, now, raw_root):
        self.calls += 1
        self.counter[0] += self.cost
        self.clock.t += self.tick
        return self.result or SourceResult(n_raw=3, n_requests=self.cost)


def ensure_make(tmp_path, resolution=None, sources=("wikivoyage", "reddit"), **kw):
    clock, counter = FakeClock(), [0]
    cols = [FakeCollector(s, clock, counter, **kw.get(s, {})) for s in sources]
    res = resolution or Resolution("resolved", ensure_entry())
    ing = PlaceIngestor(PlaceStore(tmp_path / "live"), Registry(), cols, lambda t, regs: res,
                        clock=clock, request_counter=lambda: counter[0])
    return ing, clock, cols, counter


def test_new_place_is_collected_then_a_repeat_makes_no_network_calls(tmp_path):
    ing, clock, cols, _ = ensure_make(tmp_path)
    first = ing.ensure_place("munnar")
    assert first.status == "collected" and first.network_requests == 2
    assert first.chunk_counts is None and first.state == "KL"
    second = ing.ensure_place("munnar")
    assert second.status == "collected" and second.network_requests == 0
    assert [c.calls for c in cols] == [1, 1]


def test_only_expired_sources_are_refreshed(tmp_path):
    ing, clock, cols, _ = ensure_make(tmp_path)
    ing.ensure_place("munnar")
    clock.t += 8 * DAY  # reddit TTL is 7 days, wikivoyage 30
    again = ing.ensure_place("munnar")
    assert [c.calls for c in cols] == [1, 2]
    assert again.status == "collected" and again.network_requests == 1


def test_expired_source_with_no_budget_left_is_reported_stale(tmp_path):
    ing, clock, cols, _ = ensure_make(tmp_path)
    ing.ensure_place("munnar")
    clock.t += 8 * DAY
    st = ing.ensure_place("munnar", Budget(deadline=clock.t - 1))
    assert st.status == "stale" and st.network_requests == 0 and "reddit" in st.reason


def test_deadline_mid_run_returns_ingesting_and_a_later_call_resumes(tmp_path):
    ing, clock, cols, _ = ensure_make(tmp_path, wikivoyage={"tick": 5.0})
    st = ing.ensure_place("munnar", Budget(deadline=clock.t + 3))
    assert st.status == "ingesting" and "reddit" in st.reason
    assert [c.calls for c in cols] == [1, 0]
    done = ing.ensure_place("munnar")
    assert done.status == "collected" and [c.calls for c in cols] == [1, 1]


def test_expired_deadline_before_anything_runs_is_new(tmp_path):
    ing, clock, cols, _ = ensure_make(tmp_path)
    st = ing.ensure_place("munnar", Budget(deadline=clock.t - 1))
    assert st.status == "new" and all(c.calls == 0 for c in cols)


def test_incomplete_source_is_resumed(tmp_path):
    partial = SourceResult(n_raw=1, n_requests=1, complete=False)
    ing, clock, cols, _ = ensure_make(tmp_path, sources=("reddit",), reddit={"result": partial})
    assert ing.ensure_place("munnar").status == "ingesting"
    cols[0].result = None
    assert ing.ensure_place("munnar").status == "collected" and cols[0].calls == 2


def test_error_is_recorded_not_retried_for_an_hour_and_skipped_is_not_an_error(tmp_path):
    err = SourceResult(error="HTTP 500")
    skip = SourceResult(skipped="no credentials")
    ing, clock, cols, _ = ensure_make(tmp_path, reddit={"result": skip}, wikivoyage={"result": err})
    st = ing.ensure_place("munnar")
    assert st.status == "collected"
    assert st.sources["wikivoyage"]["error"] == "HTTP 500"
    assert st.sources["reddit"]["skipped"] == "no credentials"
    ing.ensure_place("munnar")
    assert cols[0].calls == 1  # error not retried within the hour
    assert cols[1].calls == 2  # skipped sources are looked at again (no network)
    clock.t += 3601
    ing.ensure_place("munnar")
    assert cols[0].calls == 2


def test_rejected_and_ambiguous_ingest_nothing(tmp_path):
    for res in (Resolution("rejected", reason="outside scope"),
                Resolution("ambiguous", candidates=[{"name": "A"}, {"name": "B"}])):
        ing, _, cols, _ = ensure_make(tmp_path, resolution=res)
        st = ing.ensure_place("x")
        assert st.status == "rejected" and st.place is None
        assert all(c.calls == 0 for c in cols)
    assert st.candidates == [{"name": "A"}, {"name": "B"}]
    assert not (tmp_path / "live" / "places").exists()


def test_new_place_is_added_to_the_live_registry_only(tmp_path):
    res = Resolution("resolved", ensure_entry("Vagamon"), new=True)
    ing, *_ = ensure_make(tmp_path, resolution=res)
    ing.ensure_place("vagamon")
    assert PlaceStore(tmp_path / "live").load_registry().get("Vagamon") is not None


ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "collect", ROOT / "scripts" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==== from test_youtube_policy.py: D27: YouTube transcripts are dropped. The collector must refuse
# to run them, with zero


YT_ENTRY = PlaceRegistryEntry(name="Munnar", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=10.0, lon=77.0,
                           wikipedia_title="Munnar", wikivoyage_title="Munnar")


WIKI = json.dumps({"query": {"pages": [{"title": "Munnar", "extract": "Munnar text " * 40,
                                        "revisions": [{"revid": 1,
                                                       "timestamp": "2026-01-02T00:00:00Z"}]}]}})


class Spy:
    def __init__(self):
        self.urls = []

    def __call__(self, method, url, headers, timeout):
        self.urls.append(url)
        if "api.php" in url:
            return TransportResponse(200, {}, WIKI)
        return TransportResponse(404, {}, "")


def cfg(tmp_path):
    return CollectSettings("someone@example.org", "UA/0.1 (someone@example.org)",
                           tmp_path / "cache", youtube_api_key="FAKE-KEY-FOR-TEST")


def ingestor(tmp_path, only=None):
    spy = Spy()
    c = cfg(tmp_path)
    http = make_http(c, transport=spy, request_log=None)
    cols = build_collectors(c, http, None, only)
    http.min_delay_s = 0  # collectors set per-host delays; no real waiting in tests
    http.host_delays.clear()
    http._sleep = lambda s: None
    store = PlaceStore(tmp_path / "live")
    ing = PlaceIngestor(store, Registry(), cols, lambda t, r: Resolution("resolved", YT_ENTRY),
                        clock=lambda: 1_000_000.0, request_counter=lambda: http.requests_made)
    return ing, spy, store, cols


def test_youtube_is_refused_d27_with_zero_requests_and_no_youtube_network_path(tmp_path):
    # shipped config: transcripts off; "youtube" stays a known source name, off by default
    doc = json.loads((REPO_ROOT / "config" / "sources.json").read_text("utf-8"))
    assert doc["youtube_transcripts"] is False
    assert load_run_config()["youtube_transcripts"] is False
    assert "youtube" in SOURCES
    c = cfg(tmp_path / "defaults")
    cols = build_collectors(c, make_http(c, transport=Spy(), request_log=None), None)
    assert [x.source for x in cols] == ["wikivoyage", "wikipedia", "web", "reddit"]
    # a full default run never touches googleapis.com or youtube.com
    ing, spy, store, cols = ingestor(tmp_path / "full")
    st = ing.ensure_place("munnar", Budget())
    assert st.status == "collected" and "youtube" not in st.sources
    assert spy.urls, "the run did make (fake) requests"
    assert not [u for u in spy.urls if "googleapis.com" in u or "youtube.com" in u]
    assert not (store.raw_root / "youtube").exists()
    # an explicit request is reported as skipped (D27), not as a failure, with zero requests
    ing, spy, store, cols = ingestor(tmp_path / "explicit", only=["wikipedia", "youtube"])
    st = ing.ensure_place("munnar", Budget())
    rec = st.sources["youtube"]
    assert st.status == "collected" and rec["error"] is None
    assert "D27" in rec["skipped"] and rec["n_raw"] == 0 and rec["n_requests"] == 0
    assert not [u for u in spy.urls if "googleapis.com" in u or "youtube.com" in u]


def test_resume_does_not_retry_or_change_a_skipped_or_withdrawn_youtube_record(tmp_path):
    ing, spy, store, cols = ingestor(tmp_path, only=["wikipedia", "youtube"])
    ing.ensure_place("munnar", Budget())
    n = len(spy.urls)
    again = ing.ensure_place("munnar", Budget())
    assert len(spy.urls) == n  # everything cached or skipped: no new network call at all
    assert again.network_requests == 0 and "D27" in again.sources["youtube"]["skipped"]
    # a record marked withdrawn (D27) is left alone when youtube is not run
    doc = store.load("Munnar")
    doc["sources"]["youtube"] = {"skipped": "withdrawn (D27)", "n_raw": 0, "n_requests": 0,
                                 "error": None, "ingested_at": None, "expires_at": None,
                                 "complete": True}
    store.save("Munnar", doc)
    ing2, _, store2, _ = ingestor(tmp_path)  # default list: no youtube collector
    st = ing2.ensure_place("munnar", Budget())
    assert st.sources["youtube"]["skipped"] == "withdrawn (D27)" and st.status == "collected"
