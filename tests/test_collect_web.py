from __future__ import annotations

import json

import pytest

from src.collect.raw import RawStore
from src.collect.types import Budget
from src.collect.web import (
    SourcesConfigError,
    WebCollector,
    extract_page,
    load_allowlist,
    parse_sitemap,
    place_keys,
    url_names_place,
)
from src.common.cache import DiskCache
from src.common.http import HttpClient, TransportResponse
from src.common.schemas import PlaceRegistryEntry

UA = "Test/0.1 (x@example.org)"
ALLOW = {"example.org": {"domain": "example.org", "source_type": "reference",
                         "raw_source": "sahapedia", "license": "CC BY-NC-SA 4.0",
                         "min_delay_s": 0.0, "max_pages_per_place": 2}}
LONG = "Munnar tea estates and the Eravikulam park. " * 20
PAGE = ('<html><head><title>Munnar story</title>'
        '<meta property="article:published_time" content="2024-03-02T08:00:00Z"></head>'
        f'<body><nav>MENU</nav><main><p>{LONG}</p></main><footer>FOOT</footer>'
        '<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">cc</a>'
        '</body></html>')
SITEMAP = """<urlset><url><loc>https://example.org/places/munnar-tea</loc><lastmod>2025-01-09</lastmod></url>
<url><loc>https://example.org/places/hampi</loc></url>
<url><loc>https://example.org/blog/munnar</loc></url>
<url><loc>https://example.org/blog/munnar-3</loc></url></urlset>"""


def entry(name="Munnar", aliases=()):
    return PlaceRegistryEntry(name=name, aliases=list(aliases), state="KL", region_cluster="x",
                              popularity="popular", lat=10.0, lon=77.0)


def make(tmp_path, routes):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        r = routes.get(url)
        return r if r is not None else TransportResponse(404, {}, "")

    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0)
    return http, calls


def test_place_matching_uses_whole_words_only():
    keys = place_keys(entry("Munnar"))
    assert url_names_place("https://x.org/places/munnar-tea", keys)
    assert not url_names_place("https://x.org/places/munnarville", keys)
    assert place_keys(entry("Kodagu", ["Coorg", "Hyd"])) == ["coorg", "kodagu"]


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
    http, calls = make(tmp_path, routes)
    r = WebCollector(http, ALLOW).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
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
    http, _ = make(tmp_path, routes)
    r = WebCollector(http, ALLOW).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 0 and "differs" in (r.error or "")


def test_robots_disallow_means_nothing_is_fetched(tmp_path):
    routes = {"https://example.org/robots.txt": TransportResponse(
        200, {}, "User-agent: *\nDisallow: /\n"),
        "https://example.org/sitemap.xml": TransportResponse(200, {}, SITEMAP)}
    http, calls = make(tmp_path, routes)
    r = WebCollector(http, ALLOW).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
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
    from src.collect.web import applies_to
    assert applies_to({"states": ["KL"]}, entry())
    assert not applies_to({"states": ["AP"]}, entry())
    assert applies_to({"places": ["Munnar"]}, entry())
    assert not applies_to({"places": ["X"]}, entry())
    assert applies_to({}, entry())


def test_out_of_scope_domain_is_never_contacted(tmp_path):
    allow = {"example.org": {**ALLOW["example.org"], "states": ["AP"]}}
    http, calls = make(tmp_path, {})
    r = WebCollector(http, allow).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert calls == [] and r.n_raw == 0


def test_sitemap_listed_in_robots_on_the_bare_host_is_used(tmp_path):
    allow = {"www.example.org": {**ALLOW["example.org"], "domain": "www.example.org"}}
    routes = {
        "https://www.example.org/robots.txt": TransportResponse(
            200, {}, "User-agent: *\nDisallow: /admin\nSitemap: https://example.org/idx.xml\n"),
        "https://example.org/idx.xml": TransportResponse(
            200, {}, "<sitemapindex><sitemap><loc>https://example.org/p1.xml</loc></sitemap>"
                     "</sitemapindex>"),
        "https://example.org/p1.xml": TransportResponse(
            200, {}, "<urlset><url><loc>https://www.example.org/munnar</loc></url></urlset>"),
        "https://www.example.org/munnar": TransportResponse(200, {}, PAGE)}
    http, _ = make(tmp_path, routes)
    r = WebCollector(http, allow).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 1


def test_403_on_any_fetch_aborts_with_block_detected(tmp_path):
    from src.collect.types import BlockDetected
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(403, {}, "")}
    http, calls = make(tmp_path, routes)
    with pytest.raises(BlockDetected):
        WebCollector(http, ALLOW).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert calls[-1].endswith("/sitemap.xml") and len(calls) == 2  # nothing after the 403


def test_a_page_with_too_little_text_is_skipped_but_recorded(tmp_path):
    shell = "<html><head><title>App</title></head><body><div id='root'>Loading</div></body></html>"
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(
                  200, {}, "<urlset><url><loc>https://example.org/munnar</loc></url></urlset>"),
              "https://example.org/munnar": TransportResponse(200, {}, shell)}
    http, _ = make(tmp_path, routes)
    manifest = tmp_path / "manifest.jsonl"
    r = WebCollector(http, ALLOW, manifest=manifest).collect(
        entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 0 and not RawStore(tmp_path / "raw").read("sahapedia", "Munnar")
    assert any("munnar" in n and "too_short" in n and "chars" in n for n in r.notes)
    line = json.loads(manifest.read_text("utf-8").splitlines()[0])
    assert line["url"] == "https://example.org/munnar" and line["skipped"] == "too_short"
    assert line["chars"] < line["min_chars"] == 400 and line["place"] == "Munnar"
    assert "body" not in line and "text" not in line  # no content in the manifest


def test_discovery_gaps_are_kept_as_notes_even_when_a_page_was_collected(tmp_path):
    allow = {**ALLOW, "other.org": {**ALLOW["example.org"], "domain": "other.org"}}
    routes = {"https://example.org/robots.txt": TransportResponse(404, {}, ""),
              "https://example.org/sitemap.xml": TransportResponse(
                  200, {}, "<urlset><url><loc>https://example.org/munnar</loc></url></urlset>"),
              "https://example.org/munnar": TransportResponse(200, {}, PAGE),
              "https://other.org/robots.txt": TransportResponse(404, {}, ""),
              "https://other.org/sitemap.xml": TransportResponse(404, {}, "")}
    http, _ = make(tmp_path, routes)
    r = WebCollector(http, allow).collect(entry(), Budget(), lambda: 0, tmp_path / "raw")
    assert r.n_raw == 1 and r.error is None
    assert any("other.org" in n and "no sitemap URL names Munnar" in n for n in r.notes)
