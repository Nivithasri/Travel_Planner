"""Gather the evidence behind docs/sources.md: for every candidate domain, its live robots.txt
(rules for our User-Agent, fully blocked crawlers, crawl-delay, sitemaps), whether the homepage is
reachable, and the sentences of its terms / copyright pages that mention crawling, copying,
commercial use or licences.

Live network use through the polite cached client (robots.txt is checked before every page, a
missing robots.txt means no rules, an unreachable one means disallowed). Needs CONTACT_EMAIL
in .env.
This script only reads; it does not collect content for the corpus.

    python scripts/gather_source_evidence.py [--out data/manifests/source_evidence.json]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.allowlist import refuse_without_historical  # noqa: E402
from src.common.cache import DiskCache  # noqa: E402
from src.common.config import REPO_ROOT, load_collect_settings  # noqa: E402
from src.common.http import FetchError, HttpClient, RobotsDisallowed  # noqa: E402
from src.common.requestlog import DEFAULT_REQUEST_LOG, RequestLog  # noqa: E402

# Candidate domains (scheme defaults to https). Group is informational.
CANDIDATES = {
    "official": [
        "www.keralatourism.org", "www.tamilnadutourism.tn.gov.in", "www.karnatakatourism.org",
        "telanganatourism.gov.in", "tourism.telangana.gov.in", "aptourism.gov.in",
        "tourism.ap.gov.in", "tourism.py.gov.in", "http://pondytourism.in",
        "www.incredibleindia.gov.in", "tourism.gov.in", "whc.unesco.org",
        "periyartigerreserve.org", "asi.nic.in", "tirumala.org", "hrce.tn.gov.in",
    ],
    "reference_or_blog": ["sahapedia.org", "www.inditales.com", "wikitravel.org"],
    "commercial": [
        "www.tripoto.com", "www.holidify.com", "www.thrillophilia.com", "www.lonelyplanet.com",
        "www.tripadvisor.in", "www.thehindu.com", "www.nativeplanet.com",
    ],
}
# Terms pages that link discovery on the homepage does not find; tried by path when listed.
EXTRA_TERMS = {
    "www.keralatourism.org": [
        "/terms-and-conditions", "/terms-of-use", "/disclaimer", "/privacy-policy"],
    "www.inditales.com": ["/privacy-policy/", "/terms-and-conditions/", "/disclaimer/"],
    "www.holidify.com": ["/terms-of-service", "/terms", "/terms-and-conditions", "/privacy-policy"],
    "www.nativeplanet.com": ["/terms-of-use", "/terms-and-conditions", "/privacy-policy"],
    "periyartigerreserve.org": ["/terms-and-conditions", "/Disclaimer", "/privacy-policy"],
    "www.thehindu.com": ["/terms-of-use/", "/terms-and-conditions/", "/disclaimer/"],
    "whc.unesco.org": ["/en/syndication/"],
}
EXTRA_ABSOLUTE = {"whc.unesco.org": ["https://www.unesco.org/en/terms-use"]}
HOST_DELAY_S = 1.0
KEYWORDS = re.compile(
    r"scrap|crawl|robot|spider|automat|\bbots?\b|copy|reproduc|republish|redistribut|commercial|"
    r"prior (written )?(permission|consent)|licen[sc]e|creative commons|personal use|data min|"
    r"text and data|framing|hyperlink|attribution|acknowledg",
    re.I,
)
TERMS_LINK = re.compile(
    r"terms|condition|disclaimer|copyright|legal|website[- ]policy|hyperlink", re.I
)
SAMPLE_PATHS = ["/", "/sitemap.xml", "/search?q=munnar", "/destinations/", "/blog/"]


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def key_sentences(text: str, limit: int = 10) -> list[str]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if KEYWORDS.search(sent) and 25 < len(sent) < 450:
            out.append(sent.strip())
        if len(out) >= limit:
            break
    return out


def robots_groups(text: str) -> list[tuple[list[str], list[tuple[str, str]]]]:
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    last_was_agent = False
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            if not last_was_agent:
                groups.append(([], []))
            groups[-1][0].append(v)
            last_was_agent = True
        else:
            last_was_agent = False
            if groups:
                groups[-1][1].append((k, v))
    return groups


def analyse_robots(text: str, ua: str) -> dict:
    rp = RobotFileParser()
    rp.parse(text.splitlines())
    groups = robots_groups(text)
    star = [r for ags, r in groups if "*" in ags]
    star_disallow = [v for rules in star for k, v in rules if k == "disallow" and v]
    delay = next((v for rules in star for k, v in rules if k == "crawl-delay"), None)
    blocked = sorted({a for ags, rules in groups for a in ags
                      if ("disallow", "/") in rules and ("allow", "/") not in rules})
    return {
        "allowed_for_our_user_agent": {p: rp.can_fetch(ua, p) for p in SAMPLE_PATHS},
        "star_disallow_count": len(star_disallow),
        "star_disallow_sample": star_disallow[:6],
        "star_blanket_disallow": "/" in star_disallow,
        "crawl_delay_star": delay,
        "sitemaps": re.findall(r"(?im)^sitemap:\s*(\S+)", text)[:4],
        "fully_blocked_agents": blocked,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "manifests" / "source_evidence.json"))
    ap.add_argument("--domains", help="comma-separated domains to check instead of CANDIDATES "
                    "(group 'blog'); use with --out so the main evidence file is not overwritten")
    ap.add_argument("--historical", action="store_true",
                    help="allow this one-off script to run: it bypasses the D43 allowlist")
    args = ap.parse_args()
    refused = refuse_without_historical("gather_source_evidence.py", args.historical)
    if refused:
        return refused
    candidates = ({"blog": [d.strip() for d in args.domains.split(",") if d.strip()]}
                  if args.domains else CANDIDATES)
    cfg = load_collect_settings()
    http = HttpClient(DiskCache(cfg.cache_dir), cfg.user_agent, min_delay_s=HOST_DELAY_S,
                      max_retries=1, request_log=RequestLog(DEFAULT_REQUEST_LOG))
    result = {"retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "user_agent_product": cfg.user_agent.split(" ", 1)[0], "domains": {}}

    def read_terms(url: str, label: str) -> dict:
        try:
            t = http.fetch(url)
            txt = strip_html(t.text)
            return {"url": url, "label": label, "status": t.status, "chars": len(txt),
                    "key_sentences": key_sentences(txt)}
        except RobotsDisallowed:
            return {"url": url, "label": label, "status": "robots-disallowed"}
        except FetchError as exc:
            return {"url": url, "label": label, "status": f"error: {exc}"[:110]}

    for group, domains in candidates.items():
        for entry in domains:
            scheme = "http" if entry.startswith("http://") else "https"
            host = entry.replace("http://", "")
            rec: dict = {"group": group, "scheme": scheme}
            try:
                r = http.fetch(f"{scheme}://{host}/robots.txt", check_robots=False)
                rec["robots_status"] = r.status
                if r.status == 200 and "<html" not in r.text[:400].lower():
                    rec["robots"] = analyse_robots(r.text, cfg.user_agent)
                else:
                    rec["robots_note"] = "no robots.txt (404 or an HTML page): no rules"
            except FetchError as exc:
                rec["robots_status"] = f"error: {exc}"[:150]
            pages: list[tuple[str, str]] = []
            try:
                home = http.fetch(f"{scheme}://{host}/")
                rec["home_status"] = home.status
                rec["cc_mentions"] = sorted(set(re.findall(
                    r"creativecommons\.org/(?:licenses|publicdomain)/[\w./-]+", home.text)))[:5]
                for m in re.finditer(r'(?is)<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                                     home.text):
                    href, label = m.group(1), strip_html(m.group(2))
                    full = urljoin(f"{scheme}://{host}/", href)
                    if urlsplit(full).netloc.endswith(host.replace("www.", "")) and (
                        TERMS_LINK.search(label) or TERMS_LINK.search(href)
                    ) and full not in [p[0] for p in pages]:
                        pages.append((full, label[:60]))
                pages.sort(key=lambda x: 0 if re.search("terms", x[0] + x[1], re.I) else 1)
                pages = pages[:3]
            except RobotsDisallowed:
                rec["home_status"] = "robots-disallowed"
            except FetchError as exc:
                rec["home_status"] = f"error: {exc}"[:150]
            if not pages:  # nothing discovered: try the usual paths, remember what answered
                guesses = EXTRA_TERMS.get(host, [])
                pages = [(f"{scheme}://{host}{p}", "guessed path") for p in guesses]
            pages += [(u, "linked from UNESCO") for u in EXTRA_ABSOLUTE.get(host, [])]
            rec["terms_pages"] = [read_terms(u, lab) for u, lab in pages]
            # identical sizes under different guessed paths = a catch-all "page not found" shell
            sizes = [t["chars"] for t in rec["terms_pages"] if t.get("status") == 200]
            rec["soft_404_suspected"] = len(sizes) > 1 and len(set(sizes)) == 1
            result["domains"][entry] = rec
            print(entry, rec.get("robots_status"), rec.get("home_status"),
                  len(rec["terms_pages"]), file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}; network_requests={http.requests_made}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
