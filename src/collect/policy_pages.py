"""Fetch a FIXED list of policy pages (config/sources.json -> policy_fetch) and pull out the
sentences that matter for a decision. Used for the YouTube developer policies (D29).

Narrow by design: only the listed hosts and the listed pages are ever requested, one request per
page, through HttpClient (robots-checked, logged, cached). Nothing here goes into the corpus.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from src.common.config import REPO_ROOT
from src.common.http import FetchError, HttpClient

CONFIG_PATH = REPO_ROOT / "config" / "sources.json"

TOPICS = {
    "storage_retention": r"\b(stor(e|ed|age|ing)|refresh|calendar days|retain|retention|"
                         r"cache|delete)\b",
    "corpus_dataset": r"\b(backup|archive|dataset|machine learning|train|derivative|"
                      r"database|copy|copies)\b",
    "automated_scraping": r"\b(scrap\w*|automat\w*|crawl\w*|robots?|bot|"
                          r"other than through|circumvent)\b",
    "consequences": r"\b(terminat\w*|suspend\w*|revok\w*|disabl\w*|audit\w*|restrict\w*|"
                    r"api key|project|account)\b",
    "quota": r"\b(quota|units?|videos\.list|commentthreads\.list|search\.list)\b",
}


def load_policy_config(path: Path = CONFIG_PATH) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    cfg = doc["policy_fetch"]
    hosts = {h.lower() for h in cfg["hosts"]}
    for url in cfg["pages"]:
        if urlsplit(url).netloc.lower() not in hosts:
            raise ValueError(f"policy page outside the policy_fetch hosts: {url}")
    return cfg


def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 25 < len(s.strip()) < 600]


def by_topic(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {t: [] for t in TOPICS}
    for s in sentences(text):
        for topic, pat in TOPICS.items():
            if re.search(pat, s, re.IGNORECASE):
                out[topic].append(s)
    return out


def fetch_pages(http: HttpClient, cfg: dict) -> dict[str, dict]:
    """url -> {status, chars, text or error}. One request per page; failures are recorded."""
    out: dict[str, dict] = {}
    for url in cfg["pages"]:
        try:
            res = http.fetch(url)
            text = strip_html(res.text)
            out[url] = {"status": res.status, "chars": len(text), "text": text,
                        "from_cache": res.from_cache}
        except FetchError as exc:
            out[url] = {"status": exc.status, "chars": 0, "text": "", "error": str(exc)[:200]}
    return out
