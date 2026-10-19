"""Per-request log: one JSON line per request or redirect hop, no content, no query strings.

Fields: ts (UTC), run_id, host, path, kind, status, from_cache, attempt (1-based; 0 for a
cache hit), source (the collector in progress, when known), note (e.g. "redirect-refused").
Written through `append_jsonl`, so every line is flushed and a crash loses at most the last one.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from src.common.cache import append_jsonl
from src.common.config import REPO_ROOT

DEFAULT_REQUEST_LOG = REPO_ROOT / "data" / "manifests" / "request_log.jsonl"


def classify_url(url: str) -> str:
    p = urlsplit(url)
    host, path = p.netloc.lower(), p.path.lower()
    if path.endswith("robots.txt"):
        return "robots"
    if "sitemap" in path:
        return "sitemap"
    if host.endswith("googleapis.com") or path.endswith("api.php") or path.endswith("/sparql"):
        return "api"
    if host.endswith("youtube.com") and (path.startswith("/youtubei/") or "/api/" in path):
        return "transcript"
    return "page"


class RequestLog:
    def __init__(self, path: str | Path, run_id: str | None = None,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.clock = clock
        self.source: str | None = None  # set by the caller while a collector runs

    def write(self, url: str, status: int | None, from_cache: bool, attempt: int,
              note: str | None = None, kind: str | None = None) -> None:
        p = urlsplit(url)
        rec = {
            "ts": datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "run_id": self.run_id, "host": p.netloc.lower(), "path": p.path,
            "kind": kind or classify_url(url), "status": status, "from_cache": from_cache,
            "attempt": attempt, "source": self.source, "note": note,
        }
        append_jsonl(self.path, [rec])
