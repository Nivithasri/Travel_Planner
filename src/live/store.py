"""Same-day cache and credit ledger of the live layer (D100).

The cache keeps ONLY the cleaned minimal fields of a result (url, title, snippet, published date),
never the raw API body, for the current UTC day only: files of any other day are deleted when the
cache is opened, and with cache_enabled false nothing is read or written. The ledger counts credits
per calendar month and refuses to spend past the configured stop (never pay)."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from src.common.cache import atomic_write_text

MINIMAL_FIELDS = ("url", "title", "content", "published_date")


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def minimal(result: dict, max_chars: int) -> dict:
    out = {k: result.get(k) for k in MINIMAL_FIELDS}
    out["content"] = str(out.get("content") or "")[:max_chars]
    out["title"] = str(out.get("title") or "")[:200]
    out["url"] = str(out.get("url") or "")
    out["published_date"] = out.get("published_date") or None
    return out


class DayCache:
    def __init__(self, directory: Path, enabled: bool,
                 today: Callable[[], str] = _utc_today):
        self.dir, self.enabled, self._today = Path(directory), enabled, today
        self.purged = 0
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.purged = self.purge()

    def purge(self) -> int:
        """Delete every cache file that is not from today."""
        n = 0
        for f in self.dir.glob("*.json"):
            if not f.name.startswith(self._today() + "_"):
                f.unlink()
                n += 1
        return n

    def _path(self, query: str) -> Path:
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        return self.dir / f"{self._today()}_{h}.json"

    def get(self, query: str) -> list[dict] | None:
        if not self.enabled:
            return None
        p = self._path(query)
        if not p.exists():
            return None
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return doc["results"] if doc.get("day") == self._today() else None

    def put(self, query: str, results: list[dict]) -> None:
        if not self.enabled:
            return
        atomic_write_text(self._path(query), json.dumps(
            {"day": self._today(), "query": query, "results": results}, sort_keys=True))


class Ledger:
    def __init__(self, path: Path, monthly_stop_at: int, monthly_credits: int,
                 today: Callable[[], str] = _utc_today):
        self.path, self.stop_at, self.limit, self._today = (
            Path(path), monthly_stop_at, monthly_credits, today)
        self.month = self._today()[:7]
        self.credits = 0
        self.searches = 0
        if self.path.exists():
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
                if doc.get("month") == self.month:
                    self.credits, self.searches = int(doc["credits"]), int(doc["searches"])
            except (ValueError, KeyError):
                pass  # an unreadable ledger starts the month at zero (credits are re-checked)

    def can_spend(self, credits: int) -> bool:
        return self.credits + credits <= self.stop_at

    def spend(self, credits: int) -> None:
        if not self.can_spend(credits):
            raise RuntimeError("the monthly live-search credit stop would be exceeded")
        self.credits += credits
        self.searches += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(
            {"month": self.month, "credits": self.credits, "searches": self.searches,
             "stop_at": self.stop_at, "limit": self.limit}, sort_keys=True))
