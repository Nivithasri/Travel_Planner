"""Query cache of the Phase 8 page (D111).

The key is sha256 of the canonical JSON of (query, overrides, use_live). A corpus-only answer
(use_live False) is stored as JSON under .cache/app/ and in session memory. A live-on answer is
held in session memory ONLY: nothing derived from live results is ever written to disk
(D109), and a live-on request is never served from a disk entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "app"
DISK_LIVE_STATUSES = ("disabled", "not_run")


def cache_key(query: str, overrides: dict, use_live: bool) -> str:
    payload = {"query": query, "overrides": overrides, "use_live": bool(use_live)}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def disk_allowed(use_live: bool, view: dict) -> bool:
    """Only a pure corpus-only answer may touch the disk (D109, D111)."""
    if use_live:
        return False
    if view.get("live_status") not in DISK_LIVE_STATUSES:
        return False
    return not any(s.get("live") for s in (view.get("final") or {}).get("sources", []))


class QueryCache:
    def __init__(self, directory: Path | str | None = None, memory: dict | None = None):
        self.directory = Path(directory) if directory else CACHE_DIR
        self.memory = memory if memory is not None else {}

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, query: str, overrides: dict, use_live: bool) -> dict | None:
        key = cache_key(query, overrides, use_live)
        if not use_live:
            path = self._path(key)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return self.memory.get(key)

    def put(self, query: str, overrides: dict, use_live: bool, view: dict) -> Path | None:
        """Keep the view in session memory; write it to disk only when disk_allowed()."""
        key = cache_key(query, overrides, use_live)
        entry = {**view, "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
        self.memory[key] = entry
        if not disk_allowed(use_live, entry):
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def clear(self) -> int:
        """Remove this cache's own JSON files and the session memory; returns the file count."""
        self.memory.clear()
        n = 0
        if self.directory.is_dir():
            for p in self.directory.glob("*.json"):
                p.unlink(missing_ok=True)
                n += 1
        return n
