"""Disk cache keyed by sha256(kind, model/endpoint, canonical JSON of inputs).

Layout: <root>/<kind>/<key[:2]>/<key>.json, one file per entry, so the cache is resumable and needs
no lock. Writes are atomic (temp file in the same directory, then os.replace).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

_MISS = object()

# Windows: os.replace raises PermissionError while another process holds the file
_REPLACE_ATTEMPTS = 8
_REPLACE_DELAY_S = 0.05


def _replace_with_retry(src: str, dst: Path, sleep=time.sleep) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            sleep(_REPLACE_DELAY_S * (attempt + 1))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(kind: str, model: str, inputs: Any) -> str:
    payload = canonical_json({"kind": kind, "model": model, "inputs": inputs})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class DiskCache:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def path_for(self, kind: str, model: str, inputs: Any) -> Path:
        key = cache_key(kind, model, inputs)
        return self.root / kind / key[:2] / f"{key}.json"

    def _read(self, path: Path) -> Any:
        """Return the stored value, or _MISS for anything unreadable (absent, unreadable, bad
        UTF-8, invalid JSON, wrong shape). A bad entry is recomputed and overwritten."""
        try:
            with open(path, encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):  # ValueError covers JSONDecodeError and UnicodeDecodeError
            return _MISS
        if not isinstance(entry, dict) or "value" not in entry:
            return _MISS
        return entry["value"]


    def get(self, kind: str, model: str, inputs: Any, default: Any = None) -> Any:
        value = self._read(self.path_for(kind, model, inputs))
        return default if value is _MISS else value

    def set(self, kind: str, model: str, inputs: Any, value: Any) -> None:
        entry = {"kind": kind, "model": model, "inputs": inputs, "value": value}
        atomic_write_text(self.path_for(kind, model, inputs), canonical_json(entry))

    def get_or_compute(
        self, kind: str, model: str, inputs: Any, fn: Callable[[], Any]
    ) -> Any:
        """Return the cached value, else call fn() and store its result. A hit never calls fn.

        A cached None is a valid hit. If fn raises, nothing is stored.
        """
        path = self.path_for(kind, model, inputs)
        value = self._read(path)
        if value is not _MISS:
            return value
        value = fn()
        entry = {"kind": kind, "model": model, "inputs": inputs, "value": value}
        atomic_write_text(path, canonical_json(entry))
        return value


def _repair_jsonl_tail(p: Path) -> None:
    """Make the file end on a record boundary before appending.

    If the last line has no trailing newline (a crash mid-write): when it is a complete JSON
    value, keep it and add the newline; otherwise it is a partial record, so cut it off.
    """
    if not p.exists() or p.stat().st_size == 0:
        return
    data = p.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1  # start of the unterminated last line (0 if there is only one)
    try:
        json.loads(data[cut:].decode("utf-8"))
    except ValueError:  # JSONDecodeError and UnicodeDecodeError
        with open(p, "r+b") as fh:
            fh.truncate(cut)
    else:
        with open(p, "ab") as fh:
            fh.write(b"\n")


def append_jsonl(path: str | os.PathLike, records: Iterable[Any]) -> int:
    """Append records as JSON lines; flushed and fsynced so a crash loses at most the tail.

    A partial last line left by an earlier crash is repaired first, so the new records never
    get glued onto it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _repair_jsonl_tail(p)
    n = 0
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    return n


def read_jsonl(path: str | os.PathLike) -> Iterator[Any]:
    """Yield records. Only an unterminated, unparseable final line (a crash mid-write) is
    skipped. Any other bad line raises: a complete line that is corrupt is not a resumable state,
    and a valid record is never dropped silently."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")
    unterminated_tail = not text.endswith("\n")
    while lines and lines[-1] == "":
        lines.pop()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if unterminated_tail and i == len(lines) - 1:
                return
            raise
