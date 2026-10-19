"""Raw JSON envelopes for facts (D37): one file per request under data/raw/<source>/<place>/."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.collect.raw import iso_utc
from src.common.cache import append_jsonl, atomic_write_text, read_jsonl
from src.common.http import FetchResult, full_url
from src.places.registry import slugify


def envelope_path(raw_root: Path, source: str, place_dir: str, filename: str) -> Path:
    return Path(raw_root) / source / place_dir / filename


def make_envelope(source: str, place: str, url: str, params: dict | None, res: FetchResult,
                  licence: str, attribution: str, source_revision: dict | None = None) -> dict:
    return {
        "source": source, "place": place, "request_url": full_url(url, params),
        "params": params or {}, "retrieved_at": iso_utc(res.retrieved_at), "status": res.status,
        "from_cache": res.from_cache, "licence": licence, "attribution": attribution,
        "source_revision": source_revision, "response": res.json(),
    }


def manifest_line(env: dict, path: Path) -> dict:
    """One raw-manifest line for an envelope: url, source, place, sha256 of the file, size.
    No content."""
    data = Path(path).read_bytes()
    return {"url": env["request_url"], "source": env["source"], "place": env["place"],
            "retrieved_at": env["retrieved_at"], "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "kind": "facts_envelope"}


def write_envelope(path: Path, env: dict, manifest: Path | None = None) -> bool:
    """Write the envelope; an existing file is kept when this answer came from the cache.
    A new write adds a line to the raw manifest when one is given."""
    if path.exists() and env["from_cache"]:
        return False
    atomic_write_text(path, json.dumps(env, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    if manifest is not None:
        append_jsonl(manifest, [manifest_line(env, path)])
    return True


def backfill_manifest(raw_root: Path, source: str, manifest: Path) -> int:
    """Add manifest lines for envelopes already on disk (no network). Idempotent: a (url, sha256)
    pair already in the manifest is not added twice. Returns the number of lines added."""
    seen = set()
    if Path(manifest).exists():
        seen = {(r.get("url"), r.get("sha256")) for r in read_jsonl(manifest)}
    added = 0
    for path in sorted((Path(raw_root) / source).glob("*/*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        line = manifest_line(env, path)
        if (line["url"], line["sha256"]) in seen:
            continue
        append_jsonl(manifest, [line])
        seen.add((line["url"], line["sha256"]))
        added += 1
    return added


def place_dir(name: str) -> str:
    return slugify(name)
