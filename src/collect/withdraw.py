"""Withdraw a source from the snapshot (used for the YouTube transcripts, D27).

Deletes the raw records and the cache entries of one source, appends a "withdrawn" line per record
to the manifest (existing manifest lines are never changed or deleted), marks the per-place status
as skipped, and then proves nothing is left: distinctive phrases taken from the records AT RUNTIME
(kept in memory only, never written to a file or passed on a command line) are searched for in the
whole working tree and in git history. Only counts are reported.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.common.cache import append_jsonl, atomic_write_text
from src.places.registry import slugify

_WINDOW = re.compile(r"(?<= )[A-Za-z0-9 ,.'-]{40}")  # plain characters: appear verbatim in JSON
PHRASE_LEN = 40


def pick_phrase(text: str, start_ratio: float = 0.5) -> str | None:
    """A 40-character phrase made of plain characters, taken from the given point of the text."""
    for ratio in (start_ratio, 0.25, 0.75, 0.1):
        m = _WINDOW.search(text, int(len(text) * ratio))
        if m:
            return m.group(0)
    m = _WINDOW.search(" " + text)
    return m.group(0) if m else None


def pick_start_phrase(text: str) -> str | None:
    """A phrase from the very beginning (what earlier reports quoted), plain characters only."""
    return pick_phrase(text, 0.0)


@dataclass
class Plan:
    raw_files: list[Path] = field(default_factory=list)
    cache_files: list[Path] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def plan_withdrawal(raw_root: Path, cache_dir: Path, source: str, cache_kind: str) -> Plan:
    plan = Plan()
    plan.raw_files = sorted((raw_root / source).glob("*/records.jsonl"))
    for f in plan.raw_files:
        plan.records += [json.loads(x) for x in f.read_text("utf-8").splitlines() if x.strip()]
    plan.cache_files = sorted((cache_dir / cache_kind).glob("*/*.json"))
    return plan


def phrases_from(plan: Plan, cache_kind_texts: bool = True) -> list[str]:
    """Phrases (middle and start of every record body), in memory only."""
    out = []
    for r in plan.records:
        for pick in (pick_phrase, pick_start_phrase):
            p = pick(r["body"])
            if p:
                out.append(p)
    return out


def count_dir_files(root: Path) -> dict[str, int]:
    """File counts per immediate sub-directory (cache kinds) and in total."""
    out: dict[str, int] = {}
    if not root.exists():
        return {"TOTAL": 0}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out[d.name] = sum(1 for f in d.rglob("*") if f.is_file())
    out["TOTAL"] = sum(out.values())
    return out


def withdraw(plan: Plan, raw_root: Path, status_root: Path, manifest: Path, source: str,
             reason: str, now: datetime | None = None) -> dict:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [{"event": "withdrawn", "source": r["source"], "place": r["place"], "url": r["url"],
              "sha256": r["content_sha256"], "withdrawn_at": stamp, "reason": reason}
             for r in plan.records]
    if lines:
        append_jsonl(manifest, lines)
    places = sorted({r["place"] for r in plan.records})
    for f in plan.raw_files:
        f.unlink()
        try:
            f.parent.rmdir()
        except OSError:
            pass
    for f in plan.cache_files:
        f.unlink()
    try:
        (raw_root / source).rmdir()
    except OSError:
        pass
    for place in places:
        path = status_root / "places" / f"{slugify(place)}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text("utf-8"))
        doc.setdefault("sources", {})[source] = {
            "skipped": f"withdrawn ({reason})", "n_raw": 0, "n_requests": 0, "error": None,
            "ingested_at": None, "expires_at": None, "complete": True, "withdrawn_at": stamp}
        text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True)
        atomic_write_text(path, text + "\n")
    return {"records": len(plan.records), "manifest_lines": len(lines)}


def _count(blob: bytes, needles: list[bytes]) -> int:
    return sum(blob.count(n) for n in needles)


def scan(repo: Path, phrases: list[str]) -> dict:
    """Counts of phrase matches: working tree (all but .env and .git), tracked files, raw .git,
    git grep over every revision (patterns on stdin), and `git log -p --all`."""
    needles = [p.encode("utf-8") for p in phrases]
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True)
                  .stdout.decode("utf-8", "replace").split("\0"))
    tree = tracked_hits = git_raw = 0
    hit_files: list[str] = []
    for f in repo.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(repo).as_posix()
        if rel.startswith(".git/"):
            try:
                git_raw += _count(f.read_bytes(), needles)
            except OSError:
                pass
            continue
        if rel == ".env":
            continue
        try:
            n = _count(f.read_bytes(), needles)
        except OSError:
            continue
        if n:
            tree += n
            hit_files.append(rel)
            if rel in tracked:
                tracked_hits += n
    patterns = b"\n".join(needles) + b"\n"
    listing = subprocess.run(["git", "rev-list", "--all"], cwd=repo, capture_output=True)
    revs = listing.stdout.split()
    grep_hits = 0
    for rev in revs:
        out = subprocess.run(["git", "grep", "-I", "-c", "-F", "-f", "-", rev.decode()],
                             cwd=repo, input=patterns, capture_output=True).stdout
        grep_hits += sum(int(line.rsplit(b":", 1)[1]) for line in out.splitlines() if line)
    log = subprocess.run(["git", "log", "-p", "--all"], cwd=repo, capture_output=True).stdout
    return {"phrases": len(needles), "working_tree": tree, "tracked_files": tracked_hits,
            "git_raw_files": git_raw, "git_grep_revisions": grep_hits, "revisions": len(revs),
            "git_log_p": _count(log, needles), "files_with_matches": hit_files}


# ---- host-filtered cache entries (API data under a retention rule, D30) ----------------------
def plan_cache_host(cache_dir: Path, host: str, kind: str = "http") -> list[Path]:
    """Cache entries of `kind` whose stored model (the host) equals `host`."""
    out = []
    for f in sorted((cache_dir / kind).glob("*/*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("model") == host:
            out.append(f)
    return out


def retrieved_range(files: list[Path]) -> tuple[str, str] | None:
    """(oldest, newest) retrieved_at of cache entries, as UTC timestamps."""
    stamps = []
    for f in files:
        try:
            v = json.loads(f.read_text(encoding="utf-8"))["value"]["retrieved_at"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        stamps.append(float(v))
    if not stamps:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.fromtimestamp(min(stamps), tz=timezone.utc).strftime(fmt),
            datetime.fromtimestamp(max(stamps), tz=timezone.utc).strftime(fmt))


def search_metadata_phrases(files: list[Path], min_len: int = 8) -> list[str]:
    """Video titles and channel names found in YouTube search responses, IN MEMORY ONLY, with the
    JSON-escaped spellings too (files escape quotes and non-ASCII differently)."""
    out: set[str] = set()
    for f in files:
        try:
            body = json.loads(json.loads(f.read_text(encoding="utf-8"))["value"]["text"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for item in body.get("items", []) if isinstance(body, dict) else []:
            sn = item.get("snippet") or {}
            for field_ in ("title", "channelTitle"):
                v = sn.get(field_)
                if isinstance(v, str) and len(v) >= min_len:
                    out |= phrase_variants(v)
    return sorted(out)


def phrase_variants(text: str) -> set[str]:
    """The text as is, and as it appears inside JSON (escaped and ASCII-escaped)."""
    variants = {text, json.dumps(text, ensure_ascii=False)[1:-1], json.dumps(text)[1:-1]}
    return {v for v in variants if v}


def delete_files(files: list[Path]) -> int:
    n = 0
    for f in files:
        f.unlink()
        n += 1
    return n


def area_of(rel: str) -> str:
    """Coarse area of a repo-relative path, for count-only reporting."""
    areas = (("data/manifests/request_log", "request_log"), ("data/manifests/", "manifests"),
             ("data/raw/", "data/raw"), ("data/processed/", "data/processed"),
             (".cache/", ".cache"), ("docs/", "docs"), ("src/", "src"), ("tests/", "tests"),
             ("scripts/", "scripts"), ("config/", "config"))
    for prefix, name in areas:
        if rel.startswith(prefix):
            return name
    return "LOG.md" if rel == "LOG.md" else "other"
