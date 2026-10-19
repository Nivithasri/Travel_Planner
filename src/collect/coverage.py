"""Coverage of the core snapshot places by source type (D22, D23).

A place passes when it has raw data from at least two distinct chunk source types AND at least one
of them is not wikivoyage or reference. Hold-out places are listed but never counted. A collector
that was skipped (no credentials, not allowed) is reported with its reason; it never counts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.collect.raw import RawStore
from src.collect.types import SOURCE_TYPE
from src.common.schemas import PlaceRegistryEntry
from src.places.registry import slugify

REFERENCE_FAMILY = {"wikivoyage", "reference"}


@dataclass
class PlaceCoverage:
    place: str
    state: str
    holdout: bool
    records_by_raw_source: dict[str, int] = field(default_factory=dict)
    types: list[str] = field(default_factory=list)
    passes: bool = False
    reason: str = ""


def passes_rule(types: set[str]) -> bool:
    return len(types) >= 2 and bool(types - REFERENCE_FAMILY)


def place_coverage(entry: PlaceRegistryEntry, raw_root: Path) -> PlaceCoverage:
    store = RawStore(raw_root)
    by_source: dict[str, int] = {}
    for raw_source in SOURCE_TYPE:
        n = len(store.read(raw_source, entry.name))
        if n:
            by_source[raw_source] = n
    types = {SOURCE_TYPE[s] for s in by_source}
    cov = PlaceCoverage(entry.name, entry.state.value, bool(entry.holdout), by_source,
                        sorted(types))
    if cov.holdout:
        cov.reason = "hold-out: not collected by design"
        return cov
    cov.passes = passes_rule(types)
    if not cov.passes:
        if not types:
            cov.reason = "nothing collected yet"
        elif len(types) < 2:
            cov.reason = f"only {len(types)} source type(s)"
        else:
            cov.reason = "only reference-wiki sources (wikivoyage/reference)"
    return cov


def skipped_sources(status_root: Path, place: str) -> dict[str, str]:
    """source -> reason, from the per-place status file written by ensure_place."""
    try:
        doc = json.loads((status_root / "places" / f"{slugify(place)}.json")
                         .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for src, rec in doc.get("sources", {}).items():
        if rec.get("skipped"):
            out[src] = f"skipped: {rec['skipped']}"
        elif rec.get("error"):
            out[src] = f"error: {rec['error']}"
        elif rec.get("notes"):
            out[src] = "notes: " + " | ".join(rec["notes"])
    return out


def render(entries: list[PlaceRegistryEntry], raw_root: Path, status_root: Path,
           skipped_everywhere: dict[str, str] | None = None) -> tuple[str, dict]:
    rows = [place_coverage(e, raw_root) for e in entries]
    counted = [r for r in rows if not r.holdout]
    n_pass = sum(r.passes for r in counted)
    totals: dict[str, int] = {}
    for r in rows:
        for s, n in r.records_by_raw_source.items():
            totals[s] = totals.get(s, 0) + n
    lines = ["# Coverage report", "",
             f"Rule (D23): >= 2 source types and >= 1 that is not wikivoyage/reference. "
             f"**{n_pass} of {len(counted)}** counted places pass "
             f"({len(rows) - len(counted)} hold-out(s) excluded).", "",
             "## Records per raw source", "", "| source | records |", "|---|---:|"]
    lines += [f"| {s} | {totals.get(s, 0)} |" for s in SOURCE_TYPE]
    notes: dict[str, str] = dict(skipped_everywhere or {})
    for r in rows:
        for s, why in skipped_sources(status_root, r.place).items():
            notes.setdefault(s, why)
    if notes:
        lines += ["", "## Collectors skipped or failed", ""]
        lines += [f"- **{s}**: {why}" for s, why in sorted(notes.items())]
    lines += ["", "## Per place", "", "| place | state | types | pass | note |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.place} | {r.state} | {', '.join(r.types) or '-'} | "
                     f"{'hold-out' if r.holdout else ('yes' if r.passes else 'NO')} | {r.reason} |")
    data = {"passing": n_pass, "counted": len(counted), "records_by_source": totals,
            "notes": notes, "places": [r.__dict__ for r in rows]}
    return "\n".join(lines) + "\n", data


def raw_stats(raw_root: Path) -> dict[str, dict]:
    """Per raw source: records, bytes on disk (records.jsonl), places with data, body characters."""
    out: dict[str, dict] = {}
    for src in SOURCE_TYPE:
        files = list((raw_root / src).glob("*/records.jsonl")) if (raw_root / src).exists() else []
        n = chars = 0
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    n += 1
                    chars += len(json.loads(line).get("body", ""))
        out[src] = {"records": n, "bytes": sum(f.stat().st_size for f in files),
                    "places": len(files), "body_chars": chars}
    return out


def transcript_outcomes(cache_dir: Path, status_root: Path) -> dict[str, int]:
    """ok / none / disabled from the transcript cache; blocked from the per-place status files
    (a block is never cached, it aborts the run)."""
    out = {"ok": 0, "none": 0, "disabled": 0, "blocked": 0}
    for f in (cache_dir / "youtube_transcript").glob("*/*.json") if (
            cache_dir / "youtube_transcript").exists() else []:
        try:
            val = json.loads(f.read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError):
            continue
        if val.get("status") == "ok":
            out["ok"] += 1
        elif val.get("reason") == "TranscriptsDisabled":
            out["disabled"] += 1
        else:
            out["none"] += 1
    for f in (status_root / "places").glob("*.json") if (status_root / "places").exists() else []:
        err = (json.loads(f.read_text(encoding="utf-8")).get("sources", {})
               .get("youtube", {}).get("error") or "")
        out["blocked"] += "blocked" in err
    return out
