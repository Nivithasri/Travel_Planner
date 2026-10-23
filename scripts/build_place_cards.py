"""Build the place cards (Phase 7, D102): offline, deterministic, no LLM, no network.

    python scripts/build_place_cards.py [--include-blog]

Reads data/enriched/chunks.jsonl and provenance.jsonl, the committed place registry,
index/season_data.json and data/manifests/d33_excluded.json (all read-only) and writes
index/place_cards.json (gitignored: excerpts of CC BY-SA text, O12) and
data/manifests/place_cards_manifest.json (hashes and counts, no text, committed). Blog chunks
(no terms) are excluded unless --include-blog (D102, D103).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.facts.facts import HoldoutFilter  # noqa: E402
from src.retrieval.filters import load_provenance  # noqa: E402
from src.retrieval.query_understanding import load_core_registry  # noqa: E402
from src.tools.place_cards import CARDS_PATH, build_cards  # noqa: E402

ENRICHED = ROOT / "data" / "enriched" / "chunks.jsonl"
PROVENANCE = ROOT / "data" / "enriched" / "provenance.jsonl"
SEASON = ROOT / "index" / "season_data.json"
EXCLUDED = ROOT / "data" / "manifests" / "d33_excluded.json"
REGISTRY = ROOT / "data" / "registry" / "core_places.json"
MANIFEST = ROOT / "data" / "manifests" / "place_cards_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(path: Path = ENRICHED) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summary(cards: dict[str, dict]) -> dict[str, dict]:
    return {name: {"evidence_class": c["evidence_class"], "visibility_tercile":
                   c["visibility_tercile"], "tips": len(c["tips"]),
                   "warnings": len(c["warnings"]), "highlights": len(c["highlights"]),
                   "chunks_total": c["counts"]["chunks_total"],
                   "blog_excluded": c["counts"]["blog_excluded"],
                   "model_written_warnings": c["counts"]["model_written_warnings"],
                   "rule_based_warning_flags": c["counts"]["rule_based_warning_flags"]}
            for name, c in cards.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--include-blog", action="store_true",
                    help="also use blog chunks (inditales.com has no terms: owner decision, D103)")
    ap.add_argument("--out", type=Path, default=CARDS_PATH)
    args = ap.parse_args(argv)

    registry = load_core_registry()
    season = json.loads(SEASON.read_text(encoding="utf-8"))
    excluded = set(json.loads(EXCLUDED.read_text(encoding="utf-8"))["excluded"])
    cards = build_cards(load_chunks(), load_provenance(PROVENANCE), registry.entries, season,
                        excluded, holdout=HoldoutFilter.from_registry(registry),
                        include_blog=args.include_blog)
    meta = {"version": "cards-v1", "include_blog": args.include_blog, "n_places": len(cards),
            "inputs_sha256": {p.name: sha256(p) for p in
                              (ENRICHED, PROVENANCE, SEASON, EXCLUDED, REGISTRY)}}
    doc = {"meta": meta, "places": cards}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    MANIFEST.write_text(json.dumps(
        {**meta, "place_cards_sha256": sha256(args.out), "places": summary(cards)},
        indent=1, sort_keys=True), encoding="utf-8")
    full = [n for n, c in cards.items() if c["evidence_class"] == "full"]
    print(f"{len(cards)} cards ({len(full)} full, {len(cards) - len(full)} limited) -> {args.out}")
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
