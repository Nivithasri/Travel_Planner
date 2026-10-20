"""Hand-label sample for aspect and is_warning (D58, D68 fix round). Offline: no API call.

    python scripts/label_sample.py            # writes data/enriched/gold_sample.csv (30 chunks)
    python scripts/label_sample.py --score    # reads the filled CSV, prints accuracy and tables

The CSV shows chunk text only. Fill `aspect_gold` (timing, closure, price, scenery, food, safety,
history, other) and `is_warning_gold` (true/false) yourself; nothing here writes a label.
The sample is 30 chunks, seeded, stratified by source type (reference, wikivoyage) and over the
17 D33-eligible places, drawn ONLY from model-written rows (provenance llm), never from blog, and
never from a chunk that was in any pilot or in the earlier gold sheet (gold_sample_old.csv, renamed
and left unlabelled). The model labels of these chunks are already in
the full-run checkpoint, so `--score` needs no API call. It never overwrites a sheet without
--force. `--score` compares the LLM labels and the rule-based labels with your gold labels and,
for any gold chunk that was in a pilot the prompt was read against, reports those rows apart
("seen"), because the prompt scores optimistically on them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.cache import read_jsonl  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402
from src.process.enrich import checkpoint_files, load_checkpoint_fields  # noqa: E402
from src.process.label import (  # noqa: E402
    LabelError,
    format_score,
    gold_rows,
    parse_gold,
    read_gold_csv,
    score,
    select_gold_unseen,
    write_gold_csv,
)

ENRICHED = REPO_ROOT / "data" / "enriched"
CSV_PATH = ENRICHED / "gold_sample.csv"
OLD_CSV = ENRICHED / "gold_sample_old.csv"
GOLD_SEED = 44
GOLD_N = 30


def prompt_work_ids() -> set[str]:
    """Chunks the prompt work was read against: the v1 pilot (200), the 60-chunk pilots, and the
    checkpoints of the v1 and v3 pilots. (The v4 pilot used the same 79 chunks as v3.)"""
    ids: set[str] = set()
    for p in [ENRICHED / "pilot_selection.json", *sorted(ENRICHED.glob("pilot60_*.json"))]:
        if p.exists():
            ids |= {o["chunk_id"] for o in json.loads(p.read_text(encoding="utf-8"))["order"]}
    for p in [ENRICHED / "checkpoint_enrich-v1.jsonl", *sorted(ENRICHED.glob(
            "checkpoint_enrich-v3_*.jsonl"))]:
        if p.exists() and not p.name.endswith(".superseded.jsonl"):
            ids |= {r["chunk_id"] for r in read_jsonl(p)}
    return ids


def old_gold_ids() -> set[str]:
    return {r["chunk_id"] for r in read_gold_csv(OLD_CSV)} if OLD_CSV.exists() else set()


def default_checkpoint() -> Path | None:
    found = checkpoint_files(ENRICHED)
    return found[0] if len(found) == 1 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--score", action="store_true", help="score the filled CSV")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--checkpoint", type=Path, help="default: the one checkpoint of the prompt")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    args = ap.parse_args(argv)

    if args.score:
        try:
            gold, blank = parse_gold(read_gold_csv(args.csv))
        except (LabelError, FileNotFoundError) as exc:
            print(f"cannot score: {exc}")
            return 2
        if not gold:
            print(f"no labelled rows in {args.csv} ({blank} blank): fill aspect_gold and "
                  "is_warning_gold first")
            return 2
        ckpt = args.checkpoint or default_checkpoint()
        if ckpt is None or not Path(ckpt).exists():
            print("no single checkpoint of the current prompt found: pass --checkpoint FILE")
            return 2
        llm = load_checkpoint_fields(Path(ckpt))
        print(f"{blank} blank rows skipped; LLM labels from {Path(ckpt).name} ({len(llm)} chunks)")
        print(format_score(score(gold, llm, seen=prompt_work_ids())))
        return 0

    chunks = [json.loads(line) for line in
              (REPO_ROOT / "data" / "processed" / "chunks.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    prov_path = ENRICHED / "provenance.jsonl"
    if not prov_path.exists():
        print("data/enriched/provenance.jsonl is missing: run `python scripts/enrich.py "
              "--assemble --eligible-only` first")
        return 2
    llm_ids = {r["chunk_id"] for r in read_jsonl(prov_path) if r["source"] == "llm"}
    excluded = set(json.loads((REPO_ROOT / "data" / "manifests" / "d33_excluded.json").read_text(
        encoding="utf-8"))["excluded"])
    cand = {c["chunk_id"] for c in chunks
            if c["chunk_id"] in llm_ids and c["source_type"] != "blog"
            and c["place"] not in excluded}
    banned = prompt_work_ids() | old_gold_ids()
    ids = select_gold_unseen(chunks, cand, banned, GOLD_SEED, GOLD_N)
    if len(ids) < GOLD_N:
        print(f"only {len(ids)} eligible unseen chunks found")
        return 2
    try:
        write_gold_csv(args.csv, gold_rows(chunks, ids), force=args.force)
    except FileExistsError as exc:
        print(exc)
        return 2
    by_id = {c["chunk_id"]: c for c in chunks}
    by_type: dict[str, int] = {}
    by_place: dict[str, int] = {}
    for i in ids:
        c = by_id[i]
        by_type[c["source_type"]] = by_type.get(c["source_type"], 0) + 1
        by_place[c["place"]] = by_place.get(c["place"], 0) + 1
    print(f"wrote {args.csv.relative_to(REPO_ROOT)}: {len(ids)} chunks; by source type {by_type}; "
          f"{len(by_place)} places {dict(sorted(by_place.items()))}; "
          f"overlap with pilots/earlier gold: {len(set(ids) & banned)}; blog: 0; "
          f"all model-written: {set(ids) <= llm_ids}")
    print("Fill aspect_gold and is_warning_gold, then: python scripts/label_sample.py --score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
