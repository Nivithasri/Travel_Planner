"""Hand-label sample for aspect and is_warning (D58). Offline: no LLM call, no network.

    python scripts/label_sample.py            # writes data/enriched/gold_sample.csv (30 chunks)
    python scripts/label_sample.py --score    # reads the filled CSV, prints accuracy and tables

The CSV shows chunk text only. Fill `aspect_gold` (timing, closure, price, scenery, food, safety,
history, other) and `is_warning_gold` (true/false) yourself; nothing here writes a label. The
sample is seeded, 10 chunks per source type including 6 under-200-token tails, and excludes the
200 pilot chunks the prompt was tuned on. It never overwrites an existing sheet without --force.
`--score` compares the LLM labels (checkpoint of the current prompt version) and the rule-based
fallback labels with your gold labels; chunks the LLM has not labelled yet are left out of the
LLM score and counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schemas import REPO_ROOT, load_tokenizer_dir  # noqa: E402
from src.process.enrich import PROMPT_VERSION, load_checkpoint_fields  # noqa: E402
from src.process.label import (  # noqa: E402
    LabelError,
    format_score,
    gold_rows,
    parse_gold,
    read_gold_csv,
    score,
    select_gold,
    write_gold_csv,
)
from src.process.process import load_counter  # noqa: E402

ENRICHED = REPO_ROOT / "data" / "enriched"
CSV_PATH = ENRICHED / "gold_sample.csv"
GOLD_SEED = 43
SHORT_TOKENS = 200


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--score", action="store_true", help="score the filled CSV")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--checkpoint", type=Path,
                    default=ENRICHED / f"checkpoint_{PROMPT_VERSION}.jsonl")
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
        llm = load_checkpoint_fields(args.checkpoint) if args.checkpoint.exists() else {}
        print(f"{blank} blank rows skipped; LLM labels from {args.checkpoint.name} "
              f"({len(llm)} chunks)")
        print(format_score(score(gold, llm)))
        return 0

    chunks = [json.loads(line) for line in
              (REPO_ROOT / "data" / "processed" / "chunks.jsonl").read_text(
                  encoding="utf-8").splitlines()]
    sel = ENRICHED / "pilot_selection.json"
    if not sel.exists():
        print("data/enriched/pilot_selection.json is missing: run scripts/enrich.py --report "
              "--pilot once first")
        return 2
    pilot = [o["chunk_id"] for o in json.loads(sel.read_text(encoding="utf-8"))["order"]]
    tok = load_tokenizer_dir()
    if tok is None:
        print("TOKENIZER_DIR is not set in .env: needed to find the under-200-token tails")
        return 2
    count, _ = load_counter(tok)
    shorts = [c["chunk_id"] for c in chunks if count(c["text"]) < SHORT_TOKENS]
    ids = select_gold(chunks, pilot, shorts, GOLD_SEED)
    try:
        write_gold_csv(args.csv, gold_rows(chunks, ids), force=args.force)
    except FileExistsError as exc:
        print(exc)
        return 2
    by_type: dict[str, int] = {}
    by_id = {c["chunk_id"]: c for c in chunks}
    for i in ids:
        by_type[by_id[i]["source_type"]] = by_type.get(by_id[i]["source_type"], 0) + 1
    print(f"wrote {args.csv.relative_to(REPO_ROOT)}: {len(ids)} chunks {by_type}, "
          f"{len(set(ids) & set(shorts))} under-200-token tails, 0 from the pilot")
    print("Fill aspect_gold and is_warning_gold, then: python scripts/label_sample.py --score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
