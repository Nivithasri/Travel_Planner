"""Phase 3 batch: enrich the frozen chunks (context note + metadata) into data/enriched/ (D49-D57).

    python scripts/enrich.py --sample 20   # timing test: first 20 chunks of the pilot order
    python scripts/enrich.py --pilot       # the 200-chunk stratified pilot (D57)
    python scripts/enrich.py --full        # all 1,696 chunks, then writes chunks.jsonl
    python scripts/enrich.py --report --pilot   # summary from the checkpoint; no LLM call

Resumable: every finished chunk is appended to data/enriched/checkpoint.jsonl and every LLM reply
is in the disk cache, so stopping (Ctrl-C, dead tunnel, 5 transport failures in a row) loses
nothing; run the same command again. Reads data/processed read-only; never writes there.
The LLM endpoint comes from OLLAMA_BASE_URL in .env and is never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.collect.run import CORE_REGISTRY  # noqa: E402
from src.common.cache import append_jsonl, atomic_write_text, read_jsonl  # noqa: E402
from src.common.llm import OllamaClient  # noqa: E402
from src.common.schemas import REPO_ROOT, load_settings, load_tokenizer_dir  # noqa: E402
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.places.places import Registry  # noqa: E402
from src.process.enrich import (  # noqa: E402
    ENRICH_OPTIONS,
    ENRICHED_FIELDS,
    PROMPT_VERSION,
    StopRun,
    assemble,
    enrich_chunk,
    format_report,
    leak_scan,
    load_checkpoint,
    run_batch,
    select_pilot,
    summarize,
)
from src.process.process import check_enriched_matches_frozen, load_counter  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ENRICHED = REPO_ROOT / "data" / "enriched"
CHECKPOINT = ENRICHED / f"checkpoint_{PROMPT_VERSION}.jsonl"  # one file per prompt version
DEBUG_RAW = ENRICHED / "debug_raw.jsonl"  # UNTRUSTED model output, gitignored
SELECTION = ENRICHED / "pilot_selection.json"
OUT = ENRICHED / "chunks.jsonl"
MANIFEST = REPO_ROOT / "data" / "manifests" / "enriched_manifest.json"
SELECTION_SEED = 42
PILOT_SIZE = 200
FORCED_IDS = ("36acb56e137feb5a",)  # D57; a dropped hold-out chunk, so absent from the corpus
SHORT_TOKENS = 200


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen() -> list[dict]:
    """The frozen chunks, after checking the file against the committed freeze manifest."""
    manifest = json.loads((REPO_ROOT / "data" / "manifests" / "frozen_manifest.json")
                          .read_text(encoding="utf-8"))
    path = PROCESSED / "chunks.jsonl"
    if sha256_file(path) != manifest["chunks_jsonl_sha256"]:
        sys.exit("data/processed/chunks.jsonl does not match data/manifests/frozen_manifest.json")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pilot_order(frozen: list[dict]) -> list[dict]:
    """The recorded pilot order, created once from the seeded selection (D57)."""
    if SELECTION.exists():
        return json.loads(SELECTION.read_text(encoding="utf-8"))["order"]
    tok_dir = load_tokenizer_dir()
    if tok_dir is None:
        sys.exit("TOKENIZER_DIR is not set in .env: the pilot needs the real bge-m3 token counts "
                 "to pick the under-200-token tails")
    count, counter_name = load_counter(tok_dir)
    short_ids = [c["chunk_id"] for c in frozen if count(c["text"]) < SHORT_TOKENS]
    present = {c["chunk_id"] for c in frozen}
    forced = [i for i in FORCED_IDS if i in present]
    missing = [i for i in FORCED_IDS if i not in present]
    if missing:  # 36acb56e137feb5a was dropped at the hold-out step: not in the corpus
        print(f"NOTE: forced chunk(s) {missing} are not in the frozen corpus and cannot be "
              "enriched; recorded as forced_missing in pilot_selection.json")
    order = select_pilot(frozen, PILOT_SIZE, SELECTION_SEED, short_ids=short_ids,
                         forced_ids=forced)
    doc = {"seed": SELECTION_SEED, "n": PILOT_SIZE, "forced_ids": forced, "forced_missing": missing,
           "counter": counter_name, "short_chunks_in_corpus": len(short_ids),
           "order": [{"chunk_id": cid, "why": why} for cid, why in order]}
    atomic_write_text(SELECTION, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc["order"]


def scope_ids(args, frozen: list[dict]) -> list[str]:
    if args.full:
        return [c["chunk_id"] for c in frozen]
    order = [o["chunk_id"] for o in pilot_order(frozen)]
    return order[: args.sample] if args.sample else order


def audit(rows: list[dict], holdout: HoldoutFilter) -> int:
    """Final leak audit over every note and mentioned place of the assembled corpus."""
    hits = 0
    for r in rows:
        for text in [r["context_note"], *r["mentioned_places"]]:
            if leak_scan(text, holdout):
                hits += 1
    return hits


def write_enriched(frozen: list[dict], done: dict[str, dict], holdout: HoldoutFilter,
                   model: str) -> int:
    rows = assemble(frozen, {cid: rec["fields"] for cid, rec in done.items()})
    check_enriched_matches_frozen(rows, frozen)
    hits = audit(rows, holdout)
    if hits:
        print(f"LEAK AUDIT FAILED: {hits} hold-out hits in the assembled notes; nothing written")
        return 1
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(OUT, text)
    sm = summarize([done[c["chunk_id"]] for c in frozen])
    manifest = {
        "file": "data/enriched/chunks.jsonl", "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "lines": len(rows), "bytes": len(text.encode()),
        "frozen_chunks_sha256": sha256_file(PROCESSED / "chunks.jsonl"),
        "model": model, "prompt_version": PROMPT_VERSION, "options": ENRICH_OPTIONS,
        "fields": list(ENRICHED_FIELDS), "llm": sm["llm"], "fallback": sm["fallback"],
        "fallback_reasons": sm["fallback_reasons"], "leak_audit_hits": hits,
        "mentioned_dropped": sm["mentioned_dropped"],
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_text(MANIFEST, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(rows)} lines, sha256 {manifest['sha256']}) "
          "and data/manifests/enriched_manifest.json; check_enriched_matches_frozen passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument("--sample", type=int, metavar="N", help="first N of the pilot order")
    scope.add_argument("--pilot", action="store_true", help="the 200-chunk pilot")
    scope.add_argument("--full", action="store_true", help="every frozen chunk")
    ap.add_argument("--report", action="store_true", help="summarise the checkpoint, no LLM call")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="stop after this many consecutive transport failures (default 5)")
    args = ap.parse_args(argv)

    frozen = load_frozen()
    by_id = {c["chunk_id"]: c for c in frozen}
    ids = scope_ids(args, frozen)
    registry = Registry.load(CORE_REGISTRY)
    holdout = HoldoutFilter.from_registry(registry)
    titles = {u: (v.get("title") or "") for u, v in json.loads(
        (PROCESSED / "attribution.json").read_text(encoding="utf-8")).items()}

    if args.report:
        have = {r["chunk_id"]: r for r in read_jsonl(CHECKPOINT)} if CHECKPOINT.exists() else {}
        got = [have[i] for i in ids if i in have]
        print(f"scope: {len(ids)} chunks; in checkpoint: {len(got)}; "
              f"missing: {len(ids) - len(got)}")
        if got:
            print(format_report(summarize(got)))
        return 0

    s = load_settings()
    posts = {"n": 0}

    def counting_post(*a, **k):
        posts["n"] += 1
        return requests.post(*a, **k)

    client = OllamaClient.from_settings(post=counting_post)
    model = s.generator_model
    print(f"generator {model}; judge {s.judge_model} is not used in Phase 3; "
          f"scope {len(ids)} chunks")

    def process(chunk: dict):
        return enrich_chunk(client, model, chunk, titles[chunk["url"]], registry, holdout,
                            debug=lambda rec: append_jsonl(DEBUG_RAW, [rec]))

    ENRICHED.mkdir(parents=True, exist_ok=True)
    try:
        stats = run_batch(ids, by_id, process, CHECKPOINT, model,
                          max_consecutive_failures=args.max_failures, calls=lambda: posts["n"])
    except StopRun as exc:
        print(f"STOPPED: {exc}")
        return 3
    except KeyboardInterrupt:
        print("interrupted; everything finished so far is in the checkpoint. Re-run to resume.")
        return 130
    print(f"run: {stats}; network POSTs this run: {posts['n']}")

    done = load_checkpoint(CHECKPOINT, model)
    got = [done[i] for i in ids if i in done]
    print(format_report(summarize(got)))
    if len(got) < len(ids):
        print(f"INCOMPLETE: {len(ids) - len(got)} chunks failed; re-run the same command")
        return 1
    if args.full:
        return write_enriched(frozen, done, holdout, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
