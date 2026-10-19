"""Phase 3 batch: enrich the frozen chunks (context note + metadata) with a hosted LLM (D59).

    python scripts/enrich.py --sample 20            # first 20 chunks of the pilot order
    python scripts/enrich.py --pilot                # 60 seeded chunks + the 30 gold-sample chunks
    python scripts/enrich.py --gold                 # the 30 chunks of data/enriched/gold_sample.csv
    python scripts/enrich.py --full                 # every chunk in scope
    python scripts/enrich.py --report --pilot       # summary + projection; no API call
    python scripts/enrich.py --assemble             # write data/enriched/chunks.jsonl; no API call

Scope flags: --eligible-only keeps the 17 D33-eligible places (the 9 excluded places are later
labelled rule-only); blog chunks are NOT sent unless --include-blog (only after the owner has
confirmed the provider's data-use terms); --limit N stops after N new chunks.

Resumable and budgeted: every finished chunk is appended to a checkpoint (one file per prompt
version AND model) and every reply is in the disk cache; a client-side limiter keeps RPM, TPM, RPD
and TPD under the limits in .env (20% margin; 85% of TPD) and remembers the day's spend in
data/enriched/quota_state.json, so --full runs across several days: run the same command again.
Exit codes: 0 ok, 1 incomplete, 3 five failures in a row, 4 key/model/config rejected, 5 daily
quota used up, 130 Ctrl-C. Reads data/processed read-only. The URL, key and headers are never
printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.collect.run import CORE_REGISTRY  # noqa: E402
from src.common.cache import append_jsonl, atomic_write_text, read_jsonl  # noqa: E402
from src.common.llm import (  # noqa: E402
    AuthError,
    ModelNotFoundError,
    OpenAICompatClient,
    QuotaExhausted,
)
from src.common.schemas import (  # noqa: E402
    REPO_ROOT,
    ConfigError,
    env_duplicate_keys,
    load_llm_settings,
    load_tokenizer_dir,
)
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.places.places import Registry  # noqa: E402
from src.process.enrich import (  # noqa: E402
    ENRICH_OPTIONS,
    ENRICHED_FIELDS,
    PROMPT_VERSION,
    StopRun,
    assemble_labelled,
    candidate_pool,
    enrich_chunk,
    format_projection,
    format_report,
    leak_scan,
    load_checkpoint,
    project_days,
    run_batch,
    select_pilot,
    summarize,
)
from src.process.process import check_enriched_matches_frozen, load_counter  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
ENRICHED = REPO_ROOT / "data" / "enriched"
DEBUG_RAW = ENRICHED / "debug_raw.jsonl"  # UNTRUSTED model output (scrubbed), gitignored
GOLD_CSV = ENRICHED / "gold_sample.csv"
OUT = ENRICHED / "chunks.jsonl"
PROVENANCE = ENRICHED / "provenance.jsonl"
MANIFEST = REPO_ROOT / "data" / "manifests" / "enriched_manifest.json"
EXCLUDED_JSON = REPO_ROOT / "data" / "manifests" / "d33_excluded.json"
SELECTION_SEED = 42
PILOT_SIZE = 60
SHORT_TOKENS = 200
EXIT_STOP, EXIT_CONFIG, EXIT_QUOTA = 3, 4, 5


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def checkpoint_path(model: str) -> Path:
    """One file per prompt version AND model: results of different prompts or models are never
    mixed (load_checkpoint refuses a record from another version or model as well)."""
    return ENRICHED / f"checkpoint_{PROMPT_VERSION}_{slug(model)}.jsonl"


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


def gold_ids() -> list[str]:
    if not GOLD_CSV.exists():
        sys.exit("data/enriched/gold_sample.csv is missing: run scripts/label_sample.py first")
    with open(GOLD_CSV, encoding="utf-8-sig", newline="") as fh:
        return [r["chunk_id"] for r in csv.DictReader(fh)]


def pilot_ids(pool: list[dict], flags: str) -> list[str]:
    """The seeded 60-chunk pilot over `pool` (D59), recorded once per flag combination."""
    path = ENRICHED / f"pilot{PILOT_SIZE}_{flags}.json"
    if path.exists():
        return [o["chunk_id"] for o in json.loads(path.read_text(encoding="utf-8"))["order"]]
    tok_dir = load_tokenizer_dir()
    if tok_dir is None:
        sys.exit("TOKENIZER_DIR is not set in .env: needed to find the under-200-token tails")
    count, counter_name = load_counter(tok_dir)
    short_ids = [c["chunk_id"] for c in pool if count(c["text"]) < SHORT_TOKENS]
    order = select_pilot(pool, PILOT_SIZE, SELECTION_SEED, short_ids=short_ids, n_url=6,
                         n_short=10)
    doc = {"seed": SELECTION_SEED, "n": PILOT_SIZE, "flags": flags, "counter": counter_name,
           "pool_size": len(pool), "short_chunks_in_pool": len(short_ids),
           "order": [{"chunk_id": cid, "why": why} for cid, why in order]}
    atomic_write_text(path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return [cid for cid, _ in order]


def scope(args, frozen: list[dict], excluded: list[str]) -> tuple[list[str], dict]:
    """(chunk ids to process in order, info). Filtered by --eligible-only / --include-blog; the
    skipped chunks are counted and reported, never silently dropped."""
    kw = {"excluded_places": excluded, "eligible_only": args.eligible_only,
          "include_blog": args.include_blog}
    pool, skipped = candidate_pool(frozen, **kw)
    by_id = {c["chunk_id"]: c for c in frozen}
    flags = f"blog-{'on' if args.include_blog else 'off'}_{'elig' if args.eligible_only else 'all'}"
    if args.full:
        return [c["chunk_id"] for c in pool], {"skipped": dict(skipped)}
    gold = gold_ids()
    gold_pool, _ = candidate_pool([by_id[i] for i in gold], **kw)
    gold_ok = [c["chunk_id"] for c in gold_pool]
    info = {"skipped": {"gold_chunks_skipped": len(gold) - len(gold_ok)}}
    if args.gold:
        return gold_ok, info
    pilot = pilot_ids(pool, flags)
    if args.sample:
        return pilot[: args.sample], info
    return pilot + [i for i in gold_ok if i not in set(pilot)], info


def projections(sm: dict, s, frozen: list[dict], excluded: list[str]) -> list[str]:
    """PROJECTIONS from the measured cost per chunk (labelled as projections)."""
    if not sm["tokens_per_chunk_mean"] or not sm["calls_per_chunk_mean"]:
        return ["PROJECTION: no chunk with a measured token count yet"]
    elig = [c for c in frozen if c["place"] not in set(excluded)]
    sets = [("all chunks", len(frozen)), ("eligible places only", len(elig)),
            ("eligible places only, no blog", sum(c["source_type"] != "blog" for c in elig))]
    return [format_projection(
        f"{label}, mean {sm['tokens_per_chunk_mean']:.0f} tokens and "
        f"{sm['calls_per_chunk_mean']:.2f} calls per chunk",
        project_days(sm["tokens_per_chunk_mean"], sm["calls_per_chunk_mean"], n, s.tpd, s.rpd))
        for label, n in sets]


def audit(rows: list[dict], holdout: HoldoutFilter) -> int:
    hits = 0
    for r in rows:
        for text in [r["context_note"], *r["mentioned_places"]]:
            if leak_scan(text, holdout):
                hits += 1
    return hits


def assemble(args, frozen, excluded, titles, registry, holdout) -> int:
    ckpts = sorted(ENRICHED.glob(f"checkpoint_{PROMPT_VERSION}_*.jsonl"))
    if len(ckpts) != 1:
        print(f"expected exactly one checkpoint for {PROMPT_VERSION}, found {len(ckpts)}")
        return 1
    recs = {r["chunk_id"]: r for r in read_jsonl(ckpts[0])}
    models = {r["prov"].get("model") for r in recs.values()}
    if len(models) != 1:
        print("the checkpoint holds records of more than one model: refusing to assemble")
        return 1
    try:
        rows, prov = assemble_labelled(
            frozen, recs, titles, registry, excluded_places=excluded,
            eligible_only=args.eligible_only, include_blog=args.include_blog)
    except KeyError as exc:
        print(f"{exc.args[0]}: finish the run first (python scripts/enrich.py --full ...)")
        return 1
    check_enriched_matches_frozen(rows, frozen)
    hits = audit(rows, holdout)
    if hits:
        print(f"LEAK AUDIT FAILED: {hits} hold-out hits in the assembled notes; nothing written")
        return 1
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(OUT, text)
    ptext = "".join(json.dumps(p, sort_keys=True) + "\n" for p in prov)
    atomic_write_text(PROVENANCE, ptext)
    by_src: dict[str, int] = {}
    for p in prov:
        key = p["source"] + (f":{p['reason']}" if p["source"] != "llm" else "")
        by_src[key] = by_src.get(key, 0) + 1
    manifest = {
        "file": "data/enriched/chunks.jsonl", "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "lines": len(rows), "bytes": len(text.encode()),
        "provenance_sha256": hashlib.sha256(ptext.encode()).hexdigest(),
        "frozen_chunks_sha256": sha256_file(PROCESSED / "chunks.jsonl"),
        "model": next(iter(models)), "prompt_version": PROMPT_VERSION, "options": ENRICH_OPTIONS,
        "fields": list(ENRICHED_FIELDS), "by_source": by_src, "leak_audit_hits": hits,
        "scope": {"eligible_only": args.eligible_only, "include_blog": args.include_blog},
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    atomic_write_text(MANIFEST, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(rows)} lines, sha256 {manifest['sha256']}), "
          f"provenance.jsonl and data/manifests/enriched_manifest.json; by source {by_src}; "
          "check_enriched_matches_frozen passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, metavar="N", help="first N of the pilot order")
    sel.add_argument("--pilot", action="store_true", help="60-chunk pilot + the gold chunks")
    sel.add_argument("--gold", action="store_true", help="the 30 gold_sample.csv chunks")
    sel.add_argument("--full", action="store_true", help="every chunk in scope")
    ap.add_argument("--assemble", action="store_true", help="write the enriched file, no API call")
    ap.add_argument("--report", action="store_true", help="summarise the checkpoint, no API call")
    ap.add_argument("--limit", type=int, metavar="N", help="stop after N NEW chunks this run")
    ap.add_argument("--eligible-only", action="store_true", help="only the 17 D33-eligible places")
    ap.add_argument("--include-blog", action="store_true",
                    help="also send blog chunks (only after the data-use terms are confirmed)")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="stop after this many consecutive transport failures (default 5)")
    args = ap.parse_args(argv)
    if not (args.sample or args.pilot or args.gold or args.full or args.assemble):
        ap.error("choose one of --sample N, --pilot, --gold, --full, --assemble")

    frozen = load_frozen()
    by_id = {c["chunk_id"]: c for c in frozen}
    registry = Registry.load(CORE_REGISTRY)
    holdout = HoldoutFilter.from_registry(registry)
    excluded = json.loads(EXCLUDED_JSON.read_text(encoding="utf-8"))["excluded"]
    titles = {u: (v.get("title") or "") for u, v in json.loads(
        (PROCESSED / "attribution.json").read_text(encoding="utf-8")).items()}

    if args.assemble:
        return assemble(args, frozen, excluded, titles, registry, holdout)

    ids, info = scope(args, frozen, excluded)
    print(f"scope: {len(ids)} chunks to process; not sent: {info['skipped']}; "
          f"eligible_only={args.eligible_only} include_blog={args.include_blog}")

    if args.report:
        found = sorted(ENRICHED.glob(f"checkpoint_{PROMPT_VERSION}_*.jsonl"))
        have = {r["chunk_id"]: r for p in found for r in read_jsonl(p)}
        got = [have[i] for i in ids if i in have]
        print(f"checkpoint files for {PROMPT_VERSION}: {len(found)}; in checkpoint: {len(got)} "
              f"of {len(ids)}")
        if got:
            sm = summarize(got)
            print(format_report(sm))
            try:
                for line in projections(sm, load_llm_settings(), frozen, excluded):
                    print(line)
            except ConfigError as exc:
                print(f"(no projection: {exc})")
        return 0

    try:
        s = load_llm_settings()
    except ConfigError as exc:
        print(f"REFUSING TO RUN: {exc}")
        return EXIT_CONFIG
    dup = env_duplicate_keys()
    if dup:
        print(f"WARNING: .env defines these keys more than once (the last one wins): {dup}")
    posts = {"n": 0}

    def counting_post(*a, **k):
        posts["n"] += 1
        return requests.post(*a, **k)

    client = OpenAICompatClient.from_settings(post=counting_post)
    model, lim = s.generator_model, client.limiter
    ckpt = checkpoint_path(model)
    print(f"generator {model}; judge {s.judge_model} is not used in Phase 3; prompt "
          f"{PROMPT_VERSION}; budgets per minute {lim.rpm_eff} requests / {int(lim.tpm_eff)} "
          f"tokens, per day {lim.rpd_eff} requests / {int(lim.tpd_eff)} tokens; spent today "
          f"{lim.day_tokens} tokens / {lim.day_requests} requests")

    def process(chunk: dict):
        return enrich_chunk(client, model, chunk, titles[chunk["url"]], registry, holdout,
                            debug=lambda rec: append_jsonl(DEBUG_RAW, [rec]))

    def usage() -> dict:
        return {k: client.usage.get(k, 0) for k in ("prompt", "completion", "reasoning", "total")}

    ENRICHED.mkdir(parents=True, exist_ok=True)
    resume = "re-run the same command to resume"
    try:
        stats = run_batch(ids, by_id, process, ckpt, model,
                          max_consecutive_failures=args.max_failures, max_new=args.limit,
                          calls=lambda: posts["n"], usage=usage)
    except QuotaExhausted as exc:
        snap = lim.snapshot()
        reset = ("unknown" if exc.reset_in_s is None
                 else f"in about {exc.reset_in_s / 3600:.1f} h ({exc.reset_in_s:.0f} s)")
        print(f"DAILY QUOTA USED UP: {exc.reason}. Tokens used today {snap['tokens_today']}, "
              f"requests {snap['requests_today']}; resets {reset}. Nothing was lost: {resume}.")
        return EXIT_QUOTA
    except (ModelNotFoundError, AuthError) as exc:
        print(f"STOPPED, nothing was processed or lost: {exc}")
        return EXIT_CONFIG
    except StopRun as exc:
        print(f"STOPPED: {exc}")
        return EXIT_STOP
    except KeyboardInterrupt:
        print(f"interrupted; everything finished so far is in the checkpoint. {resume}.")
        return 130
    print(f"run: {stats}; network POSTs this run: {posts['n']}; limiter {lim.snapshot()}")

    done = load_checkpoint(ckpt, model)
    got = [done[i] for i in ids if i in done]
    sm = summarize(got)
    print(format_report(sm))
    for line in projections(sm, s, frozen, excluded):
        print(line)
    if len(got) < len(ids):
        print(f"INCOMPLETE: {len(ids) - len(got)} chunks not done yet; {resume}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
