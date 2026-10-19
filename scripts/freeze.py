"""Freeze the Phase 2 corpus (D48). Default is a dry run: every precondition with pass/fail, nothing
written. Offline.

    python scripts/freeze.py                      # dry run
    python scripts/freeze.py --apply-d33          # dry run of the D33 step (effects table)
    python scripts/freeze.py --apply-d33 --execute   # write both d33_excluded.json copies
    python scripts/freeze.py --execute            # write data/processed/FROZEN (all must pass)

The D33 sidecar is a Phase 4 query-eligibility list: chunks.jsonl is unchanged and retrieval is NOT
filtered, unless the owner records otherwise in config/freeze.json.
Owner decisions come from config/freeze.json (reddit_resolved, d33_mode). Run --apply-d33 --execute
BEFORE the freeze: once FROZEN exists nothing in data/processed may change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.run import CORE_REGISTRY, passes_rule  # noqa: E402
from src.common.schemas import REPO_ROOT, load_reference_date, load_tokenizer_dir  # noqa: E402
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.places.places import Registry  # noqa: E402
from src.process.process import TOKENIZER_FILES, ReconcileError, check_corpus  # noqa: E402

REAL_COUNTER_PREFIX = "tokenizer.json sha256:"
ENV_SOURCE = "EVAL_REFERENCE_DATE in .env"
FROZEN_FILES = ("chunks.jsonl", "drops.jsonl", "attribution.json")  # checked against the manifest
D33_NAME = "d33_excluded.json"  # in data/processed and, committed, in data/manifests


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class Ctx:
    def __init__(self, root: Path, env_file: Path | None):
        self.root = root
        self.processed = root / "data" / "processed"
        self.manifests = root / "data" / "manifests"
        self.config = read_json(root / "config" / "freeze.json")
        self.manifest = read_json(self.manifests / "processed_manifest.json")
        self.stats = read_json(self.manifests / "corpus_stats.json")
        self.env_file = env_file
        path = self.processed / "chunks.jsonl"
        self.chunks = ([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
                       if path.exists() else [])


def tokenizer_check(ctx: Ctx) -> tuple[bool, str]:
    counter = ctx.manifest.get("counter", "")
    if not counter.startswith(REAL_COUNTER_PREFIX):
        return False, f"processed_manifest counter is {counter!r}; the proxy is refused"
    tdir = load_tokenizer_dir(ctx.env_file)
    if tdir is None:
        return False, "TOKENIZER_DIR is not set in .env"
    tj = tdir / "tokenizer.json"
    if not tj.is_file():
        return False, f"{tj} does not exist"
    digest = sha256(tj)[:12]
    if counter != REAL_COUNTER_PREFIX + digest:
        return False, f"counter {counter!r} does not match {tj} (sha256:{digest})"
    return True, f"{counter} from {tdir.name}"


def tokenizer_files(tdir: Path) -> dict[str, str]:
    """sha256 of every file the counter loads from TOKENIZER_DIR (TOKENIZER_FILES)."""
    return {name: sha256(tdir / name) for name in TOKENIZER_FILES}


def tokenizers_version() -> str:
    import tokenizers

    return tokenizers.__version__


def preconditions(ctx: Ctx) -> list[tuple[str, bool, str]]:
    out = []
    frozen = ctx.processed / "FROZEN"
    out.append(("FROZEN does not exist yet", not frozen.exists(),
                str(frozen.relative_to(ctx.root))))
    files = ctx.manifest.get("files", {})
    bad = []
    for name in FROZEN_FILES:
        rel = f"data/processed/{name}"
        p = ctx.processed / name
        if not p.exists() or files.get(rel, {}).get("sha256") != sha256(p):
            bad.append(name)
    out.append(("outputs match processed_manifest.json", not bad and bool(files),
                f"mismatch or missing: {bad}" if bad else "chunks, drops, attribution match"))
    try:
        check_corpus(ctx.chunks, HoldoutFilter.from_registry(Registry.load(CORE_REGISTRY)))
        ok, why = bool(ctx.chunks), f"{len(ctx.chunks)} chunks"
    except ReconcileError as exc:
        ok, why = False, str(exc)
    out.append(("check_corpus() passes", ok, why))
    live = sum(c.get("source_type") == "live" for c in ctx.chunks)
    out.append(('no source_type "live"', live == 0, f"{live} live chunks"))
    out.append(("counter is the real bge-m3 tokenizer", *tokenizer_check(ctx)))
    fresh = ctx.stats.get("freshness", {})
    env_date = load_reference_date(ctx.env_file)
    src_ok = fresh.get("reference_date_source") == ENV_SOURCE
    date_ok = env_date is not None and env_date.isoformat() == fresh.get("reference_date")
    out.append(("reference date comes from .env, not the flag", src_ok and date_ok,
                f"stats: {fresh.get('reference_date')} from "
                f"{fresh.get('reference_date_source')!r}; "
                f".env EVAL_REFERENCE_DATE: {env_date}"))
    out.append(("Reddit recorded as resolved (config/freeze.json)",
                ctx.config.get("reddit_resolved") is True,
                f"reddit_resolved={ctx.config.get('reddit_resolved')!r} "
                f"{ctx.config.get('reddit_note', '')}".strip()))
    mode = ctx.config.get("d33_mode")
    side, copy = ctx.processed / D33_NAME, ctx.manifests / D33_NAME
    same = side.exists() and copy.exists() and sha256(side) == sha256(copy)
    out.append(("D33 applied (d33_mode sidecar; processed and manifests copies identical)",
                mode == "sidecar" and same,
                f"d33_mode={mode!r}, processed copy exists={side.exists()}, manifests copy "
                f"exists={copy.exists()}, identical={same}"))
    return out


def d33_table(ctx: Ctx) -> tuple[list[str], list[str]]:
    reg = Registry.load(CORE_REGISTRY)
    places = [e for e in reg.entries if not e.holdout]
    types: dict[str, set] = {}
    count: dict[str, int] = {}
    for c in ctx.chunks:
        types.setdefault(c["place"], set()).add(c["source_type"])
        count[c["place"]] = count.get(c["place"], 0) + 1
    fail = [e for e in places if not passes_rule(types.get(e.name, set()))]
    keep = [e for e in places if e not in fail]
    n_keep = sum(count.get(e.name, 0) for e in keep)
    hid = sum(e.popularity == "hidden_gem" for e in keep)
    rows = [
        "| behaviour | Phase 4 query-eligible places | chunks of eligible places | chunks in "
        "chunks.jsonl (all retrievable) | hidden_gem eligible |",
        "|---|---|---|---|---|",
        f"| no D33 (today) | {len(places)} | {len(ctx.chunks)} | {len(ctx.chunks)} | "
        f"{sum(e.popularity == 'hidden_gem' for e in places)} |",
        f"| sidecar (implemented): Phase 4 query-eligibility list; chunks.jsonl unchanged; "
        f"retrieval NOT filtered | "
        f"{len(keep)} | {n_keep} | {len(ctx.chunks)} | {hid} |",
        f"| exclude (NOT implemented): chunks.jsonl rewritten | {len(keep)} | {n_keep} | "
        f"{n_keep} | {hid} |",
    ]
    excluded = [f"{e.name} ({count.get(e.name, 0)} chunks, {e.popularity}, types "
                f"{sorted(types.get(e.name, set()))})" for e in fail]
    return rows, excluded


def run(root: Path, env_file: Path | None, execute: bool, apply_d33: bool) -> int:
    ctx = Ctx(root, env_file)
    if apply_d33:
        rows, excluded = d33_table(ctx)
        print("D33 (D23 at chunk level), places that would be excluded:")
        for x in excluded:
            print("  -", x)
        print("\n".join(rows))
        mode = ctx.config.get("d33_mode")
        if mode is None:
            print("REFUSED: config/freeze.json d33_mode is null "
                  "(the owner has not recorded the decision)")
            return 2
        if mode != "sidecar":
            print(f"REFUSED: d33_mode {mode!r} is not implemented; only 'sidecar' is")
            return 2
        if (ctx.processed / "FROZEN").exists():
            print("REFUSED: FROZEN exists")
            return 2
        if not execute:
            print("DRY RUN: would write data/processed/d33_excluded.json")
            return 0
        doc = {"rule": "D23 at chunk level, applied once (D33, D48)",
               "excluded": sorted(x.split(" (")[0] for x in excluded)}
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        for d in (ctx.processed, ctx.manifests):  # the manifests copy is committed (no text)
            (d / D33_NAME).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote data/processed/{D33_NAME} and data/manifests/{D33_NAME}")
        return 0
    checks = preconditions(ctx)
    for name, ok, why in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {why}")
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"{len(failed)} precondition(s) failed: the corpus is NOT frozen.")
        return 2
    if not execute:
        print("DRY RUN: all preconditions pass; --execute would write data/processed/FROZEN")
        return 0
    fresh = ctx.stats["freshness"]
    doc = {
        "chunks_jsonl_sha256": sha256(ctx.processed / "chunks.jsonl"),
        "attribution_json_sha256": sha256(ctx.processed / "attribution.json"),
        "drops_jsonl_sha256": sha256(ctx.processed / "drops.jsonl"),
        "d33_excluded_json_sha256": sha256(ctx.processed / D33_NAME),
        "manifests_d33_excluded_json_sha256": sha256(ctx.manifests / D33_NAME),
        "chunk_count": len(ctx.chunks),
        "counter": ctx.manifest["counter"],
        "tokenizer_files_sha256": tokenizer_files(load_tokenizer_dir(env_file)),
        "tokenizers_version": tokenizers_version(),
        "reference_date": fresh["reference_date"],
        "reference_date_source": fresh["reference_date_source"],
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    (ctx.processed / "FROZEN").write_text(text, encoding="utf-8", newline="\n")
    (ctx.manifests / "frozen_manifest.json").write_text(text, encoding="utf-8", newline="\n")
    print("wrote data/processed/FROZEN and data/manifests/frozen_manifest.json:",
          json.dumps(doc, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="default")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--apply-d33", action="store_true",
                    help="D33 sidecar: write data/processed/d33_excluded.json and its committed "
                         "copy data/manifests/d33_excluded.json, the Phase 4 "
                         "query-eligibility list (no dev/test query may target a listed place). "
                         "chunks.jsonl is unchanged and retrieval is NOT filtered unless the "
                         "owner records otherwise in config/freeze.json")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="tests only")
    ap.add_argument("--env-file", type=Path, default=None, help="tests only (default: .env)")
    a = ap.parse_args()
    return run(a.root, a.env_file, a.execute, a.apply_d33)


if __name__ == "__main__":
    sys.exit(main())
