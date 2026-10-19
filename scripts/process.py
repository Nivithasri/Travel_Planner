"""Phase 2 batch: clean, chunk and tag data/raw into data/processed (D48). Offline: reads only
records.jsonl files under data/raw (the facts envelopes are never read) and makes no network call.

    python scripts/process.py learn          # print learned lines, write config/boilerplate.json
    python scripts/process.py                # dry run: full report, writes nothing
    python scripts/process.py --write        # data/processed/* and data/manifests/corpus_stats.*
    python scripts/process.py --live <slug>  # data/live/raw/*/<slug> -> data/live/processed/

Never freezes: data/processed/FROZEN is created only by the owner-approved freeze step, and
--write refuses to run once it exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collect.run import CORE_REGISTRY, RAW_ROOT, SOURCE_TYPE, place_coverage  # noqa: E402
from src.common.cache import atomic_write_text, read_jsonl  # noqa: E402
from src.common.schemas import (  # noqa: E402
    REPO_ROOT,
    load_reference_date,
    load_tokenizer_dir,
)
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.places.places import Registry, slugify  # noqa: E402
from src.process.process import (  # noqa: E402
    attribution,
    check_corpus,
    corpus_stats,
    dedupe_records,
    learn_boilerplate,
    load_counter,
    near_dups,
    process_place,
    reconcile,
)

RULES_PATH = REPO_ROOT / "config" / "boilerplate.json"
LEARN_HOSTS = ("inditales.com",)
PROCESSED = REPO_ROOT / "data" / "processed"
MANIFESTS = REPO_ROOT / "data" / "manifests"
LIVE_ROOT = REPO_ROOT / "data" / "live"


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def jsonl(rows) -> str:
    return "".join(dumps(r) + "\n" for r in rows)


def load_raw(root: Path, slug: str | None = None) -> list[dict]:
    """Every record in <root>/<source>/<place>/records.jsonl for the chunkable sources only."""
    out = []
    for source in sorted(SOURCE_TYPE):
        for f in sorted((root / source).glob(f"{slug or '*'}/records.jsonl")):
            out.extend(read_jsonl(f))
    return out


def load_rules() -> dict:
    if not RULES_PATH.exists():
        sys.exit(f"{RULES_PATH} is missing: run `python scripts/process.py learn` first")
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def run_batch(write: bool, reference_date: date | None = None) -> int:
    registry = Registry.load(CORE_REGISTRY)
    order = {e.name: i for i, e in enumerate(registry.entries)}
    holdout = HoldoutFilter.from_registry(registry)
    rules = load_rules()
    count, counter_name = load_counter(load_tokenizer_dir())
    env_date = load_reference_date()
    if reference_date is not None:
        ref_date, ref_source = reference_date, "--reference-date (overrides .env)"
    elif env_date is not None:
        ref_date, ref_source = env_date, "EVAL_REFERENCE_DATE in .env"
    else:
        sys.exit("EVAL_REFERENCE_DATE is not set in .env: pass --reference-date YYYY-MM-DD (D4)")
    records = load_raw(RAW_ROOT)
    for r in records:  # scope check: Phase 1 must never have produced these
        e = registry.get(r["place"])
        if e is None or e.holdout:
            raise SystemExit(f"record for {r['place']!r} is not a collectable registry place")

    kept, dups = dedupe_records(records, order, rules)
    by_place: dict[str, list[dict]] = {}
    for r in kept:
        by_place.setdefault(r["place"], []).append(r)
    reports, rows = [], []
    for name in sorted(by_place):
        out, rep = process_place(by_place[name], registry.get(name), holdout, rules, count)
        reports.append(rep)
        rows.extend((name, c["url"], off, c) for off, c in out)
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    chunks, near = near_dups([c for *_, c in rows])
    rec_drops = [d for r in reports for d in r.record_drops]
    chunk_drops = [d for r in reports for d in r.chunk_drops]
    created = sum(r.chunks_created for r in reports)
    recon = reconcile(len(records), len(kept) - len(rec_drops), len(dups), len(rec_drops),
                      created, len(chunks), len(chunk_drops), len(near))
    drops = sorted(dups + rec_drops + chunk_drops + near,
                   key=lambda d: (d["level"], d["place"], d["url"], d.get("chunk_id", "")))
    places = [e.name for e in registry.entries if not e.holdout]
    d23_raw = {e.name: place_coverage(e, RAW_ROOT).passes for e in registry.entries
               if not e.holdout}
    check_corpus(chunks, holdout)  # raises: live chunk, hold-out mention, duplicate id, schema
    stats = corpus_stats(chunks, reports, drops, count, counter_name, ref_date, d23_raw, places,
                         ref_source)
    stats["reconciliation"] = recon
    md = render_md(stats, dups, rec_drops)
    print(md)
    if not write:
        print("\nDRY RUN: nothing written (use --write).")
        return 0
    if (PROCESSED / "FROZEN").exists():
        sys.exit("data/processed/FROZEN exists: the corpus is frozen, --write refused")
    files = {
        PROCESSED / "chunks.jsonl": jsonl(chunks),
        PROCESSED / "drops.jsonl": jsonl(drops),
        PROCESSED / "attribution.json": json.dumps(attribution(kept, chunks), sort_keys=True,
                                                   ensure_ascii=False, indent=1) + "\n",
        MANIFESTS / "corpus_stats.json": json.dumps(stats, sort_keys=True, ensure_ascii=False,
                                                    indent=1) + "\n",
        MANIFESTS / "corpus_stats.md": md + "\n",
    }
    for path, text in files.items():
        atomic_write_text(path, text)
    manifest = {"counter": counter_name, "files": {
        p.relative_to(REPO_ROOT).as_posix(): {
            "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(),
            "bytes": len(t.encode("utf-8")), "lines": t.count("\n")}
        for p, t in files.items() if p.parent == PROCESSED}}
    atomic_write_text(MANIFESTS / "processed_manifest.json",
                      json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    for p, meta in manifest["files"].items():
        print(f"wrote {p}: {meta['lines']} lines, sha256 {meta['sha256']}")
    return 0


def render_md(s: dict, dups: list[dict], rec_drops: list[dict]) -> str:
    est = s["estimate"]
    t = s["tokens"]
    lines = [
        "# Corpus statistics (Phase 2, NOT frozen)",
        "",
        "Command: `python scripts/process.py --write` (offline; reads data/raw records only).",
        f"Token counter: **{s['counter']}**. Parameters: {dumps(s['params'])}.",
        "",
        "## Reconciliation", "", *[f"- {r}" for r in s["reconciliation"]], "",
        f"## Size: {s['chunks']} chunks", "",
        f"Estimate {est['chunks']} (range {est['low']} to {est['high']}, {est['source']}); "
        f"difference {est['difference']:+d}.",
        f"Tokens per chunk: min {t['min']}, p50 {t['p50']}, p95 {t['p95']}, max {t['max']}; "
        f"under {s['params']['min']}: {t['under_min']}; over {s['params']['max_incl_overlap']}: "
        f"{t['over_max']}. Chunking: {dumps(s['chunking'])}.",
        "",
    ]
    for title, key in (("source_type", "per_source_type"), ("state", "per_state"),
                       ("published year", "per_published_year"),
                       ("revised year (wiki API revision date)", "per_revised_year"),
                       ("date_kind", "date_kind")):
        lines += [f"## Chunks per {title}", "", "| value | chunks |", "|---|---|",
                  *[f"| {k} | {v} |" for k, v in s[key].items()], ""]
    lines += ["## Chunks per place", "", "| place | chunks | by source_type |", "|---|---|---|",
              *[f"| {p} | {sum(v.values())} | {dumps(v)} |" for p, v in s["per_place"].items()],
              ""]
    f = s["freshness"]
    lines += [f"Null-date share: {s['null_date_share']}. Freshness (published dates only, window "
              f"{f['window_start']} to {f['reference_date']}; reference date from "
              f"{f['reference_date_source']}): {f['published_in_window']} chunks, "
              f"share {f['share']}. The reference date is used only here; it is in no chunk.", "",
              "## Removed text (characters per rule)", "",
              *[f"- {k}: {v}" for k, v in s["removed_chars"].items()], "",
              "## Drops", "", *[f"- {k}: {v}" for k, v in s["drops"].items()], "",
              "Exact duplicates:", *[f"- {d['url']} under {d['place']} (kept under "
                                     f"{d['kept_place']})" for d in dups], "",
              "Record-level hold-out drops (url/title):",
              *[f"- {d['url']} ({d['place']}, {d['source']}): {d['holdout']} via {d['where']} "
                f"'{d['term']}'" for d in rec_drops], "",
              "Hold-out drops per hold-out, source and level:", "",
              "| hold-out | source | level | n |", "|---|---|---|---|",
              *[f"| {d['holdout']} | {d['source']} | {d['level']} | {d['n']} |"
                for d in s["holdout_drops"]], "",
              "Boundary-free stem scan of the OUTPUT chunks (case-insensitive substring; hits "
              "are other words or concatenations, listed for review):",
              *[f"- {k}: {v}" for k, v in s["substring_scan"].items()], "",
              "## D23 / D33 dry run (nothing removed)", "",
              "| place | types in chunks | passes (chunks) | passes (raw) |", "|---|---|---|---|",
              *[f"| {p} | {', '.join(v['types_in_chunks'])} | {v['passes_chunks']} | "
                f"{v['passes_raw']} |" for p, v in s["d23_dry_run"].items()]]
    n_fail = sum(not v["passes_chunks"] for v in s["d23_dry_run"].values())
    lines += ["", f"D23 at chunk level: {len(s['d23_dry_run']) - n_fail} of "
                  f"{len(s['d23_dry_run'])} pass; {n_fail} would be excluded at the freeze (D33)."]
    return "\n".join(lines)


def run_learn() -> int:
    registry = Registry.load(CORE_REGISTRY)
    records = [r for r in load_raw(RAW_ROOT) if not registry.get(r["place"]).holdout]
    old = json.loads(RULES_PATH.read_text(encoding="utf-8")) if RULES_PATH.exists() else {}
    rejected = {h: {x["line"] for x in v.get("rejected", [])}
                for h, v in old.get("hosts", {}).items()}
    doc = learn_boilerplate(records, LEARN_HOSTS, rejected=rejected)
    for host, v in doc["hosts"].items():
        print(f"{host}: {len(v['lines'])} applied, {len(v['rejected'])} rejected by review "
              f"(>= {doc['min_records']} records, < {doc['max_chars']} chars)")
        for kind in ("lines", "rejected"):
            for x in v[kind]:
                print(f"  {kind:<8} {x['records']:>3}  {x['line']!r}")
    atomic_write_text(RULES_PATH, json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {RULES_PATH.relative_to(REPO_ROOT).as_posix()}")
    return 0


def run_live(slug: str) -> int:
    live_reg = Registry.load(LIVE_ROOT / "registry.json")
    entry = next((e for e in live_reg.entries if slugify(e.name) == slug), None)
    if entry is None:
        sys.exit(f"{slug!r} is not in data/live/registry.json")
    holdout = HoldoutFilter.from_registry(Registry.load(CORE_REGISTRY))
    count, counter_name = load_counter(load_tokenizer_dir())
    records = load_raw(LIVE_ROOT / "raw", slug)
    out, rep = process_place(records, entry, holdout, load_rules(), count)
    chunks = [c for _, c in sorted(out, key=lambda x: (x[1]["url"], x[0]))]
    path = LIVE_ROOT / "processed" / f"{slug}.jsonl"
    atomic_write_text(path, jsonl(chunks))
    print(f"{entry.name}: {len(records)} records, {rep.chunks_created} chunks created, "
          f"{len(rep.record_drops)} record and {len(rep.chunk_drops)} chunk hold-out drops, "
          f"{len(chunks)} written to {path.relative_to(REPO_ROOT).as_posix()} ({counter_name})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["learn", "run"], default="run")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--live", metavar="SLUG")
    ap.add_argument("--reference-date", type=date.fromisoformat,
                    help="overrides EVAL_REFERENCE_DATE from .env (D4)")
    args = ap.parse_args()
    if args.mode == "learn":
        return run_learn()
    if args.live:
        return run_live(args.live)
    return run_batch(args.write, args.reference_date)


if __name__ == "__main__":
    sys.exit(main())
