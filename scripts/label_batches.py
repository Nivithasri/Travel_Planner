"""Batches for a chat-based labeller, and the merge of its replies (round 2, D91). No network.

    python scripts/label_batches.py export --split dev [--size 25] [--exclude-blog]
    python scripts/label_batches.py merge  --split dev [--allow-partial] [--assessor TEXT]

EXPORT reads eval/sheets/judge2_<split>.csv and writes eval/sheets/batches/<split>_batch_NN.txt
(about --size rows each; ONE QUERY IS NEVER SPLIT, so a query with more rows than --size gets a
batch of its own) plus <split>_manifest.json (which rows are in which batch and the sheet's
sha256). A batch shows only row_id, query_id, query text, place, state, source type, page title,
date kind and the chunk text: no chunk id, URL, label, note, rubric, system, rank or provenance.
--exclude-blog leaves the blog rows out of the batches (they stay blank, for the owner to label by
hand); the manifest lists them.

MERGE reads the reply files eval/sheets/replies/<split>_*.txt (paste each reply of the chat into
one file, or all of them into one; the order does not matter) and fills eval/sheets/
judge2_<split>_labelled.csv. A reply line is  row_id|label|note  and a reply ends with an
UNSURE: line. Everything is validated first and NOTHING is written unless it all passes: each
row_id must be in the manifest, exactly once, with a label of 0, 1 or 2 (a float only as exactly
0.0, 1.0 or 2.0), and every non-excluded row must be answered (or labelled by hand) unless
--allow-partial. Rows the chat marked UNSURE get " [unsure]" in the note column.

HAND LABELS ARE NEVER OVERWRITTEN. If the labelled file already holds a label that this script did
not write (the owner's own), it is kept; where the chat also answered that row, the two are compared
and the agreement (exact, kappa) is printed: this is the check that the chat labels "like the
owner". Which labels came from the chat and which from hand is recorded, ids and numbers only, in
eval/labelling_r2_provenance.json. The importer (import_labels.py --round 2) still rejects any
blank label, so rows left blank (excluded blog rows, rows not answered) must be filled by hand
first. Exit codes: 0 ok, 1 validation problems (nothing written), 2 a file is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import pooling  # noqa: E402
from scripts.kappa import confusion, kappa  # noqa: E402
from src.common.cache import atomic_write_text  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402

SHEETS = REPO_ROOT / "eval" / "sheets"
PROVENANCE = REPO_ROOT / "eval" / "labelling_r2_provenance.json"
SPLITS = ("dev", "test")
DEFAULT_SIZE = 25
DEFAULT_ASSESSOR = "LLM chat assistant (model not recorded by this script)"
SHOWN = ("row_id", "query_id", "query_text", "place", "state", "source_type", "title",
         "date_kind", "text")  # the only sheet columns a batch shows
UNSURE_NOTE = " [unsure]"
_HEADER = re.compile(r"^row_id\s*\|\s*label\s*\|\s*note$", re.IGNORECASE)
_UNSURE = re.compile(r"^UNSURE\s*:\s*(.*)$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==== export ==========================================================================

def make_batches(rows: list[dict], size: int = DEFAULT_SIZE) -> list[list[dict]]:
    """Group the rows by query (first-appearance order, sheet order inside a query) and pack whole
    queries into batches of at most `size` rows. A query with more than `size` rows becomes one
    batch on its own; a query is never split."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["query_id"], []).append(r)
    batches: list[list[dict]] = []
    current: list[dict] = []
    for group in groups.values():
        if current and len(current) + len(group) > size:
            batches.append(current)
            current = []
        current = current + group
    if current:
        batches.append(current)
    return batches


def one_line(text: str) -> str:
    return " ".join((text or "").split())


def format_batch(name: str, rows: list[dict]) -> str:
    queries = list(dict.fromkeys(r["query_id"] for r in rows))
    out = [f"BATCH {name} | {len(rows)} rows | queries {', '.join(queries)}",
           f"Reply with ONLY the lines row_id|label|note for these {len(rows)} rows, in this "
           "order, then the UNSURE: line.", ""]
    for r in rows:
        out += [f"[row {r['row_id']}] query {r['query_id']}",
                f"QUERY: {one_line(r['query_text'])}",
                f"PLACE: {r['place']} | STATE: {r['state']} | SOURCE: {r['source_type']} | "
                f"DATE_KIND: {r['date_kind']}",
                f"TITLE: {one_line(r['title'])}",
                f"TEXT: {one_line(r['text'])}", ""]
    return "\n".join(out)


def export_split(split: str, sheets_dir: Path, size: int = DEFAULT_SIZE,
                 exclude_blog: bool = False) -> dict:
    sheet = Path(sheets_dir) / f"judge2_{split}.csv"
    rows = pooling.read_sheet(sheet)
    excluded = [r["row_id"] for r in rows if exclude_blog and r["source_type"] == "blog"]
    sendable = [r for r in rows if r["row_id"] not in set(excluded)]
    out_dir = Path(sheets_dir) / "batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{split}_batch_*.txt"):
        old.unlink()  # generated files only; a smaller re-export must not leave stale batches
    batches = {}
    for i, batch in enumerate(make_batches(sendable, size), 1):
        name = f"{split}_batch_{i:02d}"
        atomic_write_text(out_dir / f"{name}.txt", format_batch(name, batch) + "\n")
        batches[name] = {"row_ids": [r["row_id"] for r in batch],
                         "queries": list(dict.fromkeys(r["query_id"] for r in batch))}
    manifest = {"split": split, "size": size, "exclude_blog": exclude_blog,
                "sheet_sha256": sha256_file(sheet), "n_rows": len(rows),
                "excluded_row_ids": excluded, "batches": batches}
    atomic_write_text(out_dir / f"{split}_manifest.json",
                      json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ==== reply parsing ===================================================================

def parse_reply(text: str, where: str = "reply"
                ) -> tuple[list[tuple[str, int, str]], set[str], list[str]]:
    """(entries (row_id, label, note), unsure row_ids, problems) for one reply text. Tolerated:
    blank lines, code fences, an echoed header line, a bare READY. Anything else that is not a
    row line or an UNSURE line is reported, never guessed."""
    entries, unsure, problems = [], set(), []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("```") or line.upper() == "READY" or _HEADER.match(line):
            continue
        m = _UNSURE.match(line)
        if m:
            body = m.group(1).strip()
            if body.lower() not in ("", "none"):
                for tok in re.split(r"[,\s]+", body):
                    if tok.isdigit():
                        unsure.add(tok)
                    elif tok:
                        problems.append(f"{where} line {n}: UNSURE entry {tok!r} is not a row id")
            continue
        parts = line.split("|", 2)
        rid = parts[0].strip()
        if len(parts) < 2 or not rid.isdigit():
            problems.append(f"{where} line {n}: not a row line (row_id|label|note): {line[:60]!r}")
            continue
        label = pooling.parse_label(parts[1])
        if label is None:
            problems.append(f"{where} line {n}: row {rid}: label {parts[1].strip()!r} is not "
                            "0, 1 or 2")
            continue
        note = parts[2].strip() if len(parts) == 3 else ""
        if "|" in note:
            problems.append(f"{where} line {n}: row {rid}: the note contains a | character")
            continue
        entries.append((rid, label, note))
    return entries, unsure, problems


# ==== merge ===========================================================================

def agreement(pairs: list[tuple[int, int]]) -> dict:
    """Owner-vs-chat agreement on the rows both labelled: n, exact agreement, kappa (unweighted
    and linear) and the confusion table (rows: owner, columns: chat)."""
    n = len(pairs)
    table = confusion([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": n, "agree": sum(1 for a, b in pairs if a == b),
            "agreement": (sum(1 for a, b in pairs if a == b) / n) if n else None,
            "kappa": kappa(table) if n else None,
            "kappa_linear": kappa(table, "linear") if n else None, "confusion": table}


def _group_by_batch(row_ids: list[str], manifest: dict) -> dict[str, list[str]]:
    where = {rid: name for name, b in manifest["batches"].items() for rid in b["row_ids"]}
    out: dict[str, list[str]] = {}
    for rid in row_ids:
        out.setdefault(where.get(rid, "?"), []).append(rid)
    return out


def merge_split(split: str, sheets_dir: Path, provenance_path: Path, *,
                allow_partial: bool = False, assessor: str = DEFAULT_ASSESSOR
                ) -> tuple[dict, list[str]]:
    """(report, problems). Nothing is written when problems is non-empty."""
    sheets_dir = Path(sheets_dir)
    original_path = sheets_dir / f"judge2_{split}.csv"
    out_path = sheets_dir / f"judge2_{split}_labelled.csv"
    manifest_path = sheets_dir / "batches" / f"{split}_manifest.json"
    if not original_path.exists() or not manifest_path.exists():
        return {}, [f"{original_path.name} or {manifest_path.name} is missing: run "
                    f"`label_batches.py export --split {split}` first"]
    original = pooling.read_sheet(original_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest["sheet_sha256"] != sha256_file(original_path):
        return {}, [f"{original_path.name} changed since the batches were exported (sha256): "
                    "export again and redo the replies"]
    by_id = {r["row_id"]: r for r in original}
    excluded = set(manifest["excluded_row_ids"])
    sendable = {rid for b in manifest["batches"].values() for rid in b["row_ids"]}

    # ---- the replies
    reply_files = sorted((sheets_dir / "replies").glob(f"{split}_*.txt"))
    if not reply_files:
        return {}, [f"no reply files eval/sheets/replies/{split}_*.txt"]
    labels: dict[str, int] = {}
    notes: dict[str, str] = {}
    unsure: set[str] = set()
    seen_in: dict[str, str] = {}
    for f in reply_files:
        entries, uns, probs = parse_reply(f.read_text(encoding="utf-8"), f.name)
        problems += probs
        unsure |= uns
        for rid, label, note in entries:
            if rid not in by_id:
                problems.append(f"{f.name}: row {rid} is not in {original_path.name}")
            elif rid in excluded:
                problems.append(f"{f.name}: row {rid} was excluded from the batches (blog)")
            elif rid in seen_in:
                problems.append(f"{f.name}: row {rid} is answered twice (also in {seen_in[rid]})")
            else:
                seen_in[rid] = f.name
                labels[rid], notes[rid] = label, note
    for rid in sorted(unsure - set(labels)):
        problems.append(f"UNSURE names row {rid}, which has no valid reply line")

    # ---- what is already in the labelled file (the owner's own labels are never overwritten)
    prov_doc = json.loads(Path(provenance_path).read_text(encoding="utf-8")) \
        if Path(provenance_path).exists() else {}
    prev_llm = {k: int(v) for k, v in prov_doc.get(split, {}).get("llm", {}).items()}
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing_rows = pooling.read_sheet(out_path)
        changed = pooling.verify_unchanged(existing_rows, original)
        if changed:
            problems += [f"{out_path.name}: {p}" for p in changed[:10]]
        existing = {r["row_id"]: r for r in existing_rows}

    out_rows, hand, llm = [], {}, {}
    pairs: list[tuple[str, int, int]] = []
    for r in original:
        rid = r["row_id"]
        row = dict(r)
        ex = existing.get(rid)
        raw = (ex["label"] if ex else "").strip()
        ex_label = pooling.parse_label(raw) if raw else None
        if raw and ex_label is None:
            problems.append(f"{out_path.name}: row {rid} holds {raw!r}, which is not 0, 1 or 2")
        if ex_label is not None and prev_llm.get(rid) != ex_label:
            row["label"], row["note"] = str(ex_label), (ex["note"] if ex else "")
            hand[rid] = ex_label
            if rid in labels:
                pairs.append((rid, ex_label, labels[rid]))
        elif rid in labels:
            row["label"] = str(labels[rid])
            row["note"] = notes[rid] + (UNSURE_NOTE if rid in unsure else "")
            llm[rid] = labels[rid]
        else:
            row["label"], row["note"] = "", (ex["note"] if ex else "")
        out_rows.append(row)

    unanswered = [r["row_id"] for r in original
                  if r["row_id"] in sendable and r["row_id"] not in llm
                  and r["row_id"] not in hand]
    by_hand_todo = [r["row_id"] for r in original
                    if r["row_id"] in excluded and r["row_id"] not in hand]
    if unanswered and not allow_partial:
        for name, ids in _group_by_batch(unanswered, manifest).items():
            more = "..." if len(ids) > 8 else ""
            problems.append(f"{name}: no reply for {len(ids)} rows ({', '.join(ids[:8])}{more}); "
                            "redo that batch or use --allow-partial")
    problems += [f"internal: {p}" for p in pooling.verify_unchanged(out_rows, original)]
    if problems:
        return {}, problems

    tmp = out_path.with_name(out_path.name + ".tmp")
    pooling.write_sheet(tmp, out_rows)
    os.replace(tmp, out_path)
    agree = agreement([(a, b) for _, a, b in pairs])
    doc = dict(prov_doc)
    doc[split] = {
        "assessor_llm": assessor, "sheet_sha256": manifest["sheet_sha256"],
        "llm": {k: llm[k] for k in sorted(llm, key=int)}, "hand": sorted(hand, key=int),
        "unsure": sorted(unsure, key=int), "excluded_blog": sorted(excluded, key=int),
        "agreement_on_rows_labelled_both_ways": agree}
    atomic_write_text(Path(provenance_path), json.dumps(doc, indent=1, sort_keys=True) + "\n")
    dist = {g: sum(1 for v in llm.values() if v == g) for g in (0, 1, 2)}
    disagreements = [(rid, by_id[rid]["query_id"], a, b) for rid, a, b in pairs if a != b]
    return {"rows": len(original), "llm": len(llm), "hand": len(hand), "unsure": len(unsure),
            "llm_distribution": dist, "unanswered": unanswered, "blog_to_label_by_hand":
            by_hand_todo, "agreement": agree, "disagreements": disagreements,
            "out": str(out_path), "reply_files": [f.name for f in reply_files]}, []


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def print_report(split: str, rep: dict) -> None:
    print(f"{split}: wrote {rep['out']}")
    print(f"  rows {rep['rows']}: chat {rep['llm']}, by hand {rep['hand']}, "
          f"still blank {rep['rows'] - rep['llm'] - rep['hand']}")
    print(f"  chat labels 0/1/2: {rep['llm_distribution']}; marked UNSURE: {rep['unsure']}")
    if rep["unanswered"]:
        print(f"  NOT ANSWERED by the chat (blank): {len(rep['unanswered'])} rows")
    if rep["blog_to_label_by_hand"]:
        print(f"  blog rows left for you to label by hand: {len(rep['blog_to_label_by_hand'])}")
    a = rep["agreement"]
    if a["n"]:
        print(f"  CHAT vs YOUR OWN LABELS on {a['n']} rows both did: agreement "
              f"{a['agree']}/{a['n']} = {fmt(a['agreement'])}, kappa {fmt(a['kappa'])}, linear "
              f"kappa {fmt(a['kappa_linear'])}; table (rows you, columns chat) {a['confusion']}")
        for rid, qid, mine, theirs in rep["disagreements"][:15]:
            print(f"    row {rid} ({qid}): you {mine}, chat {theirs}")
    else:
        print("  no rows were labelled both by you and by the chat, so there is no agreement "
              "figure yet (label about 30 rows yourself first to get one)")
    blank = rep["rows"] - rep["llm"] - rep["hand"]
    print("  next: " + ("fill the blank rows, then " if blank else "")
          + f"python scripts/import_labels.py {split} --round 2 --sheet "
          f"eval/sheets/judge2_{split}_labelled.csv --original eval/sheets/judge2_{split}.csv")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("export", "merge"))
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--exclude-blog", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--assessor", default=DEFAULT_ASSESSOR)
    args = ap.parse_args(argv)

    if args.action == "export":
        sheet = SHEETS / f"judge2_{args.split}.csv"
        if not sheet.exists():
            print(f"{sheet} does not exist: run `python scripts/pool2.py sheets` first")
            return 2
        m = export_split(args.split, SHEETS, args.size, args.exclude_blog)
        sizes = [len(b["row_ids"]) for b in m["batches"].values()]
        print(f"{args.split}: {len(sizes)} batches in eval/sheets/batches/ (rows per batch "
              f"{sizes}); {len(m['excluded_row_ids'])} blog rows excluded; one query is never "
              "split")
        return 0
    rep, problems = merge_split(args.split, SHEETS, PROVENANCE,
                                allow_partial=args.allow_partial, assessor=args.assessor)
    if problems:
        print(f"{len(problems)} problem(s); nothing written:")
        for p in problems[:60]:
            print("  " + p)
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
        return 2 if "missing" in problems[0] and not rep else 1
    print_report(args.split, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
