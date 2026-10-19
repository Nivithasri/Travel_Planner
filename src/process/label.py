"""Hand-labelled gold sample for aspect and is_warning (D58). Pure functions; the CLI is
scripts/label_sample.py.

The sheet shows the chunk text only (no model label, so the labeller is not anchored). The owner
fills `aspect_gold` and `is_warning_gold`; `score` then compares the LLM labels and the
rule-based labels with them. Nothing in this module writes a label.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from src.common.schemas import Aspect
from src.process.enrich import rule_aspect, rule_is_warning

GOLD_COLUMNS = ["chunk_id", "place", "source_type", "text", "aspect_gold", "is_warning_gold"]
ASPECTS = sorted(a.value for a in Aspect)
_TRUE, _FALSE = {"true", "1", "yes", "y"}, {"false", "0", "no", "n"}


def _key(seed: int, chunk_id: str) -> str:
    return hashlib.sha256(f"gold:{seed}:{chunk_id}".encode()).hexdigest()


def select_gold(chunks: list[dict], exclude_ids: Iterable[str], short_ids: Iterable[str], seed: int,
                *, per_type: int = 10, n_tails: int = 6) -> list[str]:
    """Seeded sample, `per_type` chunks per source type, `n_tails` of them under-200-token tails
    (spread evenly over the types), none from `exclude_ids`. Places are taken round-robin inside a
    type so one place cannot dominate. The result is in a seeded mixed order, not grouped by
    type."""
    banned, shorts = set(exclude_ids), set(short_ids)
    types = sorted({c["source_type"] for c in chunks})
    tails_each = n_tails // len(types)
    picked: list[str] = []
    for t in types:
        pool = sorted((c for c in chunks if c["source_type"] == t and c["chunk_id"] not in banned),
                      key=lambda c: _key(seed, c["chunk_id"]))
        chosen: list[str] = []
        for want_tail, n in ((True, tails_each), (False, per_type - tails_each)):
            by_place: dict[str, list[str]] = {}
            for c in pool:
                if (c["chunk_id"] in shorts) == want_tail and c["chunk_id"] not in chosen:
                    by_place.setdefault(c["place"], []).append(c["chunk_id"])
            got = 0
            while got < n and any(by_place.values()):
                for place in sorted(by_place):
                    if got < n and by_place[place]:
                        chosen.append(by_place[place].pop(0))
                        got += 1
        picked += chosen
    return sorted(picked, key=lambda cid: _key(seed, cid))


def gold_rows(chunks: list[dict], ids: list[str]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in chunks}
    return [{"chunk_id": i, "place": by_id[i]["place"], "source_type": by_id[i]["source_type"],
             "text": by_id[i]["text"], "aspect_gold": "", "is_warning_gold": ""} for i in ids]


def write_gold_csv(path: Path, rows: list[dict], *, force: bool = False) -> None:
    """Refuses to overwrite an existing sheet: it may hold the owner's labels."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists and may hold labels; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # BOM: Excel reads it as UTF-8
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_gold_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


class LabelError(ValueError):
    pass


def parse_gold(rows: list[dict]) -> tuple[list[dict], int]:
    """(labelled rows with parsed `aspect` and `is_warning`, number of rows with no label at all).
    A half-filled row or an unknown label raises LabelError naming the chunk."""
    out, blank = [], 0
    for r in rows:
        a = (r.get("aspect_gold") or "").strip().lower()
        w = (r.get("is_warning_gold") or "").strip().lower()
        if not a and not w:
            blank += 1
            continue
        if a not in ASPECTS:
            raise LabelError(f"{r['chunk_id']}: aspect_gold {a!r} is not one of {ASPECTS}")
        if w in _TRUE:
            warn = True
        elif w in _FALSE:
            warn = False
        else:
            raise LabelError(f"{r['chunk_id']}: is_warning_gold {w!r} must be true/false")
        out.append({"chunk_id": r["chunk_id"], "text": r["text"], "aspect": a, "is_warning": warn})
    return out, blank


def _matrix(pairs: list[tuple[str, str]], labels: list[str]) -> list[list[int]]:
    idx = {x: i for i, x in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for gold, pred in pairs:
        m[idx[gold]][idx[pred]] += 1
    return m


def _system(pairs_a: list[tuple[str, str]], pairs_w: list[tuple[bool, bool]]) -> dict:
    n = len(pairs_a)
    classes = {}
    for c in ASPECTS:
        gold_n = sum(1 for g, _ in pairs_a if g == c)
        pred_n = sum(1 for _, p in pairs_a if p == c)
        hit = sum(1 for g, p in pairs_a if g == c == p)
        classes[c] = {"gold": gold_n, "predicted": pred_n, "correct": hit}
    wl = ["false", "true"]
    return {
        "n": n,
        "aspect_accuracy": sum(g == p for g, p in pairs_a) / n if n else None,
        "aspect_classes": classes,
        "aspect_confusion": _matrix(pairs_a, ASPECTS),
        "is_warning_accuracy": sum(g == p for g, p in pairs_w) / n if n else None,
        "is_warning_counts": dict(Counter((str(g).lower(), str(p).lower()) for g, p in pairs_w)),
        "is_warning_confusion": _matrix([(str(g).lower(), str(p).lower()) for g, p in pairs_w], wl),
    }


def score(gold: list[dict], llm_by_id: dict[str, dict]) -> dict:
    """Compare LLM labels (`llm_by_id`: chunk_id -> fields) and the rule-based labels with `gold`.
    Chunks the LLM has not labelled yet are left out of the LLM score only."""
    have = [g for g in gold if g["chunk_id"] in llm_by_id]
    llm = _system([(g["aspect"], llm_by_id[g["chunk_id"]]["aspect"]) for g in have],
                  [(g["is_warning"], llm_by_id[g["chunk_id"]]["is_warning"]) for g in have])
    rule = _system([(g["aspect"], rule_aspect(g["text"]).value) for g in gold],
                   [(g["is_warning"], rule_is_warning(g["text"])) for g in gold])
    return {"labelled": len(gold), "llm_missing": len(gold) - len(have), "llm": llm, "rule": rule}


def format_score(res: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def table(m: list[list[int]], labels: list[str]) -> list[str]:
        w = max(len(x) for x in [*labels, "gold|pred"]) + 1
        return (["gold|pred".ljust(w) + "".join(x[:8].rjust(9) for x in labels)]
                + [lab.ljust(w) + "".join(str(v).rjust(9) for v in row)
                   for lab, row in zip(labels, m, strict=True)])

    lines = [f"labelled rows: {res['labelled']}; LLM labels missing for {res['llm_missing']} "
             "(left out of the LLM score only)"]
    for name, title in (("llm", "LLM"), ("rule", "RULE-BASED (the fallback)")):
        s = res[name]
        lines += ["", f"== {title} vs gold, n={s['n']}",
                  f"aspect accuracy {pct(s['aspect_accuracy'])}; "
                  f"is_warning accuracy {pct(s['is_warning_accuracy'])}",
                  "per class (gold support / predicted / correct):"]
        lines += [f"  {c:<8} {v['gold']:>3} / {v['predicted']:>3} / {v['correct']:>3}"
                  for c, v in s["aspect_classes"].items()]
        lines += ["aspect confusion (rows = gold):", *table(s["aspect_confusion"], ASPECTS),
                  "is_warning confusion (rows = gold):",
                  *table(s["is_warning_confusion"], ["false", "true"])]
    return "\n".join(lines)
