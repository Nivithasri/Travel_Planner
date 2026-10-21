"""Loaders for the Phase 4 query set and relevance judgments (D70, D71).

`load_qrels("test")` raises unless `allow_test=True` (O2, CLAUDE.md rule 3): the only caller that
passes it is `run_eval.py --split test`. The dev loader never opens the test file.
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SPLITS = ("dev", "test")
TYPES = ("family", "backpacker", "honeymoon", "pilgrimage", "monsoon", "budget", "vague",
         "freshness")
QUERY_FIELDS = {"id", "text", "split", "type", "time_sensitive", "freshness_slice", "place_scope",
                "places"}
LABELS = (0, 1, 2)


class TestQrelsRefused(Exception):
    """Test qrels were requested without the explicit --split test path."""

    __test__ = False  # not a pytest class


def validate_queries(queries: list[dict]) -> list[str]:
    """Problems with the query records, as readable strings (empty list = valid)."""
    problems = []
    seen = set()
    for q in queries:
        qid = q.get("id", "?")
        if set(q) != QUERY_FIELDS:
            diff = sorted(set(q) ^ QUERY_FIELDS)
            problems.append(f"{qid}: fields {diff} differ from the contract")
            continue
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q["split"] not in SPLITS:
            problems.append(f"{qid}: split {q['split']!r}")
        if q["type"] not in TYPES:
            problems.append(f"{qid}: type {q['type']!r}")
        if q["place_scope"] not in ("named", "free"):
            problems.append(f"{qid}: place_scope {q['place_scope']!r}")
        if (q["place_scope"] == "named") != bool(q["places"]):
            problems.append(f"{qid}: places must be non-empty exactly when place_scope is named")
        for flag in ("time_sensitive", "freshness_slice"):
            if not isinstance(q[flag], bool):
                problems.append(f"{qid}: {flag} must be a bool")
        if q["freshness_slice"] is True and q["time_sensitive"] is not True:
            problems.append(f"{qid}: a freshness-slice query must be time_sensitive")
        if not isinstance(q["text"], str) or not q["text"].strip():
            problems.append(f"{qid}: empty text")
    return problems


def load_queries(split: str | None = None, path: Path | None = None) -> list[dict]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    queries = json.loads((path or EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    problems = validate_queries(queries)
    if problems:
        raise ValueError("invalid queries.json: " + "; ".join(problems))
    return [q for q in queries if split is None or q["split"] == split]


def qrels_path(split: str, directory: Path | None = None) -> Path:
    return (directory or EVAL_DIR) / f"qrels_{split}.json"


def load_qrels(split: str, *, allow_test: bool = False,
               directory: Path | None = None) -> dict[str, dict[str, int]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "test" and not allow_test:
        raise TestQrelsRefused("test qrels are loaded only by `run_eval.py --split test` "
                               "(O2, CLAUDE.md rule 3)")
    path = qrels_path(split, directory)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} does not exist yet: label the judging sheet and run "
                                "scripts/import_labels.py")
    qrels = json.loads(path.read_text(encoding="utf-8"))
    for qid, rels in qrels.items():
        for cid, label in rels.items():
            if label not in LABELS or isinstance(label, bool):
                raise ValueError(f"{path.name}: {qid}/{cid} has label {label!r}")
    return qrels


def qrels_r2_path(split: str, directory: Path | None = None) -> Path:
    return (directory or EVAL_DIR) / f"qrels_{split}_r2.json"


def load_qrels_merged(split: str, *, allow_test: bool = False, directory: Path | None = None
                      ) -> dict[str, dict[str, int]]:
    """Round 1 plus round 2 (D91). Round 2 judges only pairs that round 1 never pooled, and it
    NEVER overrides a round-1 label: a pair in both files keeps its round-1 value. The test split is
    refused exactly as load_qrels refuses it. A missing round-2 file means round 2 is not labelled
    yet, and the result is round 1 alone."""
    merged = load_qrels(split, allow_test=allow_test, directory=directory)
    path = qrels_r2_path(split, directory)
    if not path.exists():
        return merged
    r2 = json.loads(path.read_text(encoding="utf-8"))
    for qid, rels in r2.items():
        for cid, label in rels.items():
            if label not in LABELS or isinstance(label, bool):
                raise ValueError(f"{path.name}: {qid}/{cid} has label {label!r}")
            merged.setdefault(qid, {}).setdefault(cid, label)  # round 1 wins
    return merged
