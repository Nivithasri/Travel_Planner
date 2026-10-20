"""Check eval/queries.json against the Phase 4 rules (D68, D70).

    python scripts/check_queries.py

Fails (exit 1) when a query names a D33-excluded place or a hold-out (registry names and aliases,
HoldoutFilter, the independent D63 stems), when a "named" query does not contain the name or an
alias of each place it lists, when a "free" query contains any registry name or alias, or when a
listed place is unknown, excluded or a hold-out. place_scope is lexical: it says whether the query
TEXT names a registry place (D70). Also prints the split, slice and flag counts.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.qrels import load_queries, validate_queries  # noqa: E402
from src.facts.facts import HoldoutFilter  # noqa: E402
from src.places.places import Registry  # noqa: E402
from src.process.enrich import leak_scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "registry" / "core_places.json"
EXCLUDED = ROOT / "data" / "manifests" / "d33_excluded.json"


def _word_re(terms: list[str]) -> re.Pattern:
    alts = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in alts) + r")(?!\w)",
                      re.IGNORECASE)


def terms_of(entry) -> list[str]:
    return [entry.name, *entry.aliases]


def check(queries: list[dict], registry: Registry, excluded: set[str]) -> list[str]:
    problems = validate_queries(queries)
    holdout = HoldoutFilter.from_registry(registry)
    by_name = {e.name: e for e in registry.entries}
    excluded_re = _word_re([t for n in excluded if n in by_name for t in terms_of(by_name[n])])
    for q in queries:
        qid, text = q["id"], q["text"]
        m = excluded_re.search(text)
        if m:
            problems.append(f"{qid}: names a D33-excluded place ({m.group(0)})")
        hit = leak_scan(text, holdout)
        if hit:
            problems.append(f"{qid}: names a hold-out ({hit})")
        for name in q.get("places", []):
            entry = by_name.get(name)
            if entry is None:
                problems.append(f"{qid}: unknown place {name!r}")
            elif name in excluded or entry.holdout:
                problems.append(f"{qid}: lists an excluded or hold-out place {name!r}")
            elif not _word_re(terms_of(entry)).search(text):
                problems.append(f"{qid}: place {name!r} does not appear in the text")
        if q.get("place_scope") == "free":
            others = [t for e in registry.entries for t in terms_of(e) if len(t) > 3]
            m = _word_re(others).search(text)
            if m:
                problems.append(f"{qid}: place_scope free but the text contains {m.group(0)!r}")
    return problems


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    queries = load_queries()
    registry = Registry.load(REGISTRY)
    excluded = set(json.loads(EXCLUDED.read_text(encoding="utf-8"))["excluded"])
    problems = check(queries, registry, excluded)
    split = Counter(q["split"] for q in queries)
    print(f"queries {len(queries)}: dev {split['dev']}, test {split['test']}; "
          f"freshness slice {sum(q['freshness_slice'] for q in queries)} "
          f"(dev {sum(q['freshness_slice'] and q['split'] == 'dev' for q in queries)}); "
          f"time_sensitive {sum(q['time_sensitive'] for q in queries)}")
    print("types: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(q['type'] for q in queries)
                                                               .items())))
    print("place_scope: " + ", ".join(f"{k} {v}" for k, v in sorted(
        Counter(q['place_scope'] for q in queries).items())))
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        return 1
    print(f"OK: no query names one of the {len(excluded)} excluded places or a hold-out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
