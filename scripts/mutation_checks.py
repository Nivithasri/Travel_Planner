"""Mutation checks for Phase 3 (and the Phase 2 hold-out regression), D54.

    python scripts/mutation_checks.py

Applies ONE textual mutation at a time to a source file, runs the enrichment/process/facts tests,
prints the failing test names, and restores the file byte-for-byte (sha256 compared before and
after). A mutation that no test catches is reported as SURVIVED and the exit code is 1.
Nothing is committed or left changed: `git diff --stat` must be empty afterwards.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_enrich.py", "tests/test_process.py", "tests/test_facts.py",
         "tests/test_llm_api.py", "tests/test_index.py"]
EVAL_TESTS = ["tests/test_eval_metrics.py", "tests/test_eval_qrels.py", "tests/test_run_eval.py",
              "tests/test_queries.py", "tests/test_pool.py", "tests/test_import_labels.py",
              "tests/test_kappa.py"]

MUTATIONS = [
    ("M1 leak scan disabled (leak_scan returns None)", "src/process/enrich.py",
     'def leak_scan(text: str, holdout: HoldoutFilter) -> str | None:\n'
     '    """Reason string when `text` names a hold-out place (HoldoutFilter or the independent'
     ' regex)."""\n',
     'def leak_scan(text: str, holdout: HoldoutFilter) -> str | None:\n    return None\n'),
    ("M2 fallback disabled (a bad note is kept)", "src/process/enrich.py",
     '    if problem:\n        detail = {"detail"',
     '    if False:\n        detail = {"detail"'),
    ("M15 grounding check disabled (ungrounded names pass)", "src/process/enrich.py",
     '("ungrounded" if ungrounded else None)',
     'None'),
    ("M16 hold-out stems match inside words again (Gitanjali is a leak)", "src/process/enrich.py",
     'r"\\b(?:chikmagal|chikkamagal|chickmagal|araku',
     'r"(?:chikmagal|chikkamagal|chickmagal|araku'),
    ("M3 enriched-vs-frozen check disabled", "src/process/process.py",
     '    e = {c["chunk_id"]: c for c in enriched}\n',
     '    return\n    e = {c["chunk_id"]: c for c in enriched}\n'),
    ("M4 Phase 2 hold-out regression (text_match returns None)", "src/facts/facts.py",
     '        m = self._names.search(text)\n        if m:\n            term = m.group(0)',
     '        return None\n        m = self._names.search(text)\n        if m:\n'
     '            term = m.group(0)'),
    ("M5 note-copy rule disabled", "src/process/enrich.py",
     '    if copy_share(note, chunk_text) > COPY_MAX_SHARE:\n',
     '    if False:\n'),
    ("M6 transport failure turned into a fallback", "src/process/enrich.py",
     "    except JSONValidationError:\n",
     "    except (JSONValidationError, LLMError):\n"),
    ("M7 mentioned_places hold-out filter disabled", "src/process/enrich.py",
     '        if leak_scan(name, holdout):\n            dropped["holdout"] += 1\n'
     '            continue\n',
     ''),
    ("M8 rate limiter pacing disabled (never sleeps)", "src/common/llm.py",
     '            if wait <= 0:\n',
     '            if True:\n'),
    ("M9 scrubbing disabled (keys, headers and URLs pass through)", "src/common/llm.py",
     '    for pattern in _SECRET_PATTERNS:\n        text = pattern.sub("<redacted>", text)\n',
     '    return text\n    for pattern in _SECRET_PATTERNS:\n'
     '        text = pattern.sub("<redacted>", text)\n'),
    ("M10 reasoning text stored with the reply", "src/common/llm.py",
     '    return content\n\n\nclass OpenAICompatClient',
     '    return content + str(message.get("reasoning", ""))\n\n\nclass OpenAICompatClient'),
    ("M11 daily token budget not enforced", "src/common/llm.py",
     '        if not self.trust_provider and self.day_tokens + est_tokens > self.tpd_eff:\n',
     '        if False:\n'),
    ("M12 Retry-After ignored", "src/common/llm.py",
     '            if retry_after is not None:  # the server said how long to wait: honour it\n',
     '            if False:\n'),
    ("M13 checkpoint model/prompt mismatch not refused", "src/process/enrich.py",
     '        if prov.get("model") != model or prov.get("prompt_version") != PROMPT_VERSION:\n',
     '        if False:\n'),
    ("M14 blog chunks sent although --include-blog is off", "src/process/enrich.py",
     '    if chunk["source_type"] == "blog" and not include_blog:\n',
     '    if False:\n'),
    ("M17 contextual BM25 built from the plain text", "src/retrieval/build_index.py",
     r"""if kind == "contextual" else chunk["text"]""",
     r"""if False else chunk["text"]"""),
    ("M18 contextual dense text built from the plain text", "src/retrieval/build_index.py",
     """    if kind == "contextual":\n        return f"{chunk['context_note']}{DENSE""",
     """    if False:\n        return f"{chunk['context_note']}{DENSE"""),
    ("M19 tie order broken (equal scores ranked by reverse position, not chunk_id)",
     "src/retrieval/search.py",
     "key=lambda i: (-float(scores[i]), ids[i]))",
     "key=lambda i: (-float(scores[i]), -i))"),
]


# Phase 4 (D70 to D74): each mutation is run against the evaluation tests (M20 to M29).
EVAL_MUTATIONS = [
    ("M20 nDCG ideal built from the retrieved items only", "eval/metrics.py",
     "    ideal = sorted((int(v) for v in qrel.values()), reverse=True)[:k]\n",
     "    ideal = sorted((grade(qrel, c) for c in top), reverse=True)[:k]\n"),
    ("M21 MRR rank off by one", "eval/metrics.py",
     "            return 1.0 / i\n", "            return 1.0 / (i + 1)\n"),
    ("M22 unjudged chunks counted as relevant", "eval/metrics.py",
     "    return int(qrel.get(chunk_id, 0))\n", "    return int(qrel.get(chunk_id, 1))\n"),
    ("M23 P@5 divided by the judged count instead of 5", "eval/metrics.py",
     "    return sum(1 for c in top if grade(qrel, c) >= threshold) / k\n",
     "    return sum(1 for c in top if grade(qrel, c) >= threshold) / max(\n"
     "        1, sum(1 for c in top if c in qrel))\n"),
    ("M24 qrels loader allows test without the flag", "eval/qrels.py",
     '    if split == "test" and not allow_test:\n', "    if False:\n"),
    ("M25 freshness share counts revised dates", "eval/metrics.py",
     '        if r.get("date_kind") != "published" or not r.get("date"):\n',
     '        if not r.get("date"):\n'),
    ("M26 judging sheet not deduplicated (one row per contributing system)", "eval/pooling.py",
     "        for cid in ids:\n            c = chunks[cid]\n",
     '        for cid in [c for c in ids for _ in pool["pairs"][q["id"]][c]["systems"]]:\n'
     "            c = chunks[cid]\n"),
    ("M27 judging sheet leaks the contributing systems", "eval/pooling.py",
     '"label": "", "note": "", "rubric": RUBRIC_NOTE,',
     '"label": "", "note": ",".join(pool["pairs"][q["id"]][cid]["systems"]),\n'
     '                "rubric": RUBRIC_NOTE,'),
    ("M28 query check ignores aliases", "scripts/check_queries.py",
     "    return [entry.name, *entry.aliases]\n", "    return [entry.name]\n"),
    ("M29 kappa denominator wrong", "scripts/kappa.py",
     "    return (po - pe) / (1 - pe)\n", "    return (po - pe) / 1\n"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests(tests: list[str] = TESTS) -> tuple[str, list[str]]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                        *tests], cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    failed = [ln.split(" ", 1)[1].split(" - ")[0] for ln in lines if ln.startswith("FAILED ")]
    return (lines[-1] if lines else "no output"), failed


ROUTING_TESTS = ["tests/test_airports.py", "tests/test_travel_model.py",
                 "tests/test_travel_time.py", "tests/test_season.py"]

# Phase 5 (D76 to D83): each mutation is run against the routing and season tests (M30 to M41).
ROUTING_MUTATIONS = [
    ("M30 ghat surcharge ignored (hill endpoints cost nothing extra)",
     "src/tools/travel_model.py",
     'hours = plain_hours(km, fitted) + fitted["ghat_surcharge_h"] * n_hill\n    return km',
     'hours = plain_hours(km, fitted) + 0 * n_hill\n    return km'),
    ("M31 pair key not sorted (asymmetric lookup)", "src/tools/travel_model.py",
     'return "|".join(sorted((a, b)))', 'return "|".join((a, b))'),
    ("M32 hold-outs admitted as nodes", "src/tools/travel_model.py",
     "    dest = [e for e in registry.entries if not e.holdout]\n    held",
     "    dest = list(registry.entries)\n    held"),
    ("M33 allowlist not applied (out-of-region airports kept)", "src/tools/routing.py",
     "        if iata not in allow or point is None:\n", "        if point is None:\n"),
    ("M34 plain climate file preferred over the no-elevation file", "src/tools/season.py",
     '    for name in ("archive_2015_2024_noelev.json", "archive_2015_2024.json"):',
     '    for name in ("archive_2015_2024.json", "archive_2015_2024_noelev.json"):'),
    ("M35 rule_only rows counted as model-written", "src/tools/season.py",
     '"rule_only": "rule_only"}', '"rule_only": "model_written"}'),
    ("M36 rainy-day threshold exclusive (> instead of >=)", "src/tools/season.py",
     "+= 1 if pr >= rainy_day_mm else 0", "+= 1 if pr > rainy_day_mm else 0"),
    ("M37 hub distance rule inverted (keeps airports near destinations)",
     "src/tools/travel_model.py",
     'lon) > min_km for lat, lon in points', 'lon) < min_km for lat, lon in points'),
    ("M38 two-band speed ignored (the short-band speed is used for every distance)",
     "src/tools/travel_model.py",
     '    if km <= band:\n        return km / fitted["speed_short_kmh"]\n',
     '    if True:\n        return km / fitted["speed_short_kmh"]\n'),
    ("M39 airport aliases not loaded (nearest_airport codes unknown in travel_time)",
     "src/tools/travel_time.py",
     "            self.names.setdefault(_norm(iata), node_id)\n", "            continue\n"),
    ("M40 partial months counted as full years", "src/tools/season.py",
     "if mm == m and n == calendar.monthrange(y, mm)[1])", "if mm == m)"),
    ("M41 rule-based and fallback warnings listed too", "src/tools/season.py",
     'r["is_warning"] and r["source"] == "model_written"\n', 'r["is_warning"]\n'),
]


def main() -> int:
    survived = 0
    rows = []
    plan = [(m, EVAL_TESTS) for m in EVAL_MUTATIONS]
    routing = [(m, ROUTING_TESTS) for m in ROUTING_MUTATIONS]
    if "--routing" in sys.argv[1:]:  # --routing runs only M30 to M41
        plan = routing
    elif "--eval" in sys.argv[1:]:  # --eval runs only M20 to M29
        pass
    else:
        plan = [(m, TESTS) for m in MUTATIONS] + plan + routing
    for (name, rel, old, new), tests in plan:
        path = ROOT / rel
        original = path.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")
        if text.count(old) != 1:
            print(f"{name}: PATTERN NOT FOUND ({text.count(old)} matches); nothing run")
            survived += 1
            continue
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        try:
            summary, failed = run_tests(tests)
        finally:
            path.write_bytes(original)
        restored = sha(path) == before
        rows.append((name, len(failed), restored))
        print(f"\n{name}\n  {summary}\n  restored byte-identical: {restored}")
        for f in failed:
            print(f"  FAILED {f}")
        if not failed:
            survived += 1
            print("  SURVIVED: no test caught this mutation")
    print("\n| mutation | failing tests | restored |\n|---|---|---|")
    for name, n, ok in rows:
        print(f"| {name} | {n} | {ok} |")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
