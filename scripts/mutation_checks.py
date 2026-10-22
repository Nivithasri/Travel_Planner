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


RETRIEVAL_TESTS = ["tests/test_fusion.py", "tests/test_significance.py", "tests/test_ablation.py",
                   "tests/test_variants.py", "tests/test_query_understanding.py",
                   "tests/test_filters.py", "tests/test_freshness.py", "tests/test_mmr.py",
                   "tests/test_grader.py", "tests/test_pool2.py", "tests/test_label_batches.py",
                   "tests/test_rescore.py", "tests/test_retrieve.py"]

# Phase 6 (D84 to D94): each mutation is run against the retrieval tests (M42+).
RETRIEVAL_MUTATIONS = [
    ("M42 RRF rank off by one (ranks counted from 0)", "src/retrieval/fusion.py",
     "        for rank, cid in enumerate(ranking, 1):\n",
     "        for rank, cid in enumerate(ranking, 0):\n"),
    ("M43 RRF ties not broken by chunk_id (insertion order kept)", "src/retrieval/fusion.py",
     "    return sorted(score.items(), key=lambda t: (-t[1], t[0]))\n",
     "    return sorted(score.items(), key=lambda t: (-t[1],))\n"),
    ("M44 randomization test one-sided (abs dropped on the exact branch)",
     "eval/significance.py",
     "            if abs(s) / n >= obs - _EPS:\n",
     "            if s / n >= obs - _EPS:\n"),
    ("M45 Holm step-up instead of step-down", "eval/significance.py",
     "        running = max(running, (m - rank) * pvalues[idx])\n",
     "        running = max(running, (rank + 1) * pvalues[idx])\n"),
    ("M46 sub-query hold-out scan disabled", "src/retrieval/query_understanding.py",
     "        if leak_scan(q, holdout):\n", "        if False:\n"),
    ("M47 hallucinated state kept (request need not name it)",
     "src/retrieval/query_understanding.py",
     "        elif code not in place_states and not any(\n",
     "        elif False and not any(\n"),
    ("M48 hallucinated place kept (request need not name it)",
     "src/retrieval/query_understanding.py",
     "        elif entry.name not in named:\n", "        elif False:\n"),
    ("M49 too few sub-queries no longer fall back to the original query",
     "src/retrieval/query_understanding.py",
     "    if len(kept) < MIN_SUBQUERIES:\n", "    if len(kept) < 0:\n"),
    ("M50 month no longer fixes the season", "src/retrieval/query_understanding.py",
     "    if month is not None and season != _MONTH_SEASON[month]:\n",
     "    if False:\n"),
    ("M52 expansion ignores the sub-queries (request only)", "src/retrieval/hybrid.py",
     "                 for q in queries]\n", "                 for q in queries[:1]]\n"),
    ("M53 condensed nDCG keeps the unjudged chunks", "eval/ablation.py",
     '        judged = [c for c in r["ranked"] if c in qrel]\n',
     '        judged = list(r["ranked"])\n'),
    ("M54 rule-based rows lose their place_type exemption (D69)", "src/retrieval/filters.py",
     "        if provenance in RULE_BASED and not rule_based_place_type:\n", "        if False:\n"),
    ("M55 back-fill disabled (rejected candidates dropped)", "src/retrieval/postprocess.py",
     "    out = passing + rejected\n", "    out = passing\n"),
    ("M56 undated chunks treated as old (f = 0, not neutral)", "src/retrieval/freshness.py",
     '    if chunk.get("date_kind") != "published" or not chunk.get("date"):\n'
     "        return NEUTRAL, False\n",
     '    if chunk.get("date_kind") != "published" or not chunk.get("date"):\n'
     "        return 0.0, False\n"),
    ("M57 freshness weight applied to non-time-sensitive queries", "src/retrieval/freshness.py",
     "    w_q = w if time_sensitive else 0.0\n", "    w_q = w\n"),
    ("M58 state filter ignored", "src/retrieval/filters.py",
     '    if f.state and chunk.get("state") not in f.state:\n', "    if False:\n"),
    ("M59 scenery decays like a fast aspect", "src/retrieval/freshness.py",
     '"scenery": None', '"scenery": 180.0'),
    ("M60 a month no longer fixes the season", "src/retrieval/filters.py",
     "        return MONTH_SEASON[self.month] if self.month else self.season\n",
     "        return self.season\n"),
    ("M61 MMR redundancy added instead of subtracted", "src/retrieval/mmr.py",
     "            return (-(lam * rel[cid] - (1 - lam) * redundancy), prior[cid], cid)\n",
     "            return (-(lam * rel[cid] + (1 - lam) * redundancy), prior[cid], cid)\n"),
    ("M62 MMR keeps the place term for place-named queries", "eval/variants.py",
     '    place_term = not parse["places"]', "    place_term = True"),
    ("M63 rerank depth ignored (every candidate reranked)", "src/retrieval/rerank.py",
     "    head, tail = ranked[:depth], ranked[depth:]\n    missing",
     "    head, tail = ranked, []\n    missing"),
    ("M64 lambda selection takes the largest, not the smallest", "src/retrieval/mmr.py",
     "    return min(lam for lam, v in", "    return max(lam for lam, v in"),
    ("M65 rerank cache key without the weight hash", "src/retrieval/rerank.py",
     "        self.model_id = model_id_from(self.hashes)\n",
     '        self.model_id = "bge-reranker-fixed"\n'),
    ("M66 contextual passage drops the model-written note", "src/retrieval/rerank.py",
     '        return chunk["context_note"] + "\\n\\n" + chunk["text"]\n',
     '        return chunk["text"]\n'),
    ("M67 rerank cache lookup skipped (every pair re-scored)", "src/retrieval/rerank.py",
     "            hit = self.cache.get(CACHE_KIND, self.model_id, key, default=None)\n",
     "            hit = None\n"),
    ("M68 table files named after the last row (the shadowed-name bug)", "eval/ablation.py",
     '        (out_dir / f"{stem}_{split}.md")', '        (out_dir / f"{name}_{split}.md")'),
    ("M69 the grader allows a second corrective pass", "src/retrieval/grader.py",
     "MAX_CORRECTIONS = 1  # one", "MAX_CORRECTIONS = 2  # one"),
    ("M70 a rewrite naming a hold-out is kept", "src/retrieval/grader.py",
     "    if leak_scan(text, holdout):\n", "    if False:\n"),
    ("M71 contradiction no longer needs the same place", "src/retrieval/grader.py",
     'by_place.setdefault(c["place"], []).append(c)', 'by_place.setdefault("all", []).append(c)'),
    ("M72 few_sources counts chunks, not distinct URLs", "src/retrieval/grader.py",
     'return len({c["url"] for c in chunks})', "return len(chunks)"),
    ("M73 low_confidence inverted (fires above tau)", "src/retrieval/grader.py",
     "    if confidence < tau:\n", "    if confidence > tau:\n"),
    ("M74 round-2 pool includes pairs round 1 already judged", "eval/pooling2.py",
     "                if cid in judged.get(qid, set()):\n                    continue\n",
     "                if False:\n                    continue\n"),
    ("M75 round-2 labels override round-1 labels", "eval/qrels.py",
     "            merged.setdefault(qid, {}).setdefault(cid, label)  # round 1 wins\n",
     "            merged.setdefault(qid, {})[cid] = label\n"),
    ("M76 a system gap is silently dropped, not reported", "eval/pooling2.py",
     "                gaps.setdefault(system, []).append(qid)\n", "                pass\n"),
    ("M77 round-2 import accepts pairs round 1 already judged", "scripts/import_labels.py",
     '                  if cid in pool1["pairs"].get(qid, {}))\n', "                  if False)\n"),
    ("M78 the test split derives its tau candidates from the test scores",
     "scripts/corrective_rewrites.py",
     '    if split == "dev":\n        return tau_candidates(confidences)\n',
     "    if True:\n        return tau_candidates(confidences)\n"),
    ("M79 a batch may split a query (rows are cut at the size)", "scripts/label_batches.py",
     "        if current and len(current) + len(group) > size:\n",
     "        if len(current) >= size:\n"),
    ("M80 the owner's hand labels are overwritten by the chat", "scripts/label_batches.py",
     "        elif ex_label is not None and prev_llm.get(rid) != ex_label:\n",
     "        elif False:\n"),
    ("M81 any label is accepted in a reply", "scripts/label_batches.py",
     "        label = pooling.parse_label(parts[1])\n",
     "        label = int(parts[1]) if parts[1].strip().isdigit() else None\n"),
    ("M82 a row answered twice is accepted", "scripts/label_batches.py",
     "            elif rid in seen_in:\n", "            elif False:\n"),
    ("M83 a batch also shows the chunk id and URL", "scripts/label_batches.py",
     '                f"TITLE: {one_line(r[\'title\'])}",\n',
     '                f"TITLE: {one_line(r[\'title\'])} {r[\'chunk_id\']} {r[\'url\']}",\n'),
    ("M84 a sheet changed after the export is not noticed", "scripts/label_batches.py",
     '    if manifest["sheet_sha256"] != sha256_file(original_path):\n', "    if False:\n"),
    ("M85 an answered blog row that was excluded is accepted", "scripts/label_batches.py",
     "            elif rid in excluded:\n", "            elif False:\n"),
    ("M86 a hand label can be 'corrected' as if it were the chat's", "scripts/label_batches.py",
     "        elif rid not in chat:\n", "        elif False:\n"),
    ("M87 a recorded correction is treated as one of the owner's own labels",
     "scripts/label_batches.py",
     '        if ex_label is not None and corr is not None and ex_label == corr["to"]:\n',
     "        if False:\n"),
    ("M88 the keep-rule tolerance is ten times too lax", "scripts/rescore_dev.py",
     "KEEP_TOLERANCE = 0.005  # D84", "KEEP_TOLERANCE = 0.05  # D84"),
    ("M89 the corrective pass runs even when no trigger fired", "src/retrieval/retrieve.py",
     "                if triggers:  # exactly one corrective pass (plan.md item 9, D90)\n",
     "                if True:  # exactly one corrective pass (plan.md item 9, D90)\n"),
    ("M90 the corrective pass keeps the first-pass depth", "src/retrieval/retrieve.py",
     "                                                    grader.CORRECTION_DEPTH)\n",
     "                                                    DEPTH)\n"),
    ("M91 explicit filters are silently ignored", "src/retrieval/retrieve.py",
     "            needs_post = bool(filters) or cfg.use_filters or cfg.fresh_w > 0\n",
     "            needs_post = cfg.use_filters or cfg.fresh_w > 0\n"),
    ("M92 explicit filters are accepted with a reranked pipeline", "src/retrieval/retrieve.py",
     "        if filters and cfg.rerank:\n", "        if False:\n"),
    ("M93 the frozen config drifts back to the hybrid", "src/retrieval/retrieve.py",
     'FROZEN_CONFIG = RetrieveConfig(name="dense_contextual", tables=DENSE_CONTEXTUAL)\n',
     "FROZEN_CONFIG = RetrieveConfig()\n"),
    ("M94 the dense-only config also searches BM25 and fuses", "src/retrieval/retrieve.py",
     "            elif cfg.tables == DENSE_CONTEXTUAL:  # one arm: its own cosine scores,",
     "            elif False:  # one arm: its own cosine scores,"),
    ("M95 the dense-only config is accepted together with rerank", "src/retrieval/retrieve.py",
     "                self.expansion or self.use_filters or self.fresh_w or self.rerank):\n",
     "                self.expansion or self.use_filters or self.fresh_w):\n"),
    ("M51 hold-out places listed in the prompt", "src/retrieval/query_understanding.py",
     "            for e in registry.entries if not e.holdout]\n\n\ndef system_prompt",
     "            for e in registry.entries]\n\n\ndef system_prompt"),
    ("M96 filter values are not validated", "src/retrieval/filters.py",
     "    if bad:\n        raise ValueError(f\"unknown {name} value(s)",
     "    if False:\n        raise ValueError(f\"unknown {name} value(s)"),
    ("M97 an explicit empty filter no longer overrides the parsed one", "src/retrieval/filters.py",
     "return cls.from_dict({**base, **(explicit or {})})",
     "return cls.from_dict({**base, **{k: v for k, v in (explicit or {}).items() if v}})"),
    ("M98 the file parser raises a bare KeyError again", "src/retrieval/retrieve.py",
     "            raise QueryNotPrepared(\n                \"this query has no stored parse",
     "            raise KeyError(\n                \"this query has no stored parse"),
]


def main() -> int:
    survived = 0
    rows = []
    plan = [(m, EVAL_TESTS) for m in EVAL_MUTATIONS]
    routing = [(m, ROUTING_TESTS) for m in ROUTING_MUTATIONS]
    retrieval = [(m, RETRIEVAL_TESTS) for m in RETRIEVAL_MUTATIONS]
    if "--routing" in sys.argv[1:]:  # --routing runs only M30 to M41
        plan = routing
    elif "--eval" in sys.argv[1:]:  # --eval runs only M20 to M29
        pass
    elif "--retrieval" in sys.argv[1:]:  # --retrieval runs only M42+
        plan = retrieval
    else:
        plan = [(m, TESTS) for m in MUTATIONS] + plan + routing + retrieval
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
