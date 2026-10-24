"""Mutation checks for Phase 3 (and the Phase 2 hold-out regression), D54.

    python scripts/mutation_checks.py

Applies ONE textual mutation at a time to a source file, runs the enrichment/process/facts tests,
prints the failing test names, and restores the file byte-for-byte (sha256 compared before and
after). A mutation that no test catches is reported as SURVIVED and the exit code is 1.
Nothing is committed or left changed: `git diff --stat` must be empty afterwards.
"""
# ruff: noqa: E501  (the mutation tables hold exact source lines that must not be wrapped)
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
                   "tests/test_rescore.py", "tests/test_retrieve.py",
                   "tests/test_sensitivity_binary.py", "tests/test_spotcheck_r2.py"]

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
    ("M99 the sensitivity collapse keeps grade 2", "scripts/sensitivity_binary.py",
     "c: min(int(g), 1) for c, g in rels.items()", "c: int(g) for c, g in rels.items()"),
    ("M100 chat share counts the pairs that were NOT judged in round two",
     "scripts/sensitivity_binary.py",
     "sum(1 for q, c in slots if c in r2.get(q, {}))",
     "sum(1 for q, c in slots if c not in r2.get(q, {}))"),
    ("M101 binary spot-check agreement ignores owner 2 / chat 2 cells",
     "scripts/sensitivity_binary.py",
     "both1 = sum(table[i][j] for i in (1, 2) for j in (1, 2))",
     "both1 = sum(table[i][j] for i in (1, 2) for j in (1,))"),
    ("M102 the spot-check sampler ignores the minimum blog count", "scripts/spotcheck_r2.py",
     "    rotate([c for c in order if c[\"source_type\"] == \"blog\"], min_blog)\n",
     "    rotate([c for c in order if c[\"source_type\"] == \"blog\"], 0)\n"),
    ("M98 the file parser raises a bare KeyError again", "src/retrieval/retrieve.py",
     "            raise QueryNotPrepared(\n                \"this query has no stored parse",
     "            raise KeyError(\n                \"this query has no stored parse"),
]


# Phase 7 (D99 to D104): mutations M103+ run against the Phase 7 tests only (--phase7).
PHASE7_TESTS = ["tests/test_state.py", "tests/test_place_cards.py",
                "tests/test_spotcheck_cards.py", "tests/test_planner.py",
                "tests/test_writer.py", "tests/test_stream.py", "tests/test_critic.py",
                "tests/test_live.py", "tests/test_guards.py", "tests/test_graph.py",
                "tests/test_cli.py"]
PHASE7_MUTATIONS = [
    ("M103 the live gate ignores terms_status", "src/live/config.py",
     "        if self.terms_status != \"owner_accepted_risk\":\n            return \"terms_not_verified\"\n",
     "        if False:\n            return \"terms_not_verified\"\n"),
    ("M104 a rule-based row is shown as a warning", "src/tools/place_cards.py",
     "if p == \"llm\" and c.get(\"is_warning\")]", "if c.get(\"is_warning\")]"),
    ("M105 blog chunks enter the cards by default", "src/tools/place_cards.py",
     "            if c[\"source_type\"] == \"blog\" and not include_blog:\n",
     "            if False:\n"),
    ("M106 a full place uses rule-based rows", "src/tools/place_cards.py",
     "            usable = prov == \"llm\" or (limited and prov in (\"rule_only\", "
     "\"fallback\"))\n",
     "            usable = True\n"),
    ("M107 hold-out text is not dropped from the cards", "src/tools/place_cards.py",
     "            if holdout is not None and holdout.text_match(c[\"text\"]):\n",
     "            if False:\n"),
    ("M109 the planner ignores the daily travel limit", "src/agents/planner.py",
     "    return 0 if h <= 0 else math.ceil(h / limit - 1e-9) - 1\n", "    return 0\n"),
    ("M110 the planner uses unnamed excluded places", "src/agents/planner.py",
     "pool = {c.name: c for c in candidates if c.named or not c.limited}",
     "pool = {c.name: c for c in candidates}"),
    ("M111 avoid-hill-roads is not applied", "src/agents/planner.py",
     "    if request.avoid_hill_roads:\n", "    if False:\n"),
    ("M112 poor-season places are not left out", "src/agents/planner.py",
     "    if poor and len(pool) - len(poor) >= target:", "    if False:"),
    ("M113 the budget cap ignores nothing: tolerance dropped", "src/agents/budget.py",
     "cap_inr * (1 + cfg[\"tolerance\"])", "cap_inr"),
    ("M114 blog chunks reach the writer", "src/agents/evidence.py",
     "            if item.source_type == \"blog\" and not w[\"allow_blog\"]:\n",
     "            if False:\n"),
    ("M115 injection-pattern corpus text is not dropped", "src/agents/evidence.py",
     "            flags = injection_flags(item.text)\n            if flags:\n",
     "            flags = []\n            if flags:\n"),
    ("M116 URLs stay in packed evidence", "src/agents/evidence.py",
     "    t = URL.sub(\"[link removed]\", t)\n", "    pass\n"),
    ("M117 a finished stream is not cached", "src/common/llm.py",
     "        self.cache.set(\"llm_chat\", model, inputs, text)\n", "        pass\n"),
    ("M118 live ids are not marked in the expanded text", "src/agents/writer.py",
     "        return f\"[{cid}]\" if cid.startswith(\"live:\") else f\"[chunk:{cid}]\"\n",
     "        return f\"[chunk:{cid}]\"\n"),
    ("M119 the writer does not enforce the reservation budget", "src/agents/writer.py",
     "> w[\"reservation_max_tokens\"]:", "> 10**9:"),
    ("M120 an uncited bullet is not flagged", "src/agents/critic.py",
     "            if not tags:\n                f.append(",
     "            if False:\n                f.append("),
    ("M121 recency words on corpus claims are allowed", "src/agents/critic.py",
     "            if RECENCY.search(b) and not any(t.startswith(\"L\") for t in tags):\n",
     "            if False:\n"),
    ("M122 hold-out and excluded mentions are allowed", "src/agents/critic.py",
     "        if name.casefold() not in allowed and _mentions(full, name):\n",
     "        if False:\n"),
    ("M123 the LLM repair pass never runs", "src/agents/critic.py",
     "    if findings and days and repair is not None:\n", "    if False:\n"),
    ("M124 the cannot-confirm line is not added", "src/agents/critic.py",
     "    if not request.time_sensitive or live_status == \"ok\" or STATUS_LINE in draft_text:\n",
     "    if True:\n"),
    ("M125 season conflicts are not checked", "src/agents/critic.py",
     "        if s.season_verdict == \"poor\" or s.access_risk == \"high\":\n",
     "        if False:\n"),
    ("M126 a live query may carry any text", "src/live/search.py",
     "        if not is_safe_query(query, self.vocab):\n", "        if False:\n"),
    ("M127 unknown words pass the query boundary", "src/live/queries.py",
     "    return all(t in vocab or (t.isdigit() and len(t) == 4) for t in toks)\n",
     "    return True\n"),
    ("M128 places are not filtered through the registry when building queries",
     "src/live/queries.py",
     "        canon = known.get(str(p).casefold())\n", "        canon = known.get(str(p).casefold(), str(p))\n"),
    ("M129 the same-day cache is written when it is switched off", "src/live/store.py",
     "    def put(self, query: str, results: list[dict]) -> None:\n        if not self.enabled:\n",
     "    def put(self, query: str, results: list[dict]) -> None:\n        if False:\n"),
    ("M130 the cache keeps other days", "src/live/store.py",
     "            if not f.name.startswith(self._today() + \"_\"):\n", "            if False:\n"),
    ("M131 the credit stop is ignored", "src/live/store.py",
     "        return self.credits + credits <= self.stop_at\n", "        return True\n"),
    ("M132 the terms gate is skipped for real calls", "src/live/search.py",
     "        if gate != \"ok\":\n            return [], gate, False\n", "        if False:\n            pass\n"),
    ("M133 out-of-scope live results are kept", "src/live/search.py",
     "        if not r.get(\"url\") or not any(", "        if not r.get(\"url\") or False and not any("),
    ("M134 post_json follows the allowlist no more (body logged)", "src/common/http.py",
     "        if resp.status in REDIRECT_STATUS:\n            raise FetchError(f\"redirect not followed",
     "        if False:\n            raise FetchError(f\"redirect not followed"),
    ("M135 hold-out names in the request are not warned about", "src/agents/nodes.py",
     "    if held:\n        warnings.append(", "    if False:\n        warnings.append("),
    ("M136 unnamed excluded places become eligible", "src/agents/nodes.py",
     "    eligible = {n for n, c in deps.cards.items() if c[\"evidence_class\"] == \"full\"} | named\n",
     "    eligible = set(deps.cards) | named\n"),
    ("M137 node warnings replace instead of accumulate", "src/agents/nodes.py",
     "            state.warnings.extend(v)\n", "            state.warnings = list(v)\n"),
    ("M138 an unknown start city is kept", "src/agents/nodes.py",
     "                setattr(req, field_name, None)\n", "                pass\n"),
    ("M139 the freshness note is left out of the answer", "src/agents/nodes.py",
     "DOC_NOTES[\"freshness\"], DOC_NOTES[\"budget\"]", "DOC_NOTES[\"budget\"]"),
    ("M140 node timings are not recorded", "src/agents/graph.py",
     "        upd[\"timings\"] = {**upd.get(\"timings\", {}), name: round(deps.clock() - t0, 3)}\n",
     "        upd[\"timings\"] = {}\n"),
    ("M141 the run log stores the request text", "scripts/run_pipeline.py",
     "\"request_sha256\": hashlib.sha256(text.encode()", "\"request_text\": text, \"request_sha256\": hashlib.sha256(text.encode()"),
    ("M142 a registry alias is not in the live scope (Mysore pages dropped)", "src/live/search.py",
     "        scope = [*self.place_names, *self.alias_map, *STATE_NAMES.values()]\n",
     "        scope = [*self.place_names, *STATE_NAMES.values()]\n"),
    ("M143 a registry alias does not map to its place", "src/live/search.py",
     "        if not place:  # a registry alias (for example Mysore for Mysuru) names the place too\n",
     "        if False:  # a registry alias (for example Mysore for Mysuru) names the place too\n"),
    ("M144 the terms notice is left out of a live note", "src/live/search.py",
     "            res.note = f\"{res.note}; {TERMS_NOTICE}\" if res.note else TERMS_NOTICE\n",
     "            pass\n"),
    ("M145 the live result cache is switched on by default", "config/live.json",
     "\"cache_enabled\": false,", "\"cache_enabled\": true,"),
    ("M108 the card spot-check accepts an edited fixed column", "scripts/spotcheck_cards.py",
     "    if fixed_sha(sorted(sheet_rows, key=lambda r: int(r[\"row_id\"]))) != "
     "key[\"fixed_sha256\"]:\n",
     "    if False:\n"),
]


# Phase 8 (D110, D111): mutations M146+ run against the app tests only (--phase8).
PHASE8_TESTS = ["tests/test_app_logic.py", "tests/test_app_page.py"]
PHASE8_MUTATIONS = [
    ("M146 one Deps object is shared by live-on and live-off runs", "app/state.py",
     "    k = bool(use_live)\n", "    k = True\n"),
    ("M147 a live view may be written to the disk cache", "app/query_cache.py",
     "    if use_live:\n        return False\n    if view.get(\"live_status\") not in DISK_LIVE_STATUSES:\n"
     "        return False\n    return not any(s.get(\"live\") for s in (view.get(\"final\") or {})"
     ".get(\"sources\", []))\n",
     "    return True\n"),
    ("M148 the terms notice is shown although the live step did not run", "app/state.py",
     "    return live_ran(view)\n", "    return True\n"),
    ("M149 the page redactor returns text unchanged", "app/state.py",
     "        text = str(text)\n", "        return str(text)\n"),
    ("M150 a source row gets an excerpt field", "app/state.py",
     "\"Licence\": licence, \"Flags\": \", \".join(flags)})\n",
     "\"Licence\": licence, \"Flags\": \", \".join(flags),\n"
     "                     \"Excerpt\": s.get(\"text\", \"\")})\n"),
    ("M151 the form's free text is added to the places passed on", "app/state.py",
     "               \"named_places\": places}\n", "               \"named_places\": places + interests}\n"),
    ("M152 the run lock is not released by the worker", "app/state.py",
     "            _RUN_LOCK.release()\n            handle.done.set()\n", "            handle.done.set()\n"),
    ("M153 the start and end city are not validated", "app/state.py",
     "        msg = check_city(value, resolve_city)\n", "        msg = None\n"),
    ("M154 an unknown free-text place is accepted as a place name", "app/state.py",
     "    if len(found) != 1:\n        return \"unknown\", None\n",
     "    if len(found) != 1:\n        return \"ok\", text\n"),
    ("M155 the cache key leaves out use_live", "app/query_cache.py",
     "\"use_live\": bool(use_live)}", "\"use_live\": False}"),
    ("M156 the default runner never falls back to run_local", "app/state.py",
     "    except ImportError:\n        from src.agents.graph import run_local\n",
     "    except ZeroDivisionError:\n        from src.agents.graph import run_local\n"),
    ("M157 the default runner is always run_local", "app/state.py",
     "    from src.agents.graph import run_graph\n\n    return run_graph\n",
     "    from src.agents.graph import run_local\n\n    return run_local\n"),
    ("M158 the live status line counts every merged item as cited", "app/state.py",
     "    cited = sum(1 for s in (state.final or {}).get(\"sources\", []) if s.get(\"live\"))\n",
     "    cited = merged\n"),
    ("M159 the live status line is shown although live was not requested", "app/state.py",
     "    if not view.get(\"use_live\"):\n        return None\n    status = view.get(\"live_status\")\n",
     "    status = view.get(\"live_status\")\n"),
    ("M160 the switched-off note is not reworded when live was not requested", "app/state.py",
     "    if view.get(\"use_live\") or view.get(\"live_status\") not in (\"disabled\", \"not_run\"):\n"
     "        return view\n",
     "    if True:\n        return view\n"),
    ("M161 the progress banner stays after the run", "app/streamlit_app.py",
     "    banner.empty()\n", "    pass\n"),
    ("M162 the chunk id column drops the chunk: marker", "app/state.py",
     "\"Chunk id\": key,", "\"Chunk id\": cid,"),
    ("M163 the Streamlit file watcher is switched on again", ".streamlit/config.toml",
     "fileWatcherType = \"none\"\n", ""),
    ("M164 the live trace is not reset between runs", "app/state.py",
     "        trace.reset()\n        res = run(*args, **kwargs)\n", "        res = run(*args, **kwargs)\n"),
]


# Phase 8 round 3 (D115): mutations of the src changes; they run the app tests and the Phase 7 tests.
PHASE8B_TESTS = [*PHASE8_TESTS, *PHASE7_TESTS, "tests/test_purge_cache.py"]
PHASE8B_MUTATIONS = [
    ("M165 the stream is decoded with the charset requests assumed", "src/common/llm.py",
     "        resp.encoding = \"utf-8\"\n",
     ""),
    ("M166 chat_stream stores the text although cache is False", "src/common/llm.py",
     "        if cache:\n            self.cache.set(\"llm_chat\", model, inputs, text)\n",
     "        if True:\n            self.cache.set(\"llm_chat\", model, inputs, text)\n"),
    ("M167 chat_stream reads the cache although cache is False", "src/common/llm.py",
     "hit = self.cache.get(\"llm_chat\", model, inputs) if cache else None",
     "hit = self.cache.get(\"llm_chat\", model, inputs)"),
    ("M168 chat goes through the cache although cache is False", "src/common/llm.py",
     "        if not cache:\n            return self._request(payload)\n",
     "        if False:\n            return self._request(payload)\n"),
    ("M169 the writer caches a prompt that holds live text", "src/agents/writer.py",
     "    extra = {\"cache\": False} if holds_live_text(messages) else {}\n",
     "    extra = {}\n"),
    ("M170 the repair call caches a prompt that holds live text", "src/agents/critic.py",
     "        extra = {\"cache\": False} if holds_live_text(msgs) else {}\n",
     "        extra = {}\n"),
    ("M171 inferred states are used as a restriction again", "src/agents/state.py",
     "        states = [s for s in inferred if s in named]\n",
     "        states = inferred\n"),
    ("M172 explicit states are ignored", "src/agents/state.py",
     "    if o.get(\"states\"):\n        states = list(o[\"states\"])\n",
     "    if False:\n        states = list(o[\"states\"])\n"),
    ("M173 the state note leaves out the named places that were kept", "src/agents/planner.py",
     "            if outside:\n",
     "            if False:\n"),
    ("M174 the live node says switched off when live was not requested", "src/agents/nodes.py",
     "LIVE_NOTES[\"not_requested\"]",
     "LIVE_NOTES[\"disabled\"]"),
    ("M175 the purge script misses the high C1 characters", "scripts/purge_cache.py",
     "C1 = re.compile(\"[\\u0080-\\u009f]\")",
     "C1 = re.compile(\"[\\u0080-\\u008f]\")"),
    ("M176 the purge script deletes without --execute", "scripts/purge_cache.py",
     "            if args.execute:\n                p.unlink(missing_ok=True)\n",
     "            if True:\n                p.unlink(missing_ok=True)\n"),
    ("M177 interests do not rank the evidence", "src/agents/evidence.py",
     "    matched = 0 if interest_match(item, interests) else 1\n",
     "    matched = 1\n"),
    ("M178 rule-based rows can match an interest", "src/agents/evidence.py",
     "    if item.provenance != \"llm\":\n        return False\n",
     "    if False:\n        return False\n"),
    ("M179 the interest match ranks below the aspect priority", "src/agents/evidence.py",
     "(0 if is_warn else 1, matched, PRIORITY.get(key, 5), -item.score, item.chunk_id)",
     "(0 if is_warn else 1, PRIORITY.get(key, 5), matched, -item.score, item.chunk_id)"),
    ("M180 the temples interest maps to the wrong place type", "src/agents/evidence.py",
     "\"temples\": {\"place_types\": (\"temple\",)",
     "\"temples\": {\"place_types\": (\"beach\",)"),
    ("M181 the writer rule prefers the S source", "src/agents/writer.py",
     "prefer an L source over an S source",
     "prefer an S source over an L source"),
    ("M182 no uncited live item is listed", "src/agents/nodes.py",
     "        if p.item.is_live and p.alias not in cited]\n",
     "        if False]\n"),
    ("M183 cited live items are listed as uncited too", "src/agents/nodes.py",
     "        if p.item.is_live and p.alias not in cited]\n",
     "        if p.item.is_live]\n"),
    ("M184 the packer is not given the interests", "src/agents/nodes.py",
     "interests=req.interests, named_places=req.named_places,",
     "interests=(), named_places=req.named_places,"),
    ("M185 the page drops the uncited live rows", "app/state.py",
     "    entries += [(s, True) for s in final.get(\"uncited_live\", [])]\n",
     "    entries += []\n"),
    ("M186 a replayed write is reported as a live call", "app/state.py",
     "    if int(tokens.get(\"write\") or 0) > 0:\n",
     "    if True:\n"),
    ("M187 the purge accepts a directory that is not a .cache", "scripts/purge_cache.py",
     "    if \".cache\" not in p.parts:\n",
     "    if False:\n"),
    ("M188 the purge accepts data/ and eval/ inside a .cache path", "scripts/purge_cache.py",
     "        if p == root or root in p.parents:\n",
     "        if False:\n"),
    ("M189 named places do not count as in the plan", "src/agents/evidence.py",
     "in_plan = {s.place for s in plan.stops} | set(named_places)",
     "in_plan = {s.place for s in plan.stops}"),
    ("M190 live items with no resolved place are kept", "src/agents/evidence.py",
     "        if not item.place:\n",
     "        if False:\n"),
    ("M191 live items about other places are kept", "src/agents/evidence.py",
     "        if item.place not in in_plan:\n",
     "        if False:\n"),
    ("M192 the named places are not given to the packer", "src/agents/nodes.py",
     "interests=req.interests, named_places=req.named_places,",
     "interests=req.interests, named_places=(),"),
    ("M193 the in-plan count ignores the items that were left out", "app/state.py",
     "\"in_plan\": max(t.in_scope - left_out, 0)",
     "\"in_plan\": t.in_scope"),
    ("M194 the not-cited count is wrong", "app/state.py",
     "\"not_cited\": max(merged - cited, 0)",
     "\"not_cited\": cited"),
    ("M195 the answer does not say that live results were left out", "src/agents/nodes.py",
     "    if n_live_out:\n",
     "    if False:\n"),
    ("M196 citation markers are not numbered by first appearance", "app/state.py",
     "    for m in CITATION.finditer(str(final.get(\"text\", \"\"))):\n",
     "    for m in ():\n"),
    ("M197 an unknown citation id is removed from the shown text", "app/state.py",
     "        if s is None:\n            return m.group(0)\n",
     "        if s is None:\n            return \"\"\n"),
    ("M198 showing the itinerary rewrites the stored text", "app/state.py",
     "    return CITATION.sub(replace, text), aliases\n",
     "    final[\"text\"] = CITATION.sub(replace, text)\n    return final[\"text\"], aliases\n"),
    ("M199 the table alias differs from the marker in the text", "app/state.py",
     "\"Alias\": aliases.get(key, s.get(\"alias\", \"\")),",
     "\"Alias\": s.get(\"alias\", \"\"),"),
    ("M200 the short marker is not linked to the source", "app/state.py",
     "        return f\"[{marker}]({link})\" if link else marker\n",
     "        return marker\n"),
    ("M201 the out-of-scope check is not applied to live items", "src/agents/evidence.py",
     "        if scope_check is not None and scope_check(item.text):  # D120\n",
     "        if False:  # D120\n"),
    ("M202 the dropped-for-excluded count reads the wrong reason", "app/state.py",
     "if d.get(\"reason\") == \"live_names_excluded\")",
     "if d.get(\"reason\") == \"live_unresolved\")"),
    ("M203 the answer does not say that results naming an out-of-scope place were left out", "src/agents/nodes.py",
     "    if n_live_scope:\n",
     "    if False:\n"),
    ("M204 hold-out names do not block a live item", "src/agents/scope.py",
     "        return leak_scan(text, holdout) is not None\n",
     "        return False\n"),
    ("M205 an excluded place name matches inside a longer word", "src/agents/scope.py",
     "(?<!\\w)(?:",
     "(?:"),
    ("M206 a named or planned excluded place is still blocked", "src/agents/scope.py",
     "        elif e.name in excluded and e.name not in allowed:\n",
     "        elif e.name in excluded:\n"),
    ("M207 the write node does not pass the out-of-scope check to the packer", "src/agents/nodes.py",
     "scope_check=out_of_scope_matcher(deps.registry_entries, deps.excluded, allowed))",
     "scope_check=None)"),
]
# D122: the critic strips bullets that name a place outside the plan and R5 carries the day.
PHASE8_TESTS_CRITIC = ["tests/test_critic.py"]
CRITIC_MUTATIONS = [
    ("M208 the R5 mention finding carries no day", "src/agents/critic.py",
     "            f.append(CriticFinding(rule=\"R5\", day=day, message=(\n"
     "                f\"{name} is outside",
     "            f.append(CriticFinding(rule=\"R5\", message=(\n"
     "                f\"{name} is outside"),
    ("M209 the deterministic fix keeps a bullet that names a blocked place", "src/agents/critic.py",
     "            if b and any(_mentions(line, n) for n in blocked):\n",
     "            if False:\n"),
    ("M210 the deterministic fix keeps a backup that names a blocked place", "src/agents/critic.py",
     "    kept = [b for b in d.backups if not any(_mentions(b, n) for n in blocked)]\n",
     "    kept = list(d.backups)\n"),
    ("M211 a named or planned place counts as blocked", "src/agents/critic.py",
     "if n.casefold() not in allowed]\n",
     "]\n"),
    ("M212 the cycle does not pass the blocked names to the deterministic fix",
     "src/agents/critic.py",
     "ctx.packed, tuple(blocked_names(ctx)))", "ctx.packed)"),
]


def restore(path, original, tries=8):
    """Write the original bytes back, retrying a transient OS error (seen once on Windows)."""
    import time

    for attempt in range(tries):
        try:
            path.write_bytes(original)
            return
        except OSError:
            if attempt == tries - 1:
                print(f"COULD NOT RESTORE {path}: run git checkout -- that file", flush=True)
                raise
            time.sleep(0.5 * (attempt + 1))


def main() -> int:
    survived = 0
    rows = []
    plan = [(m, EVAL_TESTS) for m in EVAL_MUTATIONS]
    routing = [(m, ROUTING_TESTS) for m in ROUTING_MUTATIONS]
    retrieval = [(m, RETRIEVAL_TESTS) for m in RETRIEVAL_MUTATIONS]
    phase7 = [(m, PHASE7_TESTS) for m in PHASE7_MUTATIONS]
    phase8 = [(m, PHASE8_TESTS) for m in PHASE8_MUTATIONS]
    phase8 += [(m, PHASE8B_TESTS) for m in PHASE8B_MUTATIONS]
    phase8 += [(m, PHASE8_TESTS_CRITIC) for m in CRITIC_MUTATIONS]
    if "--critic" in sys.argv[1:]:  # --critic runs only M208 to M212 (D122)
        plan = [(m, PHASE8_TESTS_CRITIC) for m in CRITIC_MUTATIONS]
    elif "--phase8" in sys.argv[1:]:  # --phase8 runs only M146 to M212
        plan = phase8
    elif "--phase7" in sys.argv[1:]:  # --phase7 runs only M103+
        plan = phase7
    elif "--routing" in sys.argv[1:]:  # --routing runs only M30 to M41
        plan = routing
    elif "--eval" in sys.argv[1:]:  # --eval runs only M20 to M29
        pass
    elif "--retrieval" in sys.argv[1:]:  # --retrieval runs only M42+
        plan = retrieval
    else:
        plan = [(m, TESTS) for m in MUTATIONS] + plan + routing + retrieval + phase7 + phase8
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
            restore(path, original)
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
