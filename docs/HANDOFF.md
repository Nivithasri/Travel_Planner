# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is COMPLETE: the corpus is FROZEN (2026-10-19T16:33:44Z) and tagged `phase-2-done`.
`data/processed/FROZEN` exists; the hook blocks Edit/Write on every path under data/processed and `scripts/process.py --write`
refuses. NEVER modify data/processed/* (CLAUDE.md rule 2). Nothing has been pushed since the Step 0 tag fix: `phase-1-done` is on
61f1355 (remote updated by an owner-authorized force push of that one tag); `phase-2-done` is local only. Push only when asked.
Phase 3 (enrichment and indexing) is COMPLETE, confirmed by the owner, tagged `phase-3-done` (local, not pushed). Next: Phase 4 (see "Next phase" below). Read this file, LOG.md, docs/phase-status.md first.

## The frozen corpus (verify with `python -m pytest -q` and `sha256sum data/processed/*`)
| file | sha256 |
|---|---|
| chunks.jsonl (1,696 chunks) | ee11755106409fd0d6ae8656453fc1ec70560593a9c25348572f01ec804a0003 |
| drops.jsonl | 51fc0cd78423533c43a5f482d4d04ffc163e83dad80f2eb35304c2372046b2f9 |
| attribution.json (96 urls) | 0b5b73bd28b3f278aaaa8d532574dc765e194f85cd4ddc3821745d62551ee7bf |
| d33_excluded.json (also data/manifests/d33_excluded.json, committed) | a712d7b6cef25c9c5b0a0d932e45ef12e4088353640cdf60f20a251ee61b3041 |
Counter: bge-m3 tokenizer.json sha256 21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08 (matches Hugging Face, verified by
the owner), tokenizers 0.22.2. Reference date 2026-10-18 from .env. The full record is data/manifests/frozen_manifest.json (committed).
Numbers: blog 283, reference 696, wikivoyage 717; date_kind published 283, revised 1,358, sitemap_lastmod 55; tokens min 52, p50 259,
p95 287, max 300, 0 over 300. Hold-out drops: Araku Valley 8, Thanjavur 7, Thekkady 15, Chikmagalur 0. Freshness share 0.
D23: 17 of 26 places pass at chunk level (18 at raw level; Kodaikanal fails at chunk level only). D33 sidecar: 9 listed places
(Alappuzha, Auroville, Bhadrachalam, Kodaikanal, Munnar, Rameswaram, Srisailam, Varkala, Wayanad) are a Phase 4 QUERY-ELIGIBILITY list
(no dev/test query may target them); chunks.jsonl is unchanged and retrieval is NOT filtered. 17 eligible places, 1,383 chunks, 0 hidden_gem.
Accepted deviations (D48 vii, xvi): 10k-20k chunks unreachable (1,696); 3 source types, one blog host; 0 published chunks in the
18-month window (freshness comes only from the live layer, D32); no Reddit, no official, no YouTube.

## Phase 3 state (2026-10-20: COMPLETE, tag phase-3-done)
GOLD SCORE: there is NONE. `python scripts/label_sample.py --score` prints "no labelled rows ... (30 blank)": both gold_sample.csv (new) and gold_sample_old.csv have 0 labelled rows. The owner confirmed Phase 3 without labelling, so aspect and is_warning accuracy against hand labels is UNMEASURED; the only agreement numbers are LLM-vs-rule-based (not accuracy). With 30 chunks any future score would be small-sample (a 95% interval of roughly +-17 points at 80% accuracy).
What was done (details and commands in LOG.md; decisions D49 to D67): all 1,100 chunks of the 17 D33-eligible places (blog off) were enriched with prompt enrich-v4 on openai/gpt-oss-20b
(hosted free tier, D59): 1,062 model-written, 38 fallbacks (37 `ungrounded`, 1 `note_long`). "Passed its gate" means JSON validity 100%, fallback rate, leak hits 0 and exact call reconciliation; it says nothing about retrieval quality, which is measured in Phase 4/6. Tokens: 836 tokens a chunk (919,711 in all, 1,180 POSTs). The
assembled corpus is data/enriched/chunks.jsonl (1,696 rows, sha256 e4885c84031e871a2765b706dbdd9f4c0780c1b6f8ee1fb98479e9be18185135; gitignored) with data/enriched/provenance.jsonl
(1,087 llm incl. 25 pilot chunks of excluded places, 37 + 1 fallback, 288 rule_only:excluded_place, 283 rule_only:blog_not_sent). `check_enriched_matches_frozen` passes. The indexes are in
index/ (gitignored): bm25_contextual, bm25_plain, dense_plain.npy, dense_contextual.npy (1,696 x 1,024 float32, normalised), ids.json, index_manifest.json; `python scripts/build_index.py
--verify` prints VERIFIED. Committed manifests (hashes, counts, params, versions, weight-file sha256s; no text): data/manifests/enriched_manifest.json and index_manifest.json.
Build time on CPU: dense_plain 2,171 s (0.781 rows/s), dense_contextual 2,344 s (0.724 rows/s), BM25 under 1 s each.
1. OWNER ACTIONS. (a) Confirm that Phase 3 can be tagged (`git tag phase-3-done` is NOT done). (b) Add `EMBED_LOCAL_DIR=models/bge-m3-full` to .env (still missing; until then
   `scripts/query_index.py` and `build_index.py` need `--model-dir models/bge-m3-full`). (c) Remove the duplicate old GENERATOR_MODEL/JUDGE_MODEL lines in .env. (d) Fill
   data/enriched/gold_sample.csv (30 chunks, labels empty; regenerated in the fix round from model-written, non-blog chunks that were never in a pilot or the old sheet, which is kept as
   gold_sample_old.csv) and run `python scripts/label_sample.py --score`; it needs no API call (labels come from the checkpoint). (e) The provider's terms and Valincia's/other accounts' use are the owner's call (D49, D65, D67); O12 still blocks sharing.
2. USE OF THE OUTPUT. Rows with provenance `rule_only` (571) and `fallback` (38) carry RULE-BASED fields (aspect, place_type, season_tags, is_warning) and a template context note, not model
   output; Phase 6 must read provenance.jsonl before trusting those fields (aspect "other" is 44% of the model-written rows; the rule-based aspect agrees with the model on 52.6% of the 1,062 model-written eligible chunks, is_warning on 90.0%; neither is a gold label). The
   25 pilot chunks of excluded places are model-written. The 9 D33-excluded places stay out of every dev/test query (D33, D39).
3. Query (the cache key of query vectors now includes the weight-file hashes from the index manifest): `python scripts/query_index.py "<query>" --model-dir models/bge-m3-full` prints four ranked tables (bm25_contextual, bm25_plain, dense_plain, dense_contextual; ties by chunk_id).
   Three hand-run queries RAN (all four tables printed); that shows the code runs, not that retrieval is good. It is a smoke test, not an evaluation (Phase 4/6).
4. Frozen data: data/processed is read-only; chunks.jsonl is still ee11755106409fd0d6ae8656453fc1ec70560593a9c25348572f01ec804a0003. Enriched chunks and indexes are gitignored (O12).
5. Never print or log the URL, key or headers: both LLM clients scrub every error and debug record (tests). data/enriched/debug_raw.jsonl is UNTRUSTED model output.
6. Not done in Phase 3: the early sample file for Valincia (blocked by O12) and the gold score (sheet unlabelled).

## Next phase: Phase 4 (test collection and evaluation harness; owner Nivithasri)
Start in PLAN MODE and wait for approval. Do not start Phase 5. Inputs: the four indexes (`src/retrieval/search.py` `Index.table`), data/enriched/chunks.jsonl + provenance.jsonl, data/manifests/d33_excluded.json.
First steps:
1. Queries (`eval/queries.json`, contract fields id, text, split, type, time_sensitive, freshness_slice): 30 to 40 (D7/O1; prefer 40 if labelling time allows), split dev/test (about 10 dev, rest test), covering family, backpacker, honeymoon, pilgrimage, monsoon and budget trips. About 8 are the freshness slice (closures, timings, crowding), reported separately because the frozen corpus has no published chunk in the 18-month window (D32). Every query carries `time_sensitive` true/false by hand (reference for Phase 6's classifier). D68: no query names any of the 9 excluded places (Alappuzha, Auroville, Bhadrachalam, Kodaikanal, Munnar, Rameswaram, Srisailam, Varkala, Wayanad) or a hold-out (Chikmagalur, Araku Valley, Thekkady, Thanjavur); write queries before looking at any ranking, and never tune on test.
2. Pooling script: top 10 per query from bm25_contextual, bm25_plain, dense_plain, dense_contextual and a simple fusion; pool over ALL 1,696 chunks (D68), record which round and which system contributed each chunk, report the excluded-place share of every top 10.
3. Judging: 0/1/2 labels by the owner for every pooled chunk (`eval/qrels.json` by contract; per O2 write `qrels_dev.json` and `qrels_test.json` and a loader that REFUSES test qrels unless `--split test` is passed; record the pooling round per chunk; D5: a second pooling round from the final pipeline after Phase 6). Do not inspect test qrels while tuning (CLAUDE.md rule 3).
4. `eval/run_eval.py --retriever <module:function> --split dev|test` for any function following `retrieve(query, filters=None, k=10)`: nDCG@10, MRR, Recall@20, P@5, freshness share (published dates only within 18 months of EVAL_REFERENCE_DATE 2026-10-18; the share will be ~0, say so), diversity (distinct source types and places in the top 10), unjudged@10 per run (D5), results overall, per query type and for the freshness slice; the excluded-place share of the top 10 (D68). Add a thin adapter exposing the four indexes as retrievers so Phase 6 and the owner can run baselines today.
5. D69 hooks for Phase 6 (data only in Phase 4): keep a `provenance` column per pooled chunk (llm vs fallback vs rule_only) so results can be stratified; mark place-named versus place-free queries in `type` or an extra field via a decisions.md entry (contract change); never apply aspect/place_type filters to rule_only/fallback rows unflagged; do not use `mentioned_places` as a signal until Valincia acknowledges D50.
6. Cohen's kappa (O4): Valincia independently labels a shared subset of the judgments; send her the subset without my labels, compute kappa when she returns it. Needs O12 for sharing text, or she judges from the owner's screen/export; decide with the owner. If no second labeller is available, use delayed self-agreement and say so (O4).
7. Tests for the loader refusal, each metric on toy qrels (known answers), and a mutation check per metric; `pytest -q`, `ruff check .`; update LOG.md and docs/phase-status.md; tag `phase-4-done` only when the owner confirms.
Done when: one command prints all metrics for any retrieval function.

## Daily routine for the full enrichment run (Phase 3; the run FINISHED on 2026-10-20, kept as a record of how it was operated)
1. The one command, from the repo root, any time after 00:00 UTC (05:30 India time): `python scripts/enrich.py --full --eligible-only`. It resumes by itself from
   data/enriched/checkpoint_enrich-v4_openai-gpt-oss-20b.jsonl and the disk cache; running it twice does no harm. Never add --include-blog until the data-use terms are
   confirmed. Never edit the checkpoint by hand; *.superseded.jsonl files are history, not results.
2. It stops by itself when the day's token budget (85% of LLM_TPD, kept in data/enriched/quota_state.json by UTC date) is used up: exit code 5 and a line
   "DAILY QUOTA USED UP ... Tokens used today N ... resets in about H h". That is the normal end of a day, not a failure. About 209 chunks a day (812 tokens each).
3. How to read progress (no API call, safe any time): `python scripts/enrich.py --report --full --eligible-only`.
   - "scope: N chunks to process" is the whole job (1,100); "in checkpoint: X of N" is done; N minus X is what remains.
   - "fallback reasons {...}": ungrounded is expected at 2 to 3%; json_invalid, note_copy, note_leak, note_url, note_instruction should be 0 or a handful; anything above about 10% in
     total, or any leak hit, means stop and tell the owner (a prompt change is a decision entry, never a quiet edit).
   - "reconciliation: N chunks = a llm_ok (cache) + b llm_ok (network) + c fallback {...}" must add up exactly.
   - "tokens: ... mean per chunk" should stay near 812; the PROJECTION lines are estimates, not measurements.
   - The limiter numbers (tokens and requests today, http_429) are printed at the end of a run that finishes normally; on a quota stop read tokens used today in the message. http_429 should
     be 0 (the client paces itself; a 429 means the limits in .env are wrong).
4. Exit codes: 0 everything in scope is done; 1 some chunks are not done yet (re-run); 3 five failures in a row (read the scrubbed message, re-run once; if it repeats, stop and tell the owner);
   4 key, model or configuration rejected (stop: do not retry, tell the owner); 5 daily quota used up (wait for the reset, then the same command); 130 Ctrl-C.
5. When the report shows 1,100 of 1,100 done: `python scripts/enrich.py --assemble --eligible-only`, then `python scripts/build_index.py`, `python scripts/build_index.py --verify`,
   `python scripts/query_index.py "<query>"`, `python scripts/enriched_stats.py`. Do not tag phase-3-done until the owner confirms.

PROGRESS LOG (append one line per day; the numbers come from `python scripts/enrich.py --report --full --eligible-only`):
   - Day 1, 2026-10-19 UTC (started 22:46 UTC after the pilots had spent 135,985 tokens): stopped by the budget (exit 5) at 168,898 tokens and 209 requests; 39 new chunks;
     93 of 1,100 in scope done, 1,007 remain; 89 model-written + 4 fallback (all `ungrounded`); leak hits 0; 828 tokens a chunk. Resume at or after 00:00 UTC 2026-10-20 (05:30 India).
   - Day 1 continued (23:40 UTC, owner said the provider limit was not used up; `--tpd-share 0.97` lifts the client's own margin from 85% to 97% of LLM_TPD): +28 chunks, 193,324 tokens, 238 requests, 0 HTTP 429; 121 of 1,100 done, 979 remain, 6 fallbacks (all `ungrounded`). Use `--tpd-share 0.97` only when the owner agrees: the default 85% margin protects against hitting the provider's real limit.
   - Day 2, 2026-10-20 UTC, second account (D65; the owner put Valincia's key into .env; the daily count is now per key): +199 chunks, 169,491 tokens, 205 requests, 0 HTTP 429; 320 of 1,100 done, 780 remain; 11 fallbacks in all, every one `ungrounded` (7 are century ordinals such as 19th). Each account has its own daily budget, so swapping the key in .env and running the same command continues the job; the per-key counter keeps the two spends apart. Valincia's key resets at 00:00 UTC 2026-10-21 (05:30 India); the owner's key already has a fresh budget for 2026-10-20.
   - Day 2 continued with `--tpd-share 1.0` (owner's instruction; the provider table shows 200K TPD for openai/gpt-oss-20b): +36 chunks, the second account's key at 199,349 of 200,000 tokens, 0 HTTP 429; 356 of 1,100 done, 744 remain.    - Day 2, owner's own key, 01:17 to 02:04 UTC: +87 chunks, then the provider's 429 named the limit: openai/gpt-oss-20b tokens per day, 200,000 per ORGANISATION (not per key), "Used 199875" while this key's counter said 76,091, and "try again in 7m32s" for one 1,173-token request (a rolling window). 443 of 1,100 done, 657 remain. Consequence: a second key helps only if it belongs to a different organisation; check the provider's usage page before swapping keys, and expect the client to stop with exit 5 and the API's wording.
   - Other models have their own daily quotas on the provider but are NOT used: a second model would mix two models in one corpus (needs a decision entry and a pilot).

## What exists (code)
- `src/process/process.py` (clean, chunk, hold-out, dedupe, attribution, stats, `check_corpus`, `check_enriched_matches_frozen`;
  `process_place()` is pure and is reused by the live path), `scripts/process.py` (`learn` | dry run | `--write` | `--live <slug>`),
  `scripts/freeze.py` (already executed), `scripts/phase2_evidence.py` (read-only evidence; `all`), `config/boilerplate.json`
  (0 applied lines, 2 rejected), `config/freeze.json` (reddit_resolved true, d33_mode sidecar).
- `HoldoutFilter.text_match` in src/facts/facts.py (hold-out terms incl. Borra Caves, Periyar context rule O8).
- Schema: `date_kind`, `revised_date` (D48). Decisions: docs/decisions.md D48 (i-xix) holds every Phase 2 rule and amendment.
- Phase 1 layout unchanged (src/common, src/collect, src/places, src/facts; scripts/collect.py, coverage_report.py, smoke_ollama.py).
- Phase 3 code: `src/process/enrich.py` (prompt enrich-v3, validation, leak/copy scans, rule fallback, pilot selection, resumable `run_batch`, scope
  and labelled assembly, report and projection), `src/process/label.py` + `scripts/label_sample.py` (gold sheet and scoring), `scripts/enrich.py`,
  `scripts/smoke_llm.py`, `scripts/compare_runs.py`, `scripts/mutation_checks.py` (19 mutations), `OpenAICompatClient`/`RateLimiter`/`scrub_text` in
  `src/common/llm.py`, `load_llm_settings` in `src/common/schemas.py`.
- Index code: `src/retrieval/build_index.py` (BM25 contextual/plain with bm25s, dense plain/contextual with local sentence-transformers, resumable, manifest, verify), `src/retrieval/search.py`
  (exact numpy cosine, ties by chunk_id), `scripts/build_index.py`, `scripts/query_index.py` (four ranked tables).
- Tests: 670 passed (tests/test_process.py, test_enrich.py, test_llm_api.py, test_index.py, test_label_sample.py, test_hooks_scripts.py, ...); `scripts/mutation_checks.py` runs 19 mutations (M1-M19; M17-M19 cover the index code: contextual text built from plain text, tie order).

## How to run
- `pip install -e .[dev]`; Python 3.10.11 (D9); `python -m pytest -q`, `python -m ruff check .` (not on PATH).
- `git config core.hooksPath .githooks` in any fresh clone. No AI attribution anywhere (CLAUDE.md).
- models/bge-m3/tokenizer.json (17 MB) is the owner's local TOKENIZER_DIR; models/ is gitignored. `.gitattributes` pins LF for
  data/manifests/*.json and *.md (the two d33_excluded.json copies must stay byte-identical).

## OPEN OWNER ITEMS
1. O12 for SHARING: what may be shared with the partner, pushed or made public (processing on a private Kaggle is decided, D49).
2. Phase 3: confirm the hosted API's data-use terms and free-plan limits, fix .env (see Phase 3 state), later install bm25s PyStemmer sentence-transformers and download bge-m3 (EMBED_LOCAL_DIR). Reddit credentials only if Reddit is ever added to the live layer (D48 xvi).
3. Wikimedia Terms of Use section 7 wording: not in the repo (no fetched copy); needed for the report and app (Phase 8).
4. Phase 1 items still open: terms quotes and the wiki robots gap (D36/D43), the pre-squash bundle location, D24 review, outreach
   emails unsent, O13 airport allowlist file (Phase 5), attribution in the report and app (Open-Meteo, OSM, Wikimedia).
5. Build-time cleaning rules (D48 amendments a-e) were accepted as a working draft; evidence: `python scripts/phase2_evidence.py rejoin credits`.

## How Valincia gets the chunks (O12 open)
data/processed is gitignored and O12 is undecided, so chunks.jsonl is NOT pushed or uploaded. If the owner shares the file privately
she verifies it against data/manifests/frozen_manifest.json (sha256). Attribution for every url: data/processed/attribution.json.

## Note for Valincia (live layer, D31 proposed, not binding until you acknowledge)
`process_place()` chunks one place's raw records without touching data/processed; `python scripts/process.py --live <slug>` shows the
path (data/live/raw -> data/live/processed). Chunks carry date_kind; freshness counts `published` only. The frozen corpus has no fresh
content (D32). HttpClient, cache and request log are reusable (src/common). The D33 sidecar (data/manifests/d33_excluded.json) is a
query-eligibility list for Phase 4; the planner (Phase 7) should not recommend a listed place without corpus content.

## Known gaps and cautions
- Chunk ids are url + character offset in the cleaned text: they are final now (frozen).
- No topical off-topic filter beyond the wiki section blocklist (content sections such as Demographics are kept).
- Hold-out leakage through landmark names not on the term list is a residual risk (D48 viii); the reviewer's landmark list is unverified.
- Munnar lost 11 Wikivoyage chunks to the Thekkady terms (accepted, D48 xiv).
- Blog dates are the UTC day from article:published_time (the IST <time> can be a day later); the three date fields are one WordPress value.
- Hooks: keep core.hooksPath set. WORKING RULE (owner): a STOP rule means stop BEFORE acting.
