# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is COMPLETE: the corpus is FROZEN (2026-10-19T16:33:44Z) and tagged `phase-2-done`.
`data/processed/FROZEN` exists; the hook blocks Edit/Write on every path under data/processed and `scripts/process.py --write`
refuses. NEVER modify data/processed/* (CLAUDE.md rule 2). Push only when asked. Which of the older tags (phase-1/2/3-done) exist on origin was NOT
re-checked in Phase 5 (no network); `git tag` lists the local ones.
Phase 3 (enrichment and indexing) is COMPLETE, confirmed by the owner, tagged `phase-3-done`.
Phase 4 (test collection and harness) is CLOSED (2026-10-20, owner-confirmed), tagged `phase-4-done` (main and that tag were pushed to origin on the owner's instruction).
Phase 5 (routing reference data) is CLOSED (2026-10-20, owner-confirmed), tagged `phase-5-done` (lightweight, local, NOT pushed), with PROVISIONAL parameters until the owner spot-checks about 8 calibration routes. Phases 1 to 5 are closed.
Phase 6 (retrieval engine and ablations) is CLOSED (2026-10-22), tagged `phase-6-done` (local, NOT pushed). Phases 1 to 6 are closed.
Next: Phase 7 (owner Valincia). Read this file, LOG.md, docs/phase-status.md first.

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
1. OWNER ACTIONS. (a) DONE: phase-3-done is tagged. (b) EMBED_LOCAL_DIR is in .env now (the pool build used it). (c) Remove the duplicate old GENERATOR_MODEL/JUDGE_MODEL lines in .env. (d) Fill
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

## Phase 4 state (CLOSED; owner: Nivithasri)
Decisions D70 to D75 in docs/decisions.md; contract change in docs/contracts.md (query fields place_scope, places; qrels_dev/qrels_test).
WHAT EXISTS (all committed except the gitignored sheets; 793 tests pass, ruff clean, chunks.jsonl sha256 ee117551..., request log 443):
- eval/queries.json (40; dev 13 / test 27, not 12/28, D70; slice 10; place_scope lexical), scripts/check_queries.py, docs/labelling_rubric.md, docs/labelling_rules.md (owner's rules 1-13, D75).
- eval/pool.json (963 judgments), eval/qrels_dev.json (326) and eval/qrels_test.json (637): ids and ints 0/1/2 only. eval/kappa_subset.json (100). Sheets and labelled files in eval/sheets/ are gitignored (chunk text, O12): judge_{dev,test}.csv, judge_{dev,test}_labelled.csv, kappa_partner.csv, kappa_partner_labelled.csv.
- eval/metrics.py, eval/qrels.py (test refused unless --split test), eval/run_eval.py, eval/adapters.py (bm25_contextual, bm25_plain, dense_plain, dense_contextual, rrf4), eval/pooling.py, scripts/pool.py, scripts/import_labels.py (verifies a labelled sheet against its export; floats only as exactly 0.0/1.0/2.0), scripts/kappa.py, scripts/mutation_checks.py (M1-M29).
RESULTS: LOG.md 2026-10-20 (baselines dev and test, label statistics, kappa 0.825 / 0.848 on 100 judgments from one partner). Reproduce: `python eval/run_eval.py --retriever eval.adapters:dense_contextual --split dev` (needs EMBED_LOCAL_DIR; set HF_HUB_OFFLINE=1). The test split was run once per retriever; Phase 6 must tune on dev only and report test.
CAUTIONS: unjudged@10 is 0 for the five baselines BY CONSTRUCTION (the pool is their own top 10); a Phase 6 variant will retrieve unjudged chunks, so compare it using unjudged@10 and a round-2 pool (D5). Recall@20 is a lower bound. Fresh@10 is 0 for every retriever (D32). 13 dev and 27 test queries are small; no significance tests yet. D69: no aspect/place_type filter or freshness decay on rule_only/fallback rows unflagged; mentioned_places is not a signal (D50). Kappa 0.825 / 0.848 is RUBRIC-ONLY agreement (D75, confirmed): the partner saw only docs/labelling_rubric.md and the tie-break rules were agreed with her later; when the owner wrote the rules relative to labelling the test rows is unknown (rules 12 and 13 cite test rows). Quote it with that caveat.
Nothing is pending for Phase 4.

## Phase 5 state (CLOSED, owner-confirmed, tag phase-5-done; reviewed PASS WITH NOTES, fix round done; owner: Nivithasri)
Decisions D76 to D83 (amended D76, D78, D79; D82, D83 after the review) in docs/decisions.md; contracts in docs/contracts.md ("Routing and season data"). `python scripts/build_routing.py all` rebuilds config/routing_params.json (fitted block), index/travel_matrix.json, index/season_data.json and data/manifests/routing_manifest.json byte-identically (two runs, equal hashes). 868 tests pass, ruff clean, M1-M41 all caught, chunks.jsonl sha256 ee117551..., request log 443.
DATA DEPENDENCY: a REBUILD (and the tests that carry a skipif reason) needs the gitignored data/raw (Wikidata, Overpass, Open-Meteo), data/enriched and, for two Phase 3 tests, data/processed. On a fresh clone `python -m pytest -q` gives 850 passed, 18 skipped, 0 failed; the committed index files and config are enough to use travel_time and check_season.
WHAT EXISTS: config/airports_allowlist.json (O13, 33 codes), config/calibration_routes.json (30 routes, UNVERIFIED), config/routing_params.json, config/routing_sanity_cases.json; src/tools/{routing,travel_model,travel_time,season,check_season}.py; scripts/build_routing.py; tests/test_airports.py, test_travel_model.py, test_travel_time.py, test_season.py; mutations M30-M41 (`python scripts/mutation_checks.py --routing`). index/travel_matrix.json and index/season_data.json are un-ignored and committed (ids and numbers only; a test checks that every string is short and, when the corpus is present, that no chunk text appears); the rest of index/ stays ignored.
MODEL (D77, D78 amended, D82): road km = great-circle km x 1.1497; plain hours = km / 44.49 up to 150 km, then 150 / 44.49 + (km - 150) / 71.51; hours = plain hours + 1.199 per hill endpoint. 48 nodes = 26 destinations + 22 hubs (airports); hill nodes Ooty, Kodaikanal, Munnar (elevation) and Kodagu, Wayanad, Srisailam (override). 0 triangle violations (by construction, see LOG).
USE: `from src.tools.travel_time import travel_time` (hours; destinations, hubs, and airport codes COK HYD IXM MAA MYQ PNY TIR WGC which stand for the nearest destination); `from src.tools.check_season import check_season` (26 destinations only; month 1-12 or name; returns verdict, reasons, access_risk, climate, evidence counts by provenance, model-written warning ids, confidence, nearest_airport, caveats). Unknown names, hold-outs, and hubs in check_season raise UnknownPlaceError (a ValueError).
CAUTIONS (also LOG.md 2026-10-20): (1) all 30 calibration references were written from memory; the fit is PROVISIONAL; errors are agreement with those values (fit MAPE hours 11.4%, held-back 13.5%, none held-back over 25%), not accuracy. (2) road_km on hill edges is underestimated by 20 to 26% (the surcharge is in hours only). (3) Sanity 17 of 19: Chennai May and Hampi May came out `fair` (monthly mean Tmax 35.1 and 37.1 C against 38 C); thresholds unchanged, a change is a separate decision made after seeing all misses (D80); two travel cases pass at the edge of 3.5 h. (4) Verdicts are climate-only; Munnar has 2 model-written chunks of 37, Wayanad 1 of 59, Kodaikanal 3 of 41, so their tag evidence is rule-based (D69); Munnar lists no warnings (only rule-based ones exist) and check_season says so. (5) Hubs are airports, not rail junctions; minor strips stay (RMD, NVY, PUT, KJB, GBI, CDP, RQY); the military rule that gave exactly 22 was chosen after the 24-hub result (D76 amended). (6) Kodagu and Wayanad are district centroids (D21); PNY resolves to Auroville. (7) ERA5 is a 0.25-degree reanalysis (Open-Meteo CC BY 4.0, non-commercial) and OSM airports are ODbL: attribution is in both index files' meta; O12 share-alike is open.
OWNER ACTIONS: spot-check about 8 calibration routes (fill `source` in config/calibration_routes.json, then `python scripts/build_routing.py all`, which re-fits and rebuilds); a threshold change or a days-with-Tmax-at-least-38 count is a separate decision (D80 amended); tests/test_llm_api.py (Phase 3) got two skipif guards in the Phase 5 fix round.

## Phase 6 state (CLOSED 2026-10-22, tag phase-6-done is local and was not moved by the review fix round; owner: Valincia)
Decisions D84 to D96 (D95 and D96 amended) in docs/decisions.md; contracts in docs/contracts.md ("retrieve()", eval files). Tests with the .venv interpreter: see LOG.md 2026-10-22 (review fix round) for the counts; ruff clean; mutations M1-M95 in LOG.md.
FROZEN retrieve() = dense_contextual (D96 amended): dense bge-m3 cosine, depth 100, over context_note + text; no BM25, LLM, reranker or parse. eval/frozen_config.json carries it; a test fails if it differs from FROZEN_CONFIG in src/retrieval/retrieve.py. HYBRID_CONFIG (RRF of BM25 + dense) and FULL_CONFIG (expansion -> rerank 30 -> MMR 0.9 -> one corrective pass at tau 0.6314) are measured, NOT frozen; every ablated component lives in src/retrieval/ behind RetrieveConfig. `python scripts/verify_retrieve.py [--split test]` shows the engine reproduces the ablation rows' top 10 in order.
WHAT EXISTS: src/retrieval/{fusion,hybrid,query_understanding,filters,freshness,postprocess,rerank,mmr,grader,retrieve}.py; eval/{variants,ablation,significance,pooling2}.py; eval/parsed_queries.json (40 queries, qu-v1), eval/corrective_rewrites.json, corrective_tau.json, mmr_lambda.json, pool2.json, qrels_dev_r2.json, qrels_test_r2.json (load_qrels_merged), labelling_r2_provenance.json; eval/ablation/dev_merged.* and test_final.{md,json}, test_merged_<row>.json, test_run_manifest.json; scripts/{test_run,rescore_dev,verify_retrieve,pool2,label_batches,parse_queries,corrective_rewrites,tau_curve,mmr_curve,rerank_check,...}.py.
TEST SPLIT RUN ONCE (commit 0dd35c9; scripts/test_run.py refuses a second run while eval/ablation/test_run_manifest.json exists; use .venv/Scripts/python.exe, the base python lacks bm25s). nDCG@10 test (dev merged): bm25_plain 0.436 (0.524), dense_plain 0.590 (0.673), dense_contextual FROZEN 0.685 (0.793), hybrid_contextual 0.711 (0.768), expanded_hybrid 0.713 (0.717), row7_rerank 0.819 (0.771), row8_mmr 0.791 (0.772), row9_crag 0.803 (0.778). Holm over 11 pre-registered comparisons: contextual notes significant (dense +0.096, p_holm 0.033; hybrid +0.102, 0.008); dense > BM25 on contextual (+0.148, 0.032); rerank over expansion +0.106 (0.004); full chain over frozen +0.118 (0.006); hybrid vs dense_contextual +0.026 (1.0), expansion, MMR, corrective not significant. time_sensitive flag on test: 7 TP, 0 FP, 1 FN (Q25), 19 TN. Command and full table: LOG.md 2026-10-22, eval/ablation/test_final.md.
INTERPRETER: use `.venv/Scripts/python.exe` for pytest, the eval scripts and anything that builds or loads the indexes (bm25s and the retrieval stack are installed there). The base `python` lacks bm25s: `python -m pytest -q` gives 20 failures in tests/test_index.py and scripts/test_run.py crashed on it once. ruff works with either.
PHASE 7 MUST KNOW (review fix round, D97): (a) FULL_CONFIG works on the 40 eval queries ONLY. parser_from_file and rewriter_from_file raise `QueryNotPrepared` (not KeyError) for any other query, and no live parser/rewriter is wired. Do NOT enable FULL_CONFIG, expansion, MMR or the corrective pass in Phase 7 until a cached live parser (src/retrieval/query_understanding.py) and rewriter (src/retrieval/grader.py) are injected. (b) The frozen retrieve() is unfiltered and returns chunks of the 9 D33-excluded places (about 10% of the top 10; they are mostly rule-based, D69): the planner must not recommend them on that basis. (c) Choosing the full chain now is test-informed (D97); the final table labels it "best observed on test, selected after seeing test". (d) Explicit filters are validated (month 1-12, state codes, PlaceType values); [] / None clears a parsed filter. (e) The corrective pass fires on 77% of dev and 85% of test queries and the contradiction trigger on none of 40 (D97).
CAUTIONS: (1) On test the rerank rows beat the frozen row clearly although the dev keep rule dropped rerank on the hybrid base; the frozen choice was fixed before the run and was NOT changed. Phase 7 may enable FULL_CONFIG or the rerank knowingly (cost: 2 LLM calls for the full chain, 30 cross-encoder pairs per query, about 0.4 s a pair on CPU) and must state which and why. (2) Round-2 labels (dev 162, test 273) were written by a chat assistant (D95), 17 dev and 32 test rows marked UNSURE, 4 dev and 13 test labels corrected on the owner's instruction, dev rows 23 and 55 and test row 114 left as given; NO agreement figure between that chat and the owner exists (a spot-check of about 30 rows would give it). Blog text (inditales.com) was sent to the chat service; its terms were not read; O12 open. (3) 13 dev / 27 test queries: only about 0.1 nDCG is detectable. (4) Freshness share is 0 on every row (D32); the freshness row is mechanism-only on rule-based aspects (D87). (5) Popular/hidden mix not evaluable (D44). (6) The query-understanding LLM sees query text only; quota state lives in .cache/llm_quota (D93).
OWNER ACTIONS: remove duplicate GENERATOR_MODEL/JUDGE_MODEL lines in .env; optional human spot-check of ~30 round-2 rows; confirm local tags phase-2-done / phase-3-done exist (`git tag`).

## Next phase: Phase 7 (place cards and agent pipeline, live layer; owner Valincia)
Start in PLAN MODE and wait for approval. Read docs/plan.md Phase 7 and Section 2.1, D31 and D35 (live layer, YouTube Data API, still proposed), O7 (untrusted text; test with an injection chunk), D44 (popularity tercile), and the Phase 6 cautions above. Use `from src.retrieval.retrieve import retrieve, retrieve_with_trace`; live chunks are never added to the corpus, indexes or qrels. Routing: `src.tools.travel_time`, `src.tools.check_season` (see the Phase 5 section: provisional parameters, hours not km). Query understanding already exists (src/retrieval/query_understanding.py, prompt qu-v1, validated in code); reuse it in the query node.

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
  `scripts/smoke_llm.py`, `scripts/compare_runs.py`, `scripts/mutation_checks.py` (41 mutations, M1-M41), `OpenAICompatClient`/`RateLimiter`/`scrub_text` in
  `src/common/llm.py`, `load_llm_settings` in `src/common/schemas.py`.
- Index code: `src/retrieval/build_index.py` (BM25 contextual/plain with bm25s, dense plain/contextual with local sentence-transformers, resumable, manifest, verify), `src/retrieval/search.py`
  (exact numpy cosine, ties by chunk_id), `scripts/build_index.py`, `scripts/query_index.py` (four ranked tables).
- Tests: 868 passed after the Phase 5 fix round (850 passed and 18 skipped on a fresh clone; 670 at the end of Phase 3; tests/test_process.py, test_enrich.py, test_llm_api.py, test_index.py, test_label_sample.py, test_hooks_scripts.py, ...); `scripts/mutation_checks.py` runs 41 mutations (M1-M41: M1-M19 enrichment, LLM client and index code, M20-M29 evaluation, M30-M41 routing and season).

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
   emails unsent, attribution in the report and app (Open-Meteo, OSM, Wikimedia).
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
