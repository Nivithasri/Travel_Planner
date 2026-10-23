# Phase status
Tick only after verification commands were run (see LOG.md).
Phases 1 to 7 are CLOSED. Phase 8 is BUILT and awaits the owner's hand-run and /phase-review 8 (not tagged). Phase 5 is PROVISIONAL until the owner spot-checks about 8 calibration routes. Next: Phase 9 (Valincia).

## Phase 0: Scaffold and shared infrastructure
- [x] pyproject.toml, src/common/{config,cache,llm,schemas}.py (verified: `python -m pytest -q` 128 passed, `python -m ruff check .` clean)
- [x] Hooks verified firing in-session (attribution, frozen, lint); see LOG.md
- [x] First commit has no attribution (see `git log --format='%an <%ae>%n%B'`)
- [ ] Not verified against a live Ollama server (all LLM tests use mocked HTTP)

## Phase 1: Data collection and place registry (reviewed 2026-10-19, PASS WITH NOTES; tag phase-1-done)
- Verified by /phase-review 1 (commands in LOG.md, 2026-10-19): pytest 452 passed (415 after the D46 dead-code cleanup), ruff clean, manifest 179 lines reconciled, request log reconciled, D23 18 of 26
- [x] Core place registry: data/registry/core_places.json, 30 places (26 collectable + 4 hold-outs), all contract fields
- [x] Polite cached HTTP client (robots, delays, retries, allowlist D43, circuit breaker D38), place resolver (Wikidata)
- [x] Place store + ensure_place, Wikivoyage, Wikipedia and web collectors, raw manifest, coverage_report.py
- [x] Facts layer (Wikidata 26, Open-Meteo 29 files, Overpass airports 1), D36 to D43 recorded
- [x] Hand-off for the live layer (HttpClient, cache, request log reusable; note for Valincia in HANDOFF)
- [ ] Reddit collector: DEFERRED (D45). Code exists and is tested offline; no credentials, 0 records, live behaviour and access rules not confirmed. Run before the Phase 2 freeze if credentials arrive, else accepted at the freeze
- [ ] YouTube transcripts: DROPPED (D27, D30), accepted deviation; official pages: 0 records (D45)
- [x] Done when: every core destination has raw data from at least two source types (26 of 26; 18 of 26 under the stricter D23, a recorded deviation from the plan's 20), a never-seen place (Yercaud) is collected with one command, and a second call makes zero network requests (tests/test_collect.py)
- Open owner items are listed in docs/HANDOFF.md (terms quotes, wiki robots gap, O12, Reddit credentials). Pushed: 8ed3f86 "Finalize Phase 1" with tag phase-1-done; D47 simplification commit on top

## Phase 2: Cleaning and chunking (FROZEN 2026-10-19; tag phase-2-done)
- Verified (commands in LOG.md, 2026-10-19 Phase 2 entries): pytest 436 passed, ruff clean, two `--write` runs byte-identical, 7 mutations (a-g) + Phase 1 hold-out mutation caught, request log unchanged (443); `freeze.py` 8 of 8 preconditions passed, then executed
- [x] Boilerplate removal (fixed rules + reviewed rule file config/boilerplate.json), exact and near-duplicate removal (D48)
- [x] South India scope: every record's place checked against the registry (collectable, not hold-out)
- [x] Dates normalized; date_kind published | revised | sitemap_lastmod | none with BaseChunk invariants (D48, O9)
- [x] Chunks 200-300 tokens (hard max incl. overlap, bge-m3 tokenizer), source_type/url/date/state/place; deterministic ids (1,696 chunks)
- [x] Hold-out drop (url/title record level, text chunk level), counted per hold-out and source; scans 0 hits
- [x] Corpus statistics: data/manifests/corpus_stats.{md,json}; processed_manifest.json; frozen_manifest.json
- [x] D23/D33: 17 of 26 pass at chunk level; D33 applied as a sidecar query-eligibility list (data/manifests/d33_excluded.json), nothing deleted
- [x] Freeze: data/processed/FROZEN written 2026-10-19T16:33:44Z (chunks.jsonl sha256 ee117551...); tag phase-2-done (local, not pushed)
- [x] Done when: the corpus is frozen. From this point it never changes, so every metric stays reproducible.
- Deviations (accepted, D48 vii/xvi): 1,696 chunks vs the plan's 10k-20k; 3 source types, one blog host; 0 published chunks in the 18-month window; no Reddit/official/YouTube

## Phase 3: Enrichment and indexing (COMPLETE 2026-10-20, owner-confirmed, tag phase-3-done; route changed by D59: hosted API, local CPU embeddings, no Kaggle)
- [x] Enrichment code, validation, leak/copy scans, rule fallback, resumable batch, pilot selection, gold sheet (19 mutations M1-M19, each caught by at least one test)
- [x] Hosted OpenAI-compatible client with client-side rate limiter, usage accounting and scrubbing (offline tests; later used for the full run, see below)
- [x] smoke_llm.py 3/3 (owner); pilot enrich-v4 (79 chunks) passed its gate (JSON validity, fallback rate, leaks, reconciliation; not retrieval quality); index code committed and dry-run on a scratch file (BM25 x2, dense x2, query_index.py, build_index.py --verify)
- [x] --full --eligible-only finished 2026-10-20: 1,100 of 1,100 (1,062 model-written, 38 fallbacks); `--assemble` wrote data/enriched/chunks.jsonl (1,696 rows; check_enriched_matches_frozen passes)
- [x] real BM25 x2 + dense x2 built on CPU (76 min), `build_index.py --verify` VERIFIED, query_index.py ran on 3 queries (smoke test only), manifests committed (data/manifests/enriched_manifest.json, index_manifest.json)
- [ ] Early sample for Valincia (blocked by O12); gold labels and score: NOT DONE (both sheets unlabelled, `--score` has nothing to score)
- [x] Tag phase-3-done (local, not pushed)
- [x] Done when: a script takes a query and returns ranked chunks from each index (query_index.py, four tables).

## Phase 4: Test collection and evaluation harness (CLOSED 2026-10-20, owner-confirmed, tag phase-4-done)
- [x] 40 queries (dev 13, test 27, slice 10, D70), exclusion check, rubric + owner's tie-break rules (D75)
- [x] Round-1 pool 963 judgments, all labelled by the owner; eval/qrels_dev.json (326) and eval/qrels_test.json (637) imported and committed (ids and ints only)
- [x] eval/run_eval.py with five baseline adapters: dev and one test run per retriever recorded in LOG.md (nDCG@10 test: bm25_contextual 0.559, bm25_plain 0.453, dense_plain 0.614, dense_contextual 0.708, rrf4 0.665)
- [x] Kappa on 100 partner judgments: 0.825 unweighted, 0.848 linear-weighted, agreement 0.900 (one partner; RUBRIC-ONLY agreement: she did not see the tie-break rules, D75)
- [x] Mutations M1-M29 all caught; 793 tests pass; ruff clean
- [x] Owner confirmed D75 (partner saw only the rubric; the rules were agreed with her later; when the owner wrote them relative to labelling the test rows is unknown) and the phase; tag phase-4-done
- [ ] Not done: significance tests (Phase 6), round-2 pooling (D5, after Phase 6)
- [x] Done when: one command prints all metrics for any retrieval function (`python eval/run_eval.py --retriever <module:function> --split dev|test`).

## Phase 5: Routing reference data (CLOSED 2026-10-20, owner-confirmed, tag phase-5-done; review PASS WITH NOTES, fix round done; parameters PROVISIONAL until the route spot-check)
- [x] Decisions D76-D83 (amended D76, D78, D79; D82, D83 after the review); 48 nodes = 26 destinations + 22 hubs (hold-outs never nodes)
- [x] O13 airport allowlist as a data file with tests; dedupe rule; military airports excluded (D79); nearest_airport re-derived and every returned code resolves in travel_time (D83)
- [x] Travel matrix: index/travel_matrix.json (1,128 pairs), src/tools/travel_time.py; model = road factor + two-band speed + fixed ghat surcharge per hill endpoint (D78 amended, D82); hill flag on 6 nodes; 0 triangle violations (was 230, by construction)
- [x] Season data: index/season_data.json (26 places x 12 months), src/tools/check_season.py with evidence counts by provenance, model-written warnings only, confidence
- [x] Verified (commands in LOG.md): pytest 868 passed, ruff clean, M1-M41 all caught, two rebuilds identical, fresh clone 850 passed / 18 skipped / 0 failed, chunks.jsonl sha256 ee117551..., request log 443
- [ ] Calibration reference routes are UNVERIFIED (from memory): fit is PROVISIONAL until the owner spot-checks about 8 routes; errors are agreement with those values, not accuracy; road km on hill edges is 20-26% short
- [x] Sanity table 17 of 19: two heat misses (Chennai May, Hampi May) kept as measured, recorded in D80 amended (days with Tmax >= 38 is a later option, not built)
- [x] Tag phase-5-done (lightweight, local, not pushed); owner confirmed the phase
- [ ] popularity_tercile (D44): NOT built; Phase 7 reads D44 and decides
- [ ] travel_km is unreliable on hill legs (20-26% short): documented in contracts.md, use hours
- [x] Done when: both functions return instantly for every registry place (tests time 26 places x 12 months and 1,000 lookups).

## Phase 6: Retrieval engine and ablations (CLOSED 2026-10-22, tag phase-6-done; owner: Valincia)
- Verified (commands in LOG.md): pytest 1132 passed / 11 skipped, ruff clean, mutations M42-M95 caught, verify_retrieve.py dev 13/13 and test 27/27
- [x] Baselines, RRF hybrid, contextual vs plain with the D69 control, query understanding (40 parsed queries), filters with D69 exemptions, aspect-aware freshness, cross-encoder rerank, MMR, CRAG-style corrective pass (D84-D90)
- [x] Round-2 pool (435 pairs) judged by a chat assistant (D95, no agreement figure vs the owner); merged qrels
- [x] Significance: sign-flip test, bootstrap CI, Holm (D92); test split run ONCE (eval/ablation/test_final.md)
- [x] retrieve() FROZEN as dense_contextual (D96 amended before the test run); hybrid and full chain kept behind RetrieveConfig, not frozen
- [x] Test: frozen 0.685 nDCG@10; rerank rows 0.80-0.82 (reported, frozen choice not changed); contextual notes and rerank significant after Holm, hybrid, expansion, MMR, corrective not
- [ ] Caveats: chat-made round-2 labels unvalidated; freshness 0 (D32); 27 test queries; popular/hidden mix not evaluable (D44)
- [x] Done when: the ablation table is complete and `retrieve()` is frozen (it returns ranked dense_contextual results; filters, rerank and freshness exist as measured, non-frozen options)

## Phase 7: Place cards and agent pipeline (CLOSED 2026-10-23: live layer fixture-tested only)
- [x] Place cards (26, extractive, model-written warnings only), blind spot-check sheet produced (owner to fill and score)
- [x] Deterministic planner (hours, daily limit, hubs never stops, excluded places only when named, relative less-visited mix), rough budget (ASSUMED figures)
- [x] Streaming writer under the TPM budget (reservation 2.5k to 3.1k of 5k), evidence packer with injection defence, rule critic R1 to R7 with one repair pass
- [x] LangGraph graph with a local fallback, CLI, per-node timing logs
- [x] Five real end-to-end requests with the live layer OFF: 3 passed, 1 repaired, 1 failed_rules (Kerala in July: one evaluated place, poor season; reported not hidden)
- [ ] Live layer: built behind the terms gate, NOT exercised (terms not verified; provider shape assumed)
- [x] Card spot-check scored (D106): about_place 0.982, valid 0.579
- [x] /phase-review 7 done (D107); tag phase-7-done created on the owner's instruction
- [ ] Done when: a query returns a cited itinerary end to end with tips, warnings, a rough budget and backups: met with live OFF; "using live evidence when the live step succeeds" is NOT yet shown.

## Phase 8: App (BUILT 2026-10-23, D110 and D111; awaiting the owner's hand-run and /phase-review 8; NOT tagged)
- [x] One Streamlit page (app/streamlit_app.py): form, streamed itinerary, sources panel with links (live rows "live, fetched <date>"), warnings, live-status notice, collapsible timing table, query cache (corpus-only JSON on disk, live-on in session memory only), one-run-at-a-time lock, redaction, localhost config
- [x] Verified (commands in LOG.md): pytest 1369 passed / 11 skipped (91 new), ruff clean, mutations M146-M176 all caught (full suite 175 of 176; M32 only skips without data/raw) (full suite: 154 of 155 caught, M32 survives only because data/raw is absent here)
- [x] Corpus path needs no internet beyond the hosted model; live search is optional, off by default, and the page works from the corpus without it (headless tests with a fake pipeline)
- [ ] Not verified: no real browser session, no real pipeline run, LLM call or live search through the page (the owner's hand-run)
- [ ] Hand-run findings: D113 app fixes done; D115 applied D114 items 1, 2, 4, 5 (stream decoding, state restriction, live prompts uncached, not-requested note); D114 items 3 (evidence ranked by aspect, not interest) and 6 (live items rarely cited) WAIT for the owner
- [ ] Known gaps: live search covers the request's named places or states, not the planner's stops (D107 a); 25 to 40 s model load on the first request; card items 58% useful (labelled)
- [ ] Done when: the demo runs locally (headless-tested; the owner's hand-run confirms); corpus retrieval and generation need no internet beyond the hosted model, and the live step degrades to the corpus without it.

## Phase 9: Evaluation and analysis
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: all tables and cases are ready for the report.
