# Phase status
Tick only after verification commands were run (see LOG.md).

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

## Phase 2: Cleaning and chunking (built 2026-10-19; STOPPED BEFORE FREEZE; no phase-2-done tag)
- Verified (commands in LOG.md, 2026-10-19 Phase 2 entries): pytest 424 passed, ruff clean, two `--write` runs byte-identical, 6 mutations (a-f) + Phase 1 hold-out mutation caught, request log unchanged (443)
- [x] Boilerplate removal (fixed rules + reviewed rule file config/boilerplate.json), exact and near-duplicate removal (D48)
- [x] South India scope: every record's place checked against the registry (collectable, not hold-out)
- [x] Dates normalized; date_kind published | revised | sitemap_lastmod | none with BaseChunk invariants (D48, O9)
- [x] Chunks 200-300 tokens (hard max incl. overlap), source_type/url/date/state/place; deterministic ids (1,388 chunks, proxy counter)
- [x] Hold-out drop (url/title record level, text chunk level), counted per hold-out and source; output scan 0 hits
- [x] Corpus statistics: data/manifests/corpus_stats.{md,json}; sha256 manifest data/manifests/processed_manifest.json
- [x] D23/D33 dry run: 17 of 26 pass at chunk level (18 at raw level; Kodaikanal, D48 amendment g); nothing removed
- [ ] Freeze: NOT done. Needs the owner's go-ahead after Reddit is resolved, the real bge-m3 tokenizer (TOKENIZER_DIR), D33 applied, and the checklist in docs/HANDOFF.md
- [ ] Done when: the corpus is frozen. From this point it never changes, so every metric stays reproducible.

## Phase 3: Enrichment and indexing
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: a script takes a query and returns ranked chunks from each index.

## Phase 4: Test collection and evaluation harness
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: one command prints all metrics for any retrieval function.

## Phase 5: Routing reference data
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: both functions return instantly for every registry place.

## Phase 6: Retrieval engine and ablations
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: the ablation table is complete and `retrieve()` returns ranked, filtered, reranked, freshness-aware chunks.

## Phase 7: Place cards and agent pipeline
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: a query returns a cited itinerary end to end, with practical tips and warnings, a rough budget estimate, and backup alternatives.

## Phase 8: App
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: the demo runs locally; corpus retrieval and generation need no internet, the live step does and degrades to corpus and cache without it (plan.md; live layer is D31, proposed to Valincia).

## Phase 9: Evaluation and analysis
- [ ] Tasks (see docs/plan.md)
- [ ] Done when: all tables and cases are ready for the report.
