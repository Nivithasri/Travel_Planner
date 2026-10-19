# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is COMPLETE: the corpus is FROZEN (2026-10-19T16:33:44Z) and tagged `phase-2-done`.
`data/processed/FROZEN` exists; the hook blocks Edit/Write on every path under data/processed and `scripts/process.py --write`
refuses. NEVER modify data/processed/* (CLAUDE.md rule 2). Nothing has been pushed since the Step 0 tag fix: `phase-1-done` is on
61f1355 (remote updated by an owner-authorized force push of that one tag); `phase-2-done` is local only. Push only when asked.
Phase 3 (enrichment and indexing) is IN PROGRESS on a changed route (see "Phase 3 state" below). Read this file, LOG.md, docs/phase-status.md first.

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

## Phase 3 state (2026-10-20, IN PROGRESS; not tagged; FULL RUN UNDER WAY since 2026-10-19 UTC)
Route (D59, supersedes D56; amends D1, D49, D58): the Kaggle account is banned, so enrichment uses a hosted OpenAI-compatible API on its FREE tier (Groq): generator
openai/gpt-oss-20b, judge openai/gpt-oss-120b (unused in Phase 3). Embeddings are computed LOCALLY on CPU with sentence-transformers bge-m3 (models/bge-m3-full, gitignored,
weights hashed in models/bge-m3-full/WEIGHTS_MANIFEST.json). Prompt `enrich-v4` (D62 applied, D63, D64: reply cap 400 tokens). The 79-chunk pilot passed the gate (LOG
2026-10-20): 2 of 79 notes fell back as ungrounded, 0 leaks, JSON valid, the three earlier problem notes (Badami, Hampi, Auroville) fixed. The index code is committed and was dry-run on a SCRATCH file
(data/enriched/scratch/, pilot checkpoint only): it works end to end. NOTHING real is built: data/enriched/chunks.jsonl and index/ do not exist.
1. OWNER ACTIONS. (a) Add `EMBED_LOCAL_DIR=models/bge-m3-full` to .env: the key is NOT there yet (key names checked; scripts/build_index.py and query_index.py take --model-dir
   until then). (b) Delete the old Ollama GENERATOR_MODEL/JUDGE_MODEL lines in .env (each key appears twice; the last wins; the scripts warn). (c) You must have read the provider's
   data-use terms and free-plan limits (D49 does not cover a third-party API); blog chunks stay OFF (no --include-blog) until you decide. (d) Optional but valuable: fill
   data/enriched/gold_sample.csv (30 chunks; labels empty), then `python scripts/label_sample.py --score` (needs `python scripts/enrich.py --gold` first).
2. EXACT FULL-RUN PLAN (approved and started by the owner on 2026-10-19 UTC; progress log in the daily routine below).
   Command, repeated once per UTC day until it prints no INCOMPLETE:
       python scripts/enrich.py --full --eligible-only
   Scope: the 17 D33-eligible places without blog = 1,100 chunks, of which 54 are already done by the pilot (checkpoint_enrich-v4_openai-gpt-oss-20b.jsonl), so 1,046 remain.
   Budget: the client keeps 85% of LLM_TPD per UTC day (170,000 tokens at the limits in .env) and 80% of RPM/TPM/RPD; it stores the day's spend in data/enriched/quota_state.json.
   Cost measured on the v4 pilot: 812 tokens and 1.00 calls per chunk, so about 209 chunks a day, about 27 minutes of wall time a day (the TPM pacing, ~7.7 s a chunk).
   PROJECTION (not a measurement): 1,046 x 812 = 849,000 tokens = about 5 days of budget; the day of the start counts only what is left of it (135,985 of 170,000 were spent by
   the pilots on 2026-10-19 UTC, so about 41 chunks fit before the 00:00 UTC reset, which is 05:30 India time), so expect 5 to 6 calendar days.
   A day ends with exit code 5 and the message "DAILY QUOTA USED UP ... Tokens used today N ... resets in about H h ... re-run the same command to resume". Other exit codes: 3
   five failures in a row (re-run the same command), 4 key/model/config rejected (stop and read the message), 130 Ctrl-C. Nothing is lost on any stop: every chunk is checkpointed
   (data/enriched/checkpoint_enrich-v4_openai-gpt-oss-20b.jsonl) and every reply cached (.cache).
   After the last day (the command prints no INCOMPLETE):
       python scripts/enrich.py --assemble --eligible-only      # data/enriched/chunks.jsonl + provenance.jsonl + data/manifests/enriched_manifest.json
       python scripts/build_index.py                            # real BM25 x2 + dense x2 on CPU, about 81 minutes (0.69 chunks/s measured); resumable every 200 rows
       python scripts/build_index.py --verify
       python scripts/query_index.py "quiet beaches in Kerala in monsoon"
   The 313 chunks of the 9 excluded places and the 283 blog chunks get the rule-based fields and are labelled rule_only in provenance.jsonl (the schema is closed, so the label
   lives in the sidecar); Phase 6 must read it before trusting their aspect/season fields.
   Do not use --include-blog before the terms are confirmed.
3. Comparability: the v1 pilot (200 chunks, qwen2.5 via Ollama, checkpoint_enrich-v1.jsonl) and the v3 pilot are history; nothing is comparable across models or prompts
   (scripts/compare_runs.py marks it). *.superseded.jsonl files hold recomputed records (D64) and are never read as results.
4. Enriched chunks go to data/enriched/chunks.jsonl (gitignored), NEVER to data/processed (D48 xi); check_enriched_matches_frozen must pass. Phase 3 may fill context_note,
   place_type, season_tags, aspect, is_warning and the optional mentioned_places (D50, to be sent to Valincia). O12 still blocks sharing anything enriched with the partner.
5. Never print or log the URL, the key or response headers: both clients scrub every error and debug record (tests). data/enriched/debug_raw.jsonl is UNTRUSTED model output.
6. Not started: the early sample file for Valincia (O12), the final HANDOFF rewrite and the phase-3-done tag.

## Daily routine for the full enrichment run (Phase 3)
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
  `scripts/smoke_llm.py`, `scripts/compare_runs.py`, `scripts/mutation_checks.py` (14 mutations), `OpenAICompatClient`/`RateLimiter`/`scrub_text` in
  `src/common/llm.py`, `load_llm_settings` in `src/common/schemas.py`.
- Index code: `src/retrieval/build_index.py` (BM25 contextual/plain with bm25s, dense plain/contextual with local sentence-transformers, resumable, manifest, verify), `src/retrieval/search.py`
  (exact numpy cosine, ties by chunk_id), `scripts/build_index.py`, `scripts/query_index.py` (four ranked tables).
- Tests: 656 passed, 1 skipped (tests/test_process.py, test_enrich.py, test_llm_api.py, test_index.py, test_label_sample.py, test_hooks_scripts.py, ...); `scripts/mutation_checks.py` runs 16 mutations.

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
