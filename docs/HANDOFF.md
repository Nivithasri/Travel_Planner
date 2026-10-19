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

## Phase 3 state (2026-10-20, IN PROGRESS; not tagged)
Route changed (D59, supersedes D56; amends D1, D49, D58): the Kaggle account is BANNED, so there is no Kaggle notebook, no Kaggle dataset and
no ngrok Ollama (kaggle/README.md is obsolete). Enrichment uses a hosted OpenAI-compatible API on its FREE tier (Groq): generator
openai/gpt-oss-20b, judge openai/gpt-oss-120b (judge unused in Phase 3). Embeddings (plain and contextual) are computed LOCALLY on CPU with
sentence-transformers bge-m3 for the 1,696 chunks and for queries; the Ollama embed-agreement check and the Kaggle parity check are dropped
(deviations in D59). No API call has been made with the new client: it was written and tested offline with a fake transport and clock.
1. OWNER ACTIONS before the first call. (a) READ AND CONFIRM THE PROVIDER'S DATA-USE TERMS (retention, training on inputs) and the FREE-PLAN
   LIMITS from its dashboard; nothing in the repo verifies either (no network in the implementation session). D49 covered only the private
   Kaggle; it does not cover a third-party API. The first call sends reference and wikivoyage chunk text (CC BY-SA); blog chunks (inditales.com,
   a copyrighted personal blog) are sent only with --include-blog. (b) Fix .env: the URL key must be LLM_API_BASE_URL (LLM_API_URL is accepted
   as an alias); LLM_API_KEY, GENERATOR_MODEL, JUDGE_MODEL, LLM_RPM, LLM_TPM, LLM_RPD, LLM_TPD must all be set (any unset = the scripts refuse
   to run); GENERATOR_MODEL and JUDGE_MODEL appear TWICE in the current .env (names only checked; the last one wins; the scripts warn): delete
   the old Ollama pair. (c) Later, for indexing: pip install bm25s PyStemmer sentence-transformers, download bge-m3 to a folder and set
   EMBED_LOCAL_DIR in .env (the assistant never installs or downloads).
2. Run order (each step stops on any problem): `python scripts/smoke_llm.py`; then `python scripts/enrich.py --pilot` (79 chunks: 60 seeded + the
   gold chunks that are not blog, 19 new; blog off; prints tokens per chunk and a PROJECTION of days for all 1,696 chunks and for the
   eligible-only subsets). `--pilot --eligible-only` would send 71 chunks but only 12 of the 30 gold chunks are usable then (the gold sample
   was drawn from the whole corpus: 10 are blog and 8 are in excluded places). Only after the owner approves: `python scripts/enrich.py --full --eligible-only`, repeated each day until it prints no INCOMPLETE
   (exit 5 = the daily quota is used up, resets at the next UTC day; exit 3 = five failures in a row; exit 4 = key/model/config rejected), then
   `python scripts/enrich.py --assemble --eligible-only` (writes data/enriched/chunks.jsonl, provenance.jsonl and the manifest; unprocessed
   chunks of the 9 excluded places and the blog chunks get the rule-based fields, labelled rule_only in provenance.jsonl).
3. The owner's gold labels: data/enriched/gold_sample.csv (30 chunks, text only, labels empty). Fill aspect_gold and is_warning_gold, then
   `python scripts/label_sample.py --score` (needs LLM labels for those chunks: run `python scripts/enrich.py --gold`; with blog off 20 of the
   30 are sent).
4. Comparability: the v1 pilot (200 chunks, qwen2.5 via Ollama, data/enriched/checkpoint_enrich-v1.jsonl) is NOT comparable with the new model;
   any v1-vs-v3 comparison is confounded (scripts/compare_runs.py prints that). Prompt versions: v1 and v2 were Ollama-era; v2 never ran; the
   current prompt is enrich-v3 (shrunk: 1,071 characters, D60).
5. Enriched chunks go to `data/enriched/chunks.jsonl` (gitignored), NEVER to data/processed (D48 xi); `check_enriched_matches_frozen` must pass.
   Phase 3 may only fill context_note, place_type, season_tags, aspect, is_warning, plus the optional mentioned_places (D50, to be sent to
   Valincia). `place` stays as frozen. O12 still blocks sharing anything enriched with the partner.
6. Never print or log the URL, the key or response headers: OllamaClient/OpenAICompatClient scrub every error and debug record (tests).
   data/enriched/debug_raw.jsonl is UNTRUSTED model output, scrubbed of keys and URLs, gitignored.
7. Frozen-file hook: data/processed is read-only. Indexes go to index/ (gitignored). The freeze tests assert the real FROZEN is unchanged.
   Not started: BM25 (contextual and plain, D55), the dense indexes, scripts/query_index.py, the index manifest, the early sample for Valincia.

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
- Tests: 588 passed, 1 skipped (tests/test_process.py, test_enrich.py, test_llm_api.py, test_label_sample.py, test_hooks_scripts.py, ...).

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
