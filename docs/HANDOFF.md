# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is COMPLETE: the corpus is FROZEN (2026-10-19T16:33:44Z) and tagged `phase-2-done`.
`data/processed/FROZEN` exists; the hook blocks Edit/Write on every path under data/processed and `scripts/process.py --write`
refuses. NEVER modify data/processed/* (CLAUDE.md rule 2). Nothing has been pushed since the Step 0 tag fix: `phase-1-done` is on
61f1355 (remote updated by an owner-authorized force push of that one tag); `phase-2-done` is local only. Push only when asked.
Next: Phase 3 (enrichment and indexing). Start it in plan mode and wait for approval. Read this file, LOG.md, docs/phase-status.md first.

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

## Phase 3 start-up checklist
1. Enriched chunks go to `data/enriched/chunks.jsonl` (gitignored), NEVER to data/processed (D48 xi). After writing, run
   `check_enriched_matches_frozen(enriched, frozen)` (src/process/process.py): same chunk_id set and identical chunk_id, text, url,
   date, date_kind, revised_date, state, place, source_type per id. Phase 3 may only fill context_note, place_type, season_tags, aspect,
   is_warning (EnrichedChunk, D10). `place` is already set by Phase 2; O5 asks Phase 3 to check LLM-extracted places against the
   registry through aliases, with rule-based fallback.
2. `.env` now holds the Ollama keys (OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, EMBED_MODEL) as well as CONTACT_EMAIL,
   YOUTUBE_API_KEY, TOKENIZER_DIR, EVAL_REFERENCE_DATE (key names only). `scripts/smoke_ollama.py` was run by the owner against the
   Kaggle/ngrok Ollama on 2026-10-19: 6/6 PASS (generator qwen2.5:latest, judge llama3.1:8b, embed bge-m3:latest). Owner-reported;
   the tunnel URL changes each session.
3. O12 for PROCESSING is decided (D49): all chunks may go to the owner's PRIVATE Kaggle notebook/dataset; nothing enriched is shared
   with the partner or made public until the owner resolves O12. Enrichment runs locally through the tunnel (D56); the notebook embeds.
4. Read docs/papers notes if any exist (only an index README exists; none for BGE-M3 or Contextual Retrieval yet).
5. Frozen-file hook: data/processed is read-only. Indexes go to index/ (gitignored). Embeddings must use the same model and
   tokenizer family; D2 asks for a vector-parity check on a few chunks.
6. Tests: the freeze tests assert the real FROZEN is unchanged (`_real_frozen_untouched`), not absent. Do not delete data/processed/FROZEN.

## What exists (code)
- `src/process/process.py` (clean, chunk, hold-out, dedupe, attribution, stats, `check_corpus`, `check_enriched_matches_frozen`;
  `process_place()` is pure and is reused by the live path), `scripts/process.py` (`learn` | dry run | `--write` | `--live <slug>`),
  `scripts/freeze.py` (already executed), `scripts/phase2_evidence.py` (read-only evidence; `all`), `config/boilerplate.json`
  (0 applied lines, 2 rejected), `config/freeze.json` (reddit_resolved true, d33_mode sidecar).
- `HoldoutFilter.text_match` in src/facts/facts.py (hold-out terms incl. Borra Caves, Periyar context rule O8).
- Schema: `date_kind`, `revised_date` (D48). Decisions: docs/decisions.md D48 (i-xix) holds every Phase 2 rule and amendment.
- Phase 1 layout unchanged (src/common, src/collect, src/places, src/facts; scripts/collect.py, coverage_report.py, smoke_ollama.py).
- Tests: 436 (tests/test_process.py, test_hooks_scripts.py incl. freeze tooling, test_guards.py, test_common.py, ...).

## How to run
- `pip install -e .[dev]`; Python 3.10.11 (D9); `python -m pytest -q`, `python -m ruff check .` (not on PATH).
- `git config core.hooksPath .githooks` in any fresh clone. No AI attribution anywhere (CLAUDE.md).
- models/bge-m3/tokenizer.json (17 MB) is the owner's local TOKENIZER_DIR; models/ is gitignored. `.gitattributes` pins LF for
  data/manifests/*.json and *.md (the two d33_excluded.json copies must stay byte-identical).

## OPEN OWNER ITEMS
1. O12 for SHARING: what may be shared with the partner, pushed or made public (processing on a private Kaggle is decided, D49).
2. Before Phase 3 step 6: install bm25s, PyStemmer, sentence-transformers and download bge-m3 to a local folder set as EMBED_LOCAL_DIR in .env (the assistant never installs or downloads). Reddit credentials only if Reddit is ever added to the live layer (D48 xvi).
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
