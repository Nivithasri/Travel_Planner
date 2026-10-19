# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is BUILT and verified but STOPPED BEFORE THE FREEZE, on the owner's instruction (the freeze
needs the owner's go-ahead after Reddit is resolved). There is NO `phase-2-done` tag. `data/processed/FROZEN` does not exist.
Phase 1 is complete; tag `phase-1-done` is on 61f1355 (moved from 8ed3f86 on 2026-10-19 with an owner-authorized force push of
that one tag; `git ls-remote --tags origin` shows 61f1355). Push only when asked; never force a branch.
Next: the owner decides the open Phase 2 items below, then runs the freeze procedure. Do not start Phase 3.

## Verified at the end of this session (commands and outputs in LOG.md, 2026-10-19 Phase 2 entries)
`python -m pytest -q` 415 passed; `python -m ruff check .` clean; `python scripts/process.py --reference-date 2026-10-18 --write`
run twice -> identical sha256 for every output (chunks.jsonl 74d8d571...); request log 443 lines before and after (no network);
5 Phase 2 mutations and the Phase 1 hold-out mutation (13 failures) caught, `git diff` empty after each restore.

## What exists
- Code: `src/process/process.py` (clean, chunk, hold-out, dedupe, attribution, stats; `process_place()` is pure and is reused by
  the live path), `scripts/process.py` (`learn` | dry run | `--write` | `--live <slug>`, `--reference-date`).
  `HoldoutFilter.text_match` in src/facts/facts.py. Schema: `date_kind`, `revised_date` (D48). Tests: tests/test_process.py (36).
- Rule file: `config/boilerplate.json` (inditales.com: 0 applied lines, 2 rejected by review: "Travel Tips", "Meditation Hall").
- Outputs (gitignored, NOT frozen): `data/processed/chunks.jsonl` (1,389), `drops.jsonl` (31: 30 hold-out chunks, 1 exact
  duplicate), `attribution.json` (96 urls: title, source, licence, revid, lastmod). Live: `data/live/processed/yercaud.jsonl` (18).
- Committed: `data/manifests/corpus_stats.{md,json}` (per source, state, place, year, date_kind, tokens, drops, D23 dry run,
  comparison with the 1,943 estimate) and `data/manifests/processed_manifest.json` (sha256, bytes, lines; no content).
- Phase 1 layout unchanged (src/common, src/collect, src/places, src/facts; scripts/collect.py, coverage_report.py, smoke_ollama.py).
  config/core_overrides.json was deleted (no code read it).

## How to run
- `pip install -e .[dev]`; Python 3.10.11 (D9); `python -m pytest -q`, `python -m ruff check .` (not on PATH).
- `git config core.hooksPath .githooks` in any fresh clone. No AI attribution anywhere (CLAUDE.md).
- `.env` MISMATCH: it holds only CONTACT_EMAIL and YOUTUBE_API_KEY (key names checked, values never read). The previous HANDOFF said
  it also held OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, EMBED_MODEL and EVAL_REFERENCE_DATE: it does not. Phase 2 therefore
  takes `--reference-date 2026-10-18`; Phase 3 (Ollama) needs the owner to add the Ollama keys. `.env` was not modified.
- Optional `TOKENIZER_DIR` (local folder with the bge-m3 tokenizer.json; never downloaded). Unset = regex proxy counter.

## Phase 2 numbers (proxy counter; will change with the real tokenizer)
1,389 chunks (estimate 1,943, range 1,619-2,429; -554, mainly because the proxy counts 5.04 characters per token vs the estimate's 4;
cleaning removed 3.2% of characters). blog 242, reference 558, wikivoyage 589. date_kind: published 242, revised 1,104,
sitemap_lastmod 43; freshness share 0 (no published date in 2025-04-18..2026-10-18). Hold-out drops: Araku Valley 8, Thanjavur 8,
Thekkady 14, Chikmagalur 0; record-level (url/title) 0. D23 dry run: 17 of 26 at chunk level, 18 at raw level.

## OPEN OWNER ITEMS (Phase 2)
1. Kodaikanal (D48 amendment g): the agreed exact-dedupe rule keeps the Ooty/Kodaikanal post under Ooty, so Kodaikanal has no blog
   chunk and fails D23 at chunk level (17 vs 18 pass). Keep, or assign that post to Kodaikanal before the freeze.
2. Review the build-time rules (D48 amendments): inline-link joins, image-credit lines, mid-post "Read more" handling, the two
   rejected learned lines, "km."/"ft." not abbreviations, added hold-out spellings Thekkadi and Brihadeeswara.
3. "Periyar Dam" in a Madurai chunk (D48 amendment h) is kept; "Dam" is not in the O8 context list. Drop or keep?
4. `.env` keys (above). Reddit credentials (D45). O12 sharing (below).
Phase 1 items still open: terms quotes and the wiki robots gap (D36/D43), O12, the pre-squash bundle location, D24 review, outreach
emails unsent, O13 airport allowlist file (Phase 5), attribution in the report and app (Open-Meteo, OSM, Wikimedia).

## FREEZE PROCEDURE (not executed; run only on the owner's explicit go-ahead)
Preconditions: (a) Reddit resolved (credentials arrived and the collector ran, or absence accepted; D45); if new raw data arrives,
rerun `learn`, review its lines, and redo the dry run. (b) The real bge-m3 tokenizer available locally: set TOKENIZER_DIR; a freeze
with the regex proxy is not allowed (D48 2). (c) Owner decisions on items 1-3 above.
Steps: 1. `python scripts/process.py --reference-date <EVAL_REFERENCE_DATE> --write` twice; both runs must give the same sha256 in
data/manifests/processed_manifest.json, whose `counter` must name tokenizer.json. 2. Apply D33 ONCE: exclude the places that fail
D23 at chunk level from chunks.jsonl (record the list and reasons in LOG.md; raw data and registry entries are kept; the planner must
not recommend them; no Phase 4 query may target them). This needs a small `--apply-d33` step, which is not written yet; write and test
it first. 3. Rewrite the stats and manifest for the final file; assert no hold-out hits and no source_type live (the scan command in
LOG.md). 4. Create `data/processed/FROZEN` holding the sha256 of chunks.jsonl, drops.jsonl and attribution.json; commit the manifest
and stats. 5. Extend scripts/hooks/guard_frozen.py RULES to `data/processed/*.jsonl` and `data/processed/attribution.json` (today it
covers chunks*.jsonl only), and add a test that a write to a frozen file is blocked and that `scripts/process.py --write` refuses
(it already exits when FROZEN exists; there is no test for it yet). 6. Update LOG.md and phase-status, commit, `git tag phase-2-done`.

## How Valincia gets the chunks (O12 open)
data/processed is gitignored, and O12 (what may be shared, given CC BY-SA, CC BY-NC-SA and the inditales "all rights reserved" text)
is undecided, so chunks.jsonl is NOT pushed or uploaded. Until the owner decides, she receives nothing through git; if the owner
shares the file privately, she verifies it against data/manifests/processed_manifest.json (sha256). Attribution for every url is in
data/processed/attribution.json.

## Note for Valincia (live layer, D31 proposed, not binding until you acknowledge)
`process_place()` in src/process/process.py chunks one place's raw records without touching data/processed; `python
scripts/process.py --live <slug>` shows the path (data/live/raw -> data/live/processed). Chunks carry date_kind; freshness counts
`published` only. The frozen corpus has no fresh content (D32). HttpClient, cache and request log are reusable (src/common).

## Known gaps and cautions
- Token counts are a proxy until TOKENIZER_DIR is set; chunk ids (url + character offset) change if cleaning rules change before the freeze.
- No topical off-topic filter beyond the wiki section blocklist (content sections such as Demographics are kept).
- Blog dates are the UTC day from article:published_time (the IST <time> can be a day later).
- The learned-line rule finds nothing to apply on this corpus; it exists for new raw data.
- Hooks: keep core.hooksPath set. WORKING RULE (owner): a STOP rule means stop BEFORE acting.
