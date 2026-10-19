# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 2 (cleaning and chunking) is BUILT and verified but STOPPED BEFORE THE FREEZE, on the owner's instruction (the freeze
needs the owner's go-ahead after Reddit is resolved). There is NO `phase-2-done` tag. `data/processed/FROZEN` does not exist.
Phase 1 is complete; tag `phase-1-done` is on 61f1355 (moved from 8ed3f86 on 2026-10-19 with an owner-authorized force push of
that one tag; `git ls-remote --tags origin` shows 61f1355). Push only when asked; never force a branch.
Next: the owner decides the open Phase 2 items below, then runs the freeze procedure. Do not start Phase 3.

## Verified at the end of this session (commands and outputs in LOG.md, 2026-10-19 Phase 2 entries)
`python -m pytest -q` 433 passed; `python -m ruff check .` clean; `python scripts/process.py --reference-date 2026-10-18 --write`
run twice -> identical sha256 for every output (chunks.jsonl b3265a45...); request log 443 lines before and after (no network);
mutations (a-f) on the pipeline, (g) on the freeze tokenizer check and the Phase 1 hold-out mutation (13 failures) caught, `git diff
--stat` empty after each restore. Every review figure is reproducible with `python scripts/phase2_evidence.py all` (read-only).
`python scripts/freeze.py --dry-run` on the real corpus: exit 2, 4 FAIL (proxy counter, reference date not from .env, Reddit
unresolved, D33 not applied), 4 PASS.

## What exists
- Code: `src/process/process.py` (clean, chunk, hold-out, dedupe, attribution, stats; `process_place()` is pure and is reused by
  the live path), `scripts/process.py` (`learn` | dry run | `--write` | `--live <slug>`, `--reference-date`).
  `HoldoutFilter.text_match` in src/facts/facts.py. Schema: `date_kind`, `revised_date` (D48). Tests: tests/test_process.py.
  `scripts/freeze.py` (dry run by default; `--apply-d33`; `--execute`), owner decisions in `config/freeze.json`
  (reddit_resolved false, d33_mode null). `scripts/phase2_evidence.py` (read-only evidence).
- Rule file: `config/boilerplate.json` (inditales.com: 0 applied lines, 2 rejected by review: "Travel Tips", "Meditation Hall").
- Outputs (gitignored, NOT frozen): `data/processed/chunks.jsonl` (1,387), `drops.jsonl` (33: 32 hold-out chunks, 1 exact
  duplicate), `attribution.json` (96 urls: title, source, licence, revid, lastmod). Live: `data/live/processed/yercaud.jsonl` (18).
- Committed: `data/manifests/corpus_stats.{md,json}` (per source, state, place, year, date_kind, tokens, drops, D23 dry run,
  comparison with the 1,943 estimate) and `data/manifests/processed_manifest.json` (sha256, bytes, lines; no content).
- Phase 1 layout unchanged (src/common, src/collect, src/places, src/facts; scripts/collect.py, coverage_report.py, smoke_ollama.py).
  config/core_overrides.json was deleted (no code read it).

## How to run
- `pip install -e .[dev]`; Python 3.10.11 (D9); `python -m pytest -q`, `python -m ruff check .` (not on PATH).
- `git config core.hooksPath .githooks` in any fresh clone. No AI attribution anywhere (CLAUDE.md).
- `.env` MISMATCH: it holds only CONTACT_EMAIL and YOUTUBE_API_KEY (key names checked, values never read). The previous HANDOFF said
  it also held OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, EMBED_MODEL and EVAL_REFERENCE_DATE: it does not. OWNER: add
  `EVAL_REFERENCE_DATE=2026-10-18`; until then pass `--reference-date 2026-10-18` (it overrides .env when both are set; the stats
  record the value and its source). Phase 3 (Ollama) also needs the Ollama keys. `.env` was not modified.
- Optional `TOKENIZER_DIR` (local folder with the bge-m3 tokenizer.json; never downloaded). Unset = regex proxy counter.

## Phase 2 numbers (regex proxy counter; recompute with the real tokenizer)
1,387 chunks (estimate 1,943, range 1,619-2,429). Measured: 5.036 characters per proxy token vs the estimate's 4; cleaning removed
3.2% of characters. Whether bge-m3 gives more chunks is a HYPOTHESIS, not measured. Hold-out drops (all chunk level): Araku Valley 9
(incl. Borra Caves), Thanjavur 8, Thekkady 15, Chikmagalur 0; record-level (url/title) 0. D23 dry run: 17 of 26 at chunk level (Kodaikanal fails at
chunk level only), 18 at raw level; if D33 were applied: 17 places, 1,143 chunks, 0 hidden_gem places. Near-dup threshold removes
nothing (max Jaccard 0.2155). Accepted deviations (D48 vii): 10k-20k chunks unreachable; 3 source types, one blog host; 0
published chunks in the 18-month window.

## OPEN OWNER ITEMS (Phase 2)
DECIDED in the review round: Kodaikanal stays a chunk-level D23 fail (D48 i); Thekkadi/Brihadeeswara accepted (D48 ii); Periyar Dam
dropped and O8 amended (D48 iii); reference date from .env (D48 iv).
1. Review the build-time cleaning rules (D48 amendment a-e): evidence in `python scripts/phase2_evidence.py rejoin credits`.
2. DECIDED: "Borra Caves" is an Araku Valley term (D48 viii); chunk 500b7e3d23575ed7 is dropped. Residual risk accepted: landmarks
   not on the term list can still leak; the scanned landmark list was the reviewer's and is unverified.
Accepted cost (D48 xiv): the chunk-level hold-out rule removes 11 Munnar Wikivoyage chunks via the Thekkady terms.
3. Set `config/freeze.json` reddit_resolved (after D45 is settled) and d33_mode ("sidecar" is the only implemented mode).
4. `.env` keys (above). Reddit credentials (D45). O12 sharing (below). Wikimedia Terms of Use section 7 wording: not in the repo
   (no fetched copy); needed for the report and app attribution (Phase 8).
Phase 1 items still open: terms quotes and the wiki robots gap (D36/D43), O12, the pre-squash bundle location, D24 review, outreach
emails unsent, O13 airport allowlist file (Phase 5), attribution in the report and app (Open-Meteo, OSM, Wikimedia).

## FREEZE PROCEDURE (tooling built and tested in temp directories; NOT executed on the real corpus)
Preconditions, all checked by `python scripts/freeze.py` (dry run, pass/fail per line):
(a) REDDIT: resolved (collected, or absence accepted, D45) and recorded by the owner as `reddit_resolved: true` in config/freeze.json.
If new raw data arrives, rerun `learn`, review its lines, and redo the dry run.
(b) TOKENIZER: the real bge-m3 tokenizer.json available locally, TOKENIZER_DIR set in .env; processed_manifest.json `counter` must be
"tokenizer.json sha256:<first 12 hex of that file>"; the proxy is refused (D48 2). All chunk counts above are then recomputed.
(c) REFERENCE DATE: EVAL_REFERENCE_DATE=2026-10-18 set in .env; the final --write must run WITHOUT --reference-date so corpus_stats
`reference_date_source` reads "EVAL_REFERENCE_DATE in .env" and equals the .env value.
(d) D33: owner sets `d33_mode: "sidecar"`; `python scripts/freeze.py --apply-d33` (dry run, prints the excluded places and the effects
table) then `--apply-d33 --execute` writes data/processed/d33_excluded.json: a Phase 4 QUERY-ELIGIBILITY list (no dev/test query may
target a listed place). chunks.jsonl is unchanged and retrieval is NOT filtered, unless the owner records otherwise in
config/freeze.json (D48 xii). Expected today: 9 listed places, 17 eligible, 1,143 chunks of eligible places, 0 hidden_gem (D44).
(e) check_corpus() passes and no source_type live (also run inside every --write); outputs match processed_manifest.json.
Steps: 1. `python scripts/process.py --write` twice (no --reference-date); same sha256 both times. 2. Commit manifest and stats.
3. `python scripts/freeze.py --apply-d33 --execute`. 4. `python scripts/freeze.py` (dry run) must show 8 PASS. 5. `python
scripts/freeze.py --execute` writes data/processed/FROZEN and data/manifests/frozen_manifest.json (same fields, no text: sha256
of chunks.jsonl and attribution.json, chunk count, counter, sha256 of every tokenizer file the counter loads (tokenizer.json), the
tokenizers version, reference date and source, UTC timestamp; D48 xiii); commit frozen_manifest.json. After this the hook blocks Edit/Write on every data/processed
path and process --write refuses. 6. Record the FROZEN content in LOG.md, update phase-status, commit, `git tag phase-2-done`.

## Phase 3 notes (D48 xi)
- Enriched chunks go to `data/enriched/chunks.jsonl` (gitignored), NEVER to data/processed (frozen as a whole, D48 x).
- Before any Phase 3 output is used, run `check_enriched_matches_frozen(enriched, frozen)` (src/process/process.py): same chunk_id set,
  identical chunk_id, text, url, date, date_kind, revised_date, state, place, source_type per id. Phase 3 only fills context_note,
  place_type, season_tags, aspect, is_warning (EnrichedChunk). docs/contracts.md "Locations" records this.
- Phase 3 also needs the Ollama keys in .env (currently absent) and `scripts/smoke_ollama.py` run once.

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
