# HANDOFF (overwritten at the end of every phase; the next session reads this first)

## Status
Phase 1 (data collection and place registry) is COMPLETE and reviewed (`/phase-review 1`, 2026-10-19, verdict PASS WITH NOTES), tagged `phase-1-done`.
Next: Phase 2 (cleaning, chunking, freeze). Start it in plan mode and wait for approval.
PUSHED: origin/main holds "Initial commit", then 8ed3f86 "Finalize Phase 1" (all Phase 1 work squashed into one commit; tag phase-1-done
on 8ed3f86, pushed), then the D47 commit "Simplify Phase 1 code". Push only when asked, plain fast-forward, never force. The pre-squash
history exists only in the local bundle ../travel-planner-pre-squash.bundle (never pushed; it holds the old third-party channel name and
the old .env.example address, and it sits in the OneDrive-synced Documents folder).

Verified at the end of Phase 1 (commands in LOG.md, 2026-10-19 entries): `python -m pytest -q` 452 passed at the review, 415 after D46, 352 after D47 (tests of deleted code removed; LOG.md lists them); `python -m ruff check .` clean;
`wc -l data/manifests/raw_manifest.jsonl` 179 (153 file/record lines verified by sha256 + 13 youtube lines + 13 withdrawn lines); request log 443 lines in 13 runs, network + cache = lines
in every run; `python scripts/coverage_report.py` 18 of 26 places pass D23.

## What exists (all gitignored except the registry and manifests)
- Registry: `data/registry/core_places.json` (30 places: 26 collectable + 4 hold-outs Chikmagalur, Araku Valley, Thekkady, Thanjavur; 26 popular, 4 hidden_gem; popularity_signal = 12-month Wikipedia pageviews).
- Raw records `data/raw/<source>/<place>/records.jsonl`: 97 records (wikipedia 26, wikivoyage 25, blog 43, sahapedia 3, official 0, reddit 0; one inditales post is stored under both Ooty and Kodaikanal).
- Facts envelopes (not records, D37): `data/raw/wikidata` 26, `data/raw/open_meteo` 29 (26 + 3 noelev files, D42 primary rules), `data/raw/overpass/_south/airports.json` (48 elements, unfiltered). Each Open-Meteo file: 3,653 days, 0 nulls.
- Live store `data/live/` (Yercaud demo only); `.cache/http` (1,288 files after the quarantine); `.cache/_quarantine/` (64 page entries from hosts that are excluded or ask-first in docs/sources.md, moved not deleted; INDEX.jsonl lists them).
- Code (D47, 8 modules, no __init__.py; namespace packages): `src/common/{http,cache,llm,schemas}.py` (http = client + allowlist + request log + redaction; schemas = settings + pydantic models), `src/collect/sources.py` (wiki, web, Reddit collectors), `src/collect/run.py` (types, raw store, live store, coverage, ensure_place, orchestrator; YouTube is refused there in one entry, D27), `src/places/places.py` (registry, Wikidata lookups, resolver), `src/facts/facts.py` (RFC 9309 matcher, hold-out filter, api_hosts, envelopes, Wikidata/Overpass/Open-Meteo fetchers). Placeholders: `src/{process,retrieval,agents,tools}/.gitkeep`.
- Scripts: `scripts/collect.py core|facts|demo ...` (old collect_core, collect_facts, ensure_place_demo; same flags and exit codes; plan only without --execute), `scripts/coverage_report.py` (offline), `scripts/smoke_ollama.py` (Phase 3), hooks in `scripts/hooks/`. The historical scripts, the policy-page and withdraw tools, the size estimates and check_robots were deleted in D47 (the bundle and git history keep them).
- Guards: every production client refuses hosts outside config/sources.json before any network call (D43); `tests/test_guards.py` holds every test a mutation check relies on (allowlist, redirect refusals, api_hosts golden list, request-log guard, contact privacy, circuit breaker, hold-out filter). Tests: six files (test_common, test_collect, test_places, test_facts, test_hooks_scripts, test_guards).

## How to run
- Install `pip install -e .[dev]`; Python 3.10.11 (D9); `ruff`/`pytest` are not on PATH: `python -m pytest -q`, `python -m ruff check .`.
- `git config core.hooksPath .githooks` in any fresh clone. Do not change git user.name/user.email. No AI attribution anywhere (CLAUDE.md).
- `.env` (gitignored, never read or print it): OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, EMBED_MODEL, EVAL_REFERENCE_DATE (2026-10-18), CONTACT_EMAIL, optional YOUTUBE_API_KEY.
- The Ollama client has never talked to a real server (all tests mock HTTP); `scripts/smoke_ollama.py` has not been run. Run it before Phase 3.

## Decisions to know (docs/decisions.md; read D17 to D45 before Phase 2)
D17 on-demand ingestion (`ensure_place`), frozen snapshot vs live store separate. D18 Puducherry = PY. D19 popularity = top 20% of the pool by pageviews. D20 live places use template notes and CPU embeddings.
D21 Kodagu and Wayanad are district centroids. D22 source_type "reference" (Wikipedia, Sahapedia). D23 coverage rule (>= 2 source types, one not wikivoyage/reference). D24 four web domains scoped by state/place (owner review pending).
D25 automatic page discovery. D26/D34 corpus size target not met (about 1,943 chunks estimated, 19% of 10,000; no chunk-size games). D27/D30 YouTube OFF for anything stored.
D29 policy-page fetch allowlist. D31/D35 live layer proposed to Valincia, not binding until she acknowledges. D32 no curated notes; freshness comes only from the live layer.
D33 apply D23 ONCE at the Phase 2 freeze; failing places leave the frozen corpus (raw and registry kept). D36/D37/D40/D41/D43 API hosts, robots exemption, Overpass limited to one airport query, RFC 9309 matcher, allowlist.
D38 circuit breaker. D39 freeze rules and deviation numbers. D42 elevation rules. D44 no hidden_gem place survives D33; `popularity_tercile` proposed (docs/popularity-terciles.md); Phase 4 diversity = distinct source types and places.
D45 deviations: YouTube dropped, Reddit deferred, 0 official, 18 of 26 pass D23.

## Phase 1 deviations from the plan (as measured)
Reddit 0 records (no credentials; run the collector before the freeze if credentials arrive, otherwise accepted at the freeze, D45). YouTube dropped. Official 0. D23 passes 18 of 26 (plan: 20 to 30). About 1,943 chunks estimated.
0 records inside the 18-month window (newest publish date Ooty 2025-01-30). Kerala: only Kochi passes D23. The 8 D23 failures: Auroville, Rameswaram, Srisailam, Munnar, Wayanad, Alappuzha, Varkala, Bhadrachalam.
Never exercised live: Reddit, cross-host/prefix/userinfo redirect refusals (fake transport only; one same-host 301 on developers.google.com ran live, run c2141969).
Scan scopes: key-prefix scans are 0 over tracked files, all revisions, git log -p, data/, docs/, LOG.md, src, tests, scripts, config and .cache; the gitignored .cache holds 36 generic `AIzaSy` strings in 12 files (third-party page Maps keys, compared with the owner's key only by exact 8-character prefix).

## OPEN OWNER ITEMS
1. Terms quotes (config terms_status unchanged): paste primary quotes for en.wikipedia.org, en.wikivoyage.org, www.wikidata.org /w/api.php, query.wikidata.org (secondary evidence only, D43/facts-layer 5d), the wikimedia.org pageviews host (secondary evidence), Overpass and Open-Meteo.
   Also the ROBOTS gap: robots.txt for en.wikipedia.org and en.wikivoyage.org was never fetched, so the exemption for their /w/api.php calls rests on D36 and the Wikimedia API terms, not on a matcher verdict.
2. O12: what may be shared, pushed or uploaded (ODbL airports, CC BY-SA text, CC BY-NC-SA Sahapedia).
3. Done: Phase 1 squashed and pushed (8ed3f86, tag phase-1-done). Decide whether to keep or move the local pre-squash bundle (OneDrive-synced).
4. D33 is applied at the Phase 2 freeze; no hidden_gem place survives (D44).
5. Reddit credentials (D45); D24 scoping review; the ask-first emails in docs/outreach.md are unsent.
6. O13 (Phase 5): the 33-code airport allowlist data file and its test DO NOT EXIST yet.
7. ATTRIBUTION to credit in the report and app: Open-Meteo (CC BY 4.0), OpenStreetMap (ODbL), Wikimedia (CC BY-SA; Wikidata CC0).

## Phase 2: exact first steps
Plan mode first; wait for approval. Do not touch data/processed/chunks*.jsonl once `data/processed/FROZEN` exists. Read the paper notes in docs/papers/ if any apply.
1. Add the date-kind and hold-out handling to the contract through a docs/decisions.md entry first (contracts change only that way): chunk field `date_kind` = published | revised | none (O9). `BaseChunk` has no such field yet; the freshness share and every freshness metric count `published` only; null date stays neutral (D3).
2. Write per-place pure functions in `src/process/` (clean -> dedupe -> chunk -> tag) that take one place's raw records and return chunks, with no dependence on data/raw layout, so the live path (D20, `ensure_place`) can reuse them for a never-seen place. The batch job is a thin loop over data/raw.
3. Cleaning: strip boilerplate (inditales bodies start with share-button text "Facebook X Pinterest ..."); exact and near-duplicate removal (including the Ooty/Kodaikanal post stored under two places: keep one copy, assign the place by text or keep both under distinct chunk ids, decide and record); South India filter via the registry.
4. Hold-out drop (D39 iii, O8 amended): drop every chunk whose text names a hold-out (names and aliases from the registry; Kumily, Araku, Brihadisvara/Brihadeeswarar; "Periyar" only in context next to Thekkady, Kumily, Periyar Lake, Tiger Reserve, National Park or Wildlife Sanctuary). Report counts dropped per hold-out; the bare-Periyar and river/reformer cases must be tested. Hold-out hit counts today: 19 records in 18 raw files (LOG.md reconciliation).
5. Chunking: about 200 to 300 tokens with overlap; deterministic ids via `make_chunk_id(url, offset)`; carry `source_type` (wikipedia and sahapedia map to `reference`), url, date, date_kind. Do not change chunk size to chase the 10k target (D34).
6. Tests (pytest) for cleaning, dedupe, chunk boundaries and ids, date_kind, the hold-out drop; run `ruff check .` and `pytest -q`.
7. Freeze: apply D33/D23 ONCE to all 26 places (list the excluded places and why; keep their raw data and registry entries; the planner must not recommend a place without corpus content; no Phase 4 query may target an excluded place or a hold-out). Decide the Reddit question first (D45). Then write corpus statistics (chunks per source, per state, per year, null-date share, date_kind share), create `data/processed/FROZEN`, record numbers with their commands in LOG.md, update docs/phase-status.md, commit, tag `phase-2-done`.

## Note for Valincia (live layer, D31 proposed, not binding until you acknowledge)
docs/plan.md Section 2.1 and docs/contracts.md describe a live web layer (`live_retrieve`, `source_type: "live"`); the design is yours (search API, quota, terms, budgets, fusion). Reusable from Phase 1: HttpClient (src/common/http.py: robots check, throttle, request log, hard cap, allowlist), the disk cache (src/common/cache.py), the request log.
The frozen corpus has NO fresh content (D32): freshness depends only on your live layer. A small never-seen place such as Yercaud yields reference wikis only (D23 fails), so the live node cannot rely on fresh web content for such places. YouTube Data API per-request use is specified in D35 (terms UNVERIFIED).

## Known gaps and cautions
- The web collector silently drops pages under MIN_TEXT_CHARS (Kerala Tourism pages are client-rendered).
- Wikipedia and Wikivoyage records carry a revision date (`lastmod`) only, never a publish date.
- Phase 5 TODO: list handling for P31/P1435, the 33-code airport allowlist (O13), D42 elevation rules and confidence, hill-station climate values are relative only (ERA5 bias); travel_time to or from region places (Kodagu, Wayanad) is district-granular (D21).
- Hooks: the attribution Bash hook blocks commit text that merely names a trailer; keep core.hooksPath set. `cache.py` has no single-flight locking; `atomic_write_text` does not retry a Windows PermissionError.
- WORKING RULE (owner): a STOP rule means stop BEFORE acting. If a scan finds something in a tracked file or git history, stop and ask.
