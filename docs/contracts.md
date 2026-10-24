# Contracts (source of truth; change only via docs/decisions.md)

## Chunk record (JSONL, one object per line)
| field | type | notes |
|---|---|---|
| chunk_id | str | stable, deterministic (hash of url + offset) |
| text | str | ~200-300 tokens, with overlap |
| context_note | str | LLM-written "where this fits" note; "" before Phase 3 |
| source_type | enum | reddit, blog, youtube, wikivoyage, official, reference (D22: Wikipedia, Sahapedia); `live` appears only in live results, never in the frozen corpus (D31, PROPOSED to Valincia, not binding until she acknowledges) |
| url | str | |
| date | str or null | YYYY-MM-DD; the page's PUBLISH date only, null otherwise (neutral freshness, never "old"; D48) |
| date_kind | enum | published, revised, sitemap_lastmod, none (D48, O9). published iff date is set; freshness counts published only |
| revised_date | str or null | YYYY-MM-DD wiki revision date from the API; set iff date_kind is revised (D48) |
| state | str | TN, KL, KA, AP, TS, PY (PY = Union Territory of Puducherry incl. Karaikal, Mahe, Yanam; D18) |
| place | str | must match a place registry name |
| place_type | str | the chunk's SUBJECT (D53), closed enum: hill_station, beach, temple, heritage, wildlife, backwater, waterfall, city, region, other. Not a property of `place` |
| season_tags | list[str] | closed enum subset (D53): winter (Dec-Feb), summer (Mar-May), monsoon (Jun-Sep), post_monsoon (Oct-Nov), all_year (exclusive) |
| aspect | enum | timing, closure, price, scenery, food, safety, history, other |
| is_warning | bool | strict bool |
| mentioned_places | list[str] | OPTIONAL, EnrichedChunk only (D50): other registry places the chunk mentions; non-hold-out registry names only; default []. PROPOSED to Valincia, not binding until she acknowledges |

## Place registry entry
name, aliases[list], state, region_cluster, popularity ("popular" | "hidden_gem"), lat, lon.

## Function contracts
```python
retrieve(query, filters=None, k=10) -> list[dict]   # chunk fields + "score"            (Phase 6)
travel_time(place_a, place_b) -> float              # hours; destinations and hubs; UnknownPlaceError (D81)  (Phase 5)
check_season(place, month) -> dict                  # suitability + notes + evidence; 26 destinations only; UnknownPlaceError (D81)  (Phase 5)
get_place_card(place) -> dict                       # tips, warnings, citations         (Phase 7)
live_retrieve(query, places=None, k=5) -> list[dict]  # live chunks, source_type "live"    (Phase 7, Valincia; D100; query must pass the closed-vocabulary check, UnsafeLiveQuery otherwise)
```

## retrieve() (Phase 6, D96)
`src.retrieval.retrieve.retrieve(query, filters=None, k=10)` returns chunk fields plus "score" from the FROZEN config (eval/frozen_config.json, D96 as amended): dense search over the contextual text (dense_contextual); no BM25, LLM, reranker or parse. HYBRID_CONFIG (RRF of BM25 and dense over the contextual text) is measured, not frozen. `retrieve_with_trace(query, filters, k, config=FROZEN_CONFIG, **parts)` adds a trace and accepts a RetrieveConfig (expansion, use_filters, fresh_w, rerank, mmr_lambda, corrective_tau) with an injected parser, scorer and rewriter; FULL_CONFIG is the measured full chain, not frozen. `filters` = {"state": [..], "season": .., "month": 1-12, "place_type": [..]}, unknown keys raise, failing chunks move behind passing ones (back-fill); not allowed with a reranked config.

## Eval files
- eval/queries.json: [{id, text, split: dev|test, type, time_sensitive: bool, freshness_slice: bool, place_scope: named|free, places: [registry names]}] (place_scope and places added by D70)
- eval/qrels_dev.json, eval/qrels_test.json: {query_id: {chunk_id: 0|1|2}} (D71; replaces eval/qrels.json). The pooling round and contributing systems are in eval/pool.json (ids only).
- eval/qrels_dev_r2.json, eval/qrels_test_r2.json (D91, Phase 6): same shape, the round-2 judgments of the pairs the Phase 6 variants surfaced that round 1 never pooled. eval.qrels.load_qrels_merged(split) unions round 1 and round 2; round 2 never overrides a round-1 label. The round-1 files qrels_dev.json/qrels_test.json are read-only and untouched. eval/pool2.json holds the round-2 pool (ids only, no text); sheets stay gitignored (O12).
- eval/parsed_queries.json (D86, Phase 6): per eval query id, the committed query-understanding output (structured fields + 3-5 sub-queries). Derived from query text only, no corpus text.

## Cache rules
- Cache key = sha256 of (kind, model/endpoint, canonical-JSON of inputs). Stored under .cache/ as JSON/JSONL.
- Writes are atomic (temp file + rename); resumable; a cache hit never calls the network.
- Cached: downloads, LLM calls (including expanded queries for eval), embeddings, fallback searches.

## Provenance sidecar (Phase 3; data/enriched/provenance.jsonl)
One JSON object per chunk (1,696 lines), joined to data/enriched/chunks.jsonl by `chunk_id`. The chunk schema is closed, so how a chunk's fields were produced lives here.
| field | type | notes |
|---|---|---|
| chunk_id | str | same ids as the enriched file |
| source | enum | llm (model-written), fallback (model output rejected; rule-based tags + template note), rule_only (never sent to the model) |
| reason | str or null | fallback: ungrounded, note_long, ...; rule_only: excluded_place (D33), blog_not_sent (D60) |
| model | str or null | generator model id for llm and fallback rows; null for rule_only |
| prompt_version | str or null | e.g. enrich-v4; null for rule_only |
Counts at the end of Phase 3: llm 1,087, fallback 38, rule_only 571 (288 excluded_place, 283 blog_not_sent). The file is gitignored (O12) and must be shared with the partner together with the
enriched chunks once O12 is resolved; its sha256 is in data/manifests/enriched_manifest.json. `mentioned_places` (D50) is NOT acknowledged by the partner and must not be used as a ranking or filter
signal until she acknowledges it. Retrieval must not apply aspect or place_type filters to rule_only or fallback rows without flagging it (D69).

## Locations
data/raw/<source>/, data/processed/chunks.jsonl, data/enriched/chunks.jsonl, index/{bm25_contextual,bm25_plain}/, index/{dense_plain,dense_contextual}.npy + index/ids.json (D51, D52, D55),
index/travel_matrix.json, index/place_cards.json, eval/, kaggle/.
- data/processed/ is frozen as a whole once data/processed/FROZEN exists (D48 x): nothing in it is written again.
- Phase 3 writes enriched chunks ONLY to data/enriched/chunks.jsonl (D48 xi), never to data/processed. The enriched file must have
  exactly the frozen chunk_id set, and per chunk_id identical chunk_id, text, url, date, date_kind, revised_date, state, place and
  source_type; it may only fill context_note, place_type, season_tags, aspect, is_warning (EnrichedChunk). Checked by
  src/process/process.py check_enriched_matches_frozen().

## On-demand ingestion (D17, Phase 1)
```python
ensure_place(place_text: str, budget: Budget | None = None) -> PlaceStatus   # Phase 1
```
- Resolves free text to a canonical place (registry + aliases, then Wikidata); rejects places outside TN/KL/KA/AP/TS/PY
  (D18). Ambiguous text returns candidates and ingests nothing.
- Fresh place: no network call. Stale: only expired sources are refreshed. New: collect from all sources.
- `budget` is a wall-clock deadline plus per-source caps; the call is resumable and may return status "ingesting".
- PlaceStatus: place, state, status (new|ingesting|collected|ready|stale|rejected), reason, candidates, per-source
  {ingested_at, expires_at, n_raw, n_requests, error, skipped}, chunk_counts (null until Phase 2), network_requests.
- `collected` = raw data collected and fresh, NOT yet chunked or indexed. `ready` is reserved for places whose
  chunks are indexed (Phases 2-3); Phase 1 never sets it.

## Raw record (data/raw/<source>/<place_slug>/*.jsonl, one object per line)
source (wikivoyage|wikipedia|sahapedia|reddit|youtube|blog|official; maps to chunk source_type: wikipedia and sahapedia -> reference), place, url, retrieved_at (ISO UTC), publish_date
(YYYY-MM-DD|null), lastmod (YYYY-MM-DD|null, never used as publish date), content_sha256, license, title, body,
extra (source-specific, e.g. revision id, score, video id).

## Facts raw files (D37)
data/raw/<wikidata|overpass|open_meteo>/<place_slug>/*.json: one JSON envelope per request (source, place, request_url, params, retrieved_at, status, from_cache, licence,
attribution, source_revision, response). Not records.jsonl and not part of the chunk corpus; the derived place_facts table is Phase 5.

## Live store
data/live/places/<slug>.json (status), data/live/registry.json (resolved places). gitignored. The snapshot registry
data/registry/core_places.json is committed and fixed: the scripts that built it were removed (D46, D47), so no code writes it; changing it needs a decisions.md entry. Committed manifest:
data/manifests/raw_manifest.jsonl (url, source, place, retrieved_at, sha256, bytes; no content).

## Routing and season data (Phase 5; D76 to D83)
- index/travel_matrix.json: {meta, nodes, hours, km}. meta: status PROVISIONAL, decision, attribution, params {band_km, road_factor, speed_short_kmh, speed_long_kmh, ghat_surcharge_h, n_fit_*},
  nodes, hubs, pairs, triangle_violations, airport_aliases {IATA: destination node id} (D83). nodes: [{id, name, kind: destination|hub, state (null for hubs), lat, lon, hill, aliases, iata? (hubs)}].
  hours/km: {"<idA>|<idB>": float}, ids sorted, each unordered pair once, both orders resolve. Model: km = great-circle x road_factor; plain hours = km / speed_short up to band_km, then band_km / speed_short
  + (km - band_km) / speed_long; hours = plain hours + ghat_surcharge_h per hill endpoint (D78 amended, D82). Hold-outs are not nodes. No chunk text.
- index/season_data.json: {meta, places}. meta: status, decision, attribution, thresholds, rainy_day_mm, aliases {place: [alias]} (junk aliases dropped, D83), places. places[name]: {confidence, climate_file, hill,
  warning_counts {model_written, fallback, rule_only}, nearest_airport {iata, name, km}, months: {"1".."12": {climate {tmax_c, tmin_c, rain_mm, rainy_days, years}, verdict, reasons, access_risk,
  evidence {model_written|fallback|rule_only: {chunks, season_tagged, all_year}}, warning_chunk_ids (model-written warnings only, at most 5, by chunk_id), warning_sources {chunk_id: "model_written"}}}}.
  Chunk ids only, no text. A rebuild needs the gitignored data/raw and data/enriched files; the committed files are what it produces (checked by a test when the data is present).
- travel_time(a, b) -> float hours: names, registry aliases, hub names/aliases/IATA, node ids, and airport codes in airport_aliases (COK, HYD, IXM, MAA, MYQ, PNY, TIR, WGC) which stand for the nearest destination.
- check_season returns: place, month, verdict, reasons, access_risk, climate, evidence (model_written | fallback | rule_only counts), warning_chunk_ids, warning_sources, confidence, nearest_airport, caveats.
- travel_km(a, b) is NOT reliable on hill legs: road km is great-circle km x one plain factor for every edge, and on the four hill routes with reference values it is 20 to 26% short (Coimbatore airport-Ooty -25.9%, Madurai-Kodaikanal -25.4%, Kochi-Munnar -23.9%, Hyderabad-Srisailam -20.2%; reference values UNVERIFIED). Use hours (travel_time) for planning and daily-travel limits; the ghat surcharge is in hours only (D78 amended).

## Phase 7 interfaces (D99 to D105)
- `get_place_card(place) -> dict` (src/tools/place_cards.py): reads index/place_cards.json (gitignored, built by scripts/build_place_cards.py). Fields: place, aliases, state, popularity, visibility_tercile (1 to 3, full-evidence places only), evidence_class (full | limited), hill, nearest_airport, best_months {good, fair, poor, access_risk, confidence, basis}, tips/warnings/highlights [{text, chunk_id, source_type, aspect, provenance, url}], counts, recency, caveats. Warnings are model-written only (D69). Unknown names, hubs and hold-outs raise UnknownPlaceError.
- `live_retrieve(query, places=None, k=5)` (src/live/search.py): live chunks as dicts of the chunk fields plus "score" (source_type "live", chunk_id "live:<hash>", fetch_date, date only if the provider gave one). Same-day cache only, minimal stored fields, gated by config/live.json (a real call needs terms_status "owner_accepted_risk", D109; the terms are never "verified"; the result cache is off; every live answer carries the notice "live search: terms not verified, owner accepted risk"). Never part of retrieve(), the corpus, the indexes or the qrels.
- Pipeline: `scripts/run_pipeline.py` runs parse -> (retrieve || live) -> merge_grade -> plan -> write -> critic -> finalize (src/agents/graph.py: LangGraph, with run_local as the sequential fallback). The answer object holds text, sources (alias, chunk id, live flag, fetch date), plan, critic report (passed | repaired | failed_rules), live status and warnings.


## Phase 9 files (D123 to D128)
- eval/phase9/requests.json: {meta: {status DRAFT|APPROVED, time_sensitive_min}, requests: [{id P01.., type, time_sensitive, text, named_places [registry names], overrides {duration_days, budget_tier, party, month, start_city, end_city?, interests [closed list], constraints, time_sensitive?}}]}. No run before status APPROVED (request_set.load_requests(require_approved=True)).
- Run record (.cache/phase9/<arm>/<request id>.json, gitignored: it holds answer text and excerpts): {system llm_only|plain_rag|full, arm llm_only|plain_rag|full_corpus|full_live, request_id, text, citations [{alias, chunk_id, place, source_type, url, live, fetch_date, cited}], contexts [{alias, chunk_id, place, source_type, url, text}] (corpus items only, never live snippet text), tokens, timings, live {status, searches, results, in_scope, in_plan, dropped_excluded, merged, cited, not_cited, note} or null, critic, meta {git_sha, dirty, model, utc_date, ts, prompt_sha, ...}}.
- eval/sheets/phase9/<set>_sheet.csv + <set>_key.json (gitignored): blind rows R01.. (request, output with [S#]/[L#] markers, sources, four scores 1 to 5, note); the key holds request id, system, arm and a sha256 of the fixed columns. Citation sheet: C01.. (claim, passages, supported 0/1).
- eval/phase9/results/*.json and summary.md (committed): scores per request and system, paired comparisons (mean difference, bootstrap CI, sign-flip p, Holm), agreement, citation precision (corpus rate, live counts), lean judge means, live counts, latency table, case-candidate ids. Scores, counts and ids only: no answer text, no excerpt, no snippet (D100, O12).
