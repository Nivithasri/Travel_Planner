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
| place_type | str | e.g. hill_station, beach, temple, heritage, wildlife, city |
| season_tags | list[str] | e.g. ["winter", "monsoon"] |
| aspect | enum | timing, closure, price, scenery, food, safety, history, other |
| is_warning | bool | |

## Place registry entry
name, aliases[list], state, region_cluster, popularity ("popular" | "hidden_gem"), lat, lon.

## Function contracts
```python
retrieve(query, filters=None, k=10) -> list[dict]   # chunk fields + "score"            (Phase 6)
travel_time(place_a, place_b) -> float              # hours                             (Phase 5)
check_season(place, month) -> dict                  # suitability + notes               (Phase 5)
get_place_card(place) -> dict                       # tips, warnings, citations         (Phase 7)
live_retrieve(query, places=None, k=5) -> list[dict]  # live chunks, source_type "live"    (Phase 7, Valincia; D31 PROPOSED, not binding until she acknowledges)
```

## Eval files
- eval/queries.json: [{id, text, split: dev|test, type, time_sensitive: bool, freshness_slice: bool}]
- eval/qrels.json: {query_id: {chunk_id: 0|1|2}}; also record which pooling round judged each chunk.

## Cache rules
- Cache key = sha256 of (kind, model/endpoint, canonical-JSON of inputs). Stored under .cache/ as JSON/JSONL.
- Writes are atomic (temp file + rename); resumable; a cache hit never calls the network.
- Cached: downloads, LLM calls (including expanded queries for eval), embeddings, fallback searches.

## Locations
data/raw/<source>/, data/processed/chunks.jsonl, data/enriched/chunks.jsonl, index/{bm25,dense_plain,dense_ctx}/,
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
