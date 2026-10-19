# Facts layer: design and terms (amended 2026-10-18; fetchers built under D37, no place_facts table yet)

Scope: structured, source-cited facts (coordinates, elevation, climate normals, transport features) that prose chunks cannot give
reliably, for `check_season` and the planner. Phase 1 stores only raw JSON envelopes under `data/raw/<source>/<place>/`; the derived
`place_facts` table is built in Phase 5. No value is written by a model; every value carries a `source_url`.
Terms were read through the web-fetch tool on 2026-10-18 (one request per URL). A quote is text as returned by the fetch tool and under
15 words; UNVERIFIED means the page could not be read, was empty, or only a model summary exists. Not legal advice.
Robots handling follows D36 (API-host exception, owner choice). The hosts are allowlisted by D37 in `config/sources.json` (`api_hosts`).

## 1. Findings that needed the owner
1. **Open-Meteo free tier: OWNER-RESOLVED.** The terms page says: "You may only use the free API services for non-commercial purposes." The pricing
   page FAQ, as returned by the fetch tool, said: "Historical, climate, ensemble, and satellite radiation APIs require the Professional API Plan or higher."
   The owner has read the page and states that the free tier covers the Historical and Climate APIs for non-commercial use. This is the owner's reading,
   not independently verified here. Caching rule UNVERIFIED (no statement found). Attribution: CC-BY 4.0 ("You accept to the CC-BY 4.0 licence").
2. **Robots.txt** (as returned by the fetch tool): wikidata.org, en.wikipedia.org, en.wikivoyage.org `Disallow: /w/` (mobileview allowed); overpass-api.de
   `Disallow: /api/`; archive-api.open-meteo.com and climate-api.open-meteo.com `Disallow: /`. D36 lets documented API endpoints skip the check.
3. **OpenRouteService** terms and plans were unreadable (account page, no text): UNVERIFIED.
4. **ODbL** share-alike applies to a publicly used database derived from OSM (section 3.5, O12).

## 2. Sources
Common rules: HttpClient (src/common/http.py) with the contact User-Agent from `.env`, the disk cache, the request log, the per-run cap (config
`api_hosts[].max_requests_per_run` and `run_limits`), and the circuit breaker (D38). A 403, 406 or 429 stops the run (BlockDetected), no retry.
A second run makes 0 requests. Raw envelope: JSON with `source, place, request_url, params, retrieved_at, status, from_cache, licence, attribution,
source_revision, response`.

### 2.1 Wikidata (entity data)
- Terms: https://www.wikidata.org/wiki/Wikidata:Licensing , 2026-10-18: "Wikidata requires a CC0 license, which is equivalent to public domain;"
- Access: https://www.wikidata.org/wiki/Wikidata:Data_access , 2026-10-18: Special:EntityData/Q42.json "gives you Item Q42 in JSON format."; no numeric limit
  on that page; "stop sending further requests for a while" after a 429; use `maxlag` for the Action API. Etiquette (https://www.mediawiki.org/wiki/API:Etiquette):
  "Make your requests in series rather than in parallel". "API requests to Wikimedia wikis must include a meaningful User-Agent header." The foundation
  User-Agent policy page was not read: UNVERIFIED.
- Attribution: none required (CC0). Storage and redistribution: allowed. Share-alike: none. Non-commercial limit: none.
- robots.txt: exists; `Disallow: /w/` and, as returned by the fetch tool, a Disallow plus a wildcard Allow for `/wiki/Special:EntityData/`. The run prints the Python parser's verdict
  (False, a parser artifact: it ignores `*`); under RFC 9309 matching the URL is allowed by the Allow on line 436 (D41).
- **RFC 9309 check (D41):** applied to the cached wikidata.org robots.txt, the matcher allows `/wiki/Special:EntityData/Q1361.json` (rule `Allow: /wiki/Special:EntityData/*.`,
  line 436, longest match). Python's parser said False only because it ignores `*`. The D36 exception was therefore not needed for this path on that file copy.
  It remains in force for Overpass and Open-Meteo, whose robots.txt files were not checked with the matcher.
- Request budget: 1 per place via Special:EntityData (the registry holds `wikidata_id` for every entry): 26 for the 26 collectable places, 0 for hold-outs.
- Fields (raw envelope holds the whole entity): P625 coordinates, P2044 elevation, P131 district, P1435 heritage status, P1082 population, P31 instance of.
- **as_of rule (amended):** a fact's `as_of` is the statement's P585 (point in time) qualifier if present, else null. The entity's revision id and modified
  time are stored in `source_revision`, not in `as_of`. **Multi-valued statements (rule in code, `select_statement`):** deprecated-rank statements are ignored; if any
  statement has preferred rank, choose among those, otherwise among normal-rank statements; among the candidates choose the one with the latest P585; if none
  has a P585, take the first listed and mark the choice `ambiguous`.
- Allowlist entry (D37): host www.wikidata.org, min delay 1 s, cap 30 requests per run.

### 2.2 Open-Meteo, archive (ERA5) API only (monthly normals derived by us)
- Terms: https://open-meteo.com/en/terms , 2026-10-18: "You may only use the free API services for non-commercial purposes." Limits: "Less than 10'000 API calls
  per day, 5'000 per hour and 600 per minute." Licence: "You accept to the CC-BY 4.0 licence".
- Pricing page, 2026-10-18: "The free API is for non-commercial use, rate-limited to 10,000 calls/day". Weighting: requests over "a period of more than 2 weeks" count as
  multiple calls (the full rule is longer than 15 words; the page's example: 4 weeks of data equals 3.0 calls). A 10-year daily window therefore weighs many calls;
  the real weight per request is UNVERIFIED until one response is seen. See finding 1 for the plan question (owner-resolved).
- Endpoint: https://archive-api.open-meteo.com/v1/archive ("You can access data dating back to 1940."; "based on reanalysis datasets"). The CMIP6 climate endpoint is NOT used.
- Request: latitude/longitude from the registry, start_date 2015-01-01, end_date 2024-12-31, daily = temperature_2m_max, temperature_2m_min, precipitation_sum
  (the three variables the docs page lists), timezone Asia/Kolkata, `elevation` = Wikidata P2044 when present (otherwise omitted and recorded). One place per request,
  sequential, min delay 2 s. Mean temperature = (max + min) / 2 computed in code and labelled derived.
- Rainy-day rule (code, a named constant): a day is rainy when precipitation_sum >= 2.5 mm. The threshold follows the IMD convention as I recall it and is UNVERIFIED.
  Monthly figures: mean of monthly totals over the 10 years, rainy days per month averaged over the 10 years.
- Attribution: CC-BY 4.0, plus a reference to Open-Meteo; exact wording for the archive API UNVERIFIED. Storage and caching: no statement found, UNVERIFIED. Share-alike: none stated.
- robots.txt: `Disallow: /` on the archive host. Covered by D36/D37.
- Confidence and method: `medium`, retrieval_method `reanalysis`. ERA5 has a coarse grid; rainfall and temperature at hill stations (Munnar, Ooty, Kodaikanal, Kodagu)
  are biased by the orographic effect and the grid-cell elevation, so the elevation parameter helps temperature but not rainfall. Stated in every derived row.
- as_of: the data window (2015-01-01 to 2024-12-31), not a publish date.
- Request budget when approved: 1 place first to read the real call weight, then 25 more; 26 requests. Allowlist entry (D37): host archive-api.open-meteo.com, min delay 2 s, cap 30.

### 2.2a Ooty first result (2026-10-19; numbers computed from the saved envelope, not claimed correct)
Mean annual precipitation 2015-2024: 1849.3 mm. January: mean Tmax 18.6 C, mean Tmin 8.7 C. May: mean Tmax 20.9 C, mean Tmin 13.8 C. Wettest month: October (267.0 mm mean).
Note: ERA5 on a coarse grid; hill-station absolute values likely biased (rain high, daytime max low); reference normals not fetched. Use for relative seasonality only; confidence low for hill stations.

### 2.3 OpenStreetMap via Overpass (transport features)
- Terms: https://www.openstreetmap.org/copyright , 2026-10-18: "as long as you credit OpenStreetMap and its contributors"; ODbL 1.0.
- Usage policy (https://wiki.openstreetmap.org/wiki/Overpass_API): "less than 10,000 queries per day"; "download less than 1 GB data per day"; "no more than 10 minutes"; used
  regularly, "divide those numbers by 100"; "pause for 30 seconds before making a new request" after a 429 or 406; send a User-Agent or Referer; "No parallel running of multiple scripts".
  The `around` filter section lay beyond the fetched text: UNVERIFIED (standard Overpass QL, to be checked on first use).
- Attribution: https://osmfoundation.org/wiki/Licence/Attribution_Guidelines : "Attribution must be to “OpenStreetMap”." and the ODbL must be named.
- Storage allowed. Share-alike for a publicly used Derivative Database (section 3.5).
- robots.txt (overpass-api.de): `Disallow: /api/`, `Disallow: /munin/`. Covered by D36/D37.
- **Features (D40, amended again):** AIRPORTS ONLY. ONE query for `aeroway=aerodrome` with an `iata` tag over the south bounding box (lat 8.0 to 20.0,
  lon 74.0 to 85.0), stored once, unfiltered; the nearest airport per place is computed in code (haversine) after the hold-out/10 km filter. The per-place
  railway-station and bus-station queries (26) were REMOVED from code and docs: not in the plan, ODbL share-alike cost on a larger derived database, O12.
  Railheads and bus links come from the Phase 5 travel matrix. temple_count and viewpoint_count stay dropped.
- Read timeout: 90 s for this host (above the server's own `[timeout:60]`); the first try with the 30 s default timed out. One request, no retry.
- Request budget: 1 query (done 2026-10-19: 48 elements). Allowlist entry (D37): host overpass-api.de, path /api/interpreter, min delay 30 s, cap 30.
- as_of: the query time (fetch date). valid_until: fetch date + 365 days.

### 2.4 Nominatim: NOT USED
Reason, from https://operations.osmfoundation.org/policies/nominatim/ (2026-10-18): "No heavy uses (an absolute maximum of 1 request per second)." Bulk geocoding must be
"a single thread" with no distributed scripts; "Results must be cached on your side."; systematic queries are listed as forbidden. Coordinates come from Wikidata already.

### 2.5 Travel times (Phase 5 options only; nothing chosen)
| option | evidence | status |
|---|---|---|
| OSRM public demo server | https://github.com/Project-OSRM/osrm-backend/wiki/Demo-server : "Do not exceed 1 request per second." and use "restricted to reasonable, non-commercial use-cases." | DO NOT CALL (owner rule) |
| OSRM local on a Geofabrik extract | https://download.geofabrik.de/asia/india/southern-zone.html : Southern Zone .osm.pbf 532 MB, "License: ODbL 1.0"; root page: "This open data download service is offered free of charge by Geofabrik GmbH." No usage statement found; technical.html and robots.txt not read: UNVERIFIED. A bulk download needs the owner's approval | option, needs approval |
| OpenRouteService free tier | terms and plans pages redirect to an account page with no text | UNVERIFIED |
| Distance x calibrated road factor + ghat correction (O6) | no external service; calibrated on about 30 known routes | option |

### 2.6 Wikipedia and Wikivoyage attribution (CC BY-SA)
- Wikivoyage Copyleft (https://en.wikivoyage.org/wiki/Wikivoyage:Copyleft , 2026-10-18): "you must give attribution to the creators".
- Wikimedia Terms of Use section 7: exact wording UNVERIFIED (read only up to before section 7; a model summary says credit by a link to the page, a stable copy, or a list of authors,
  and name the licence). Re-read the live page before relying on it.
- Fields kept per raw record: page URL, title, revision id, retrieval time, licence string, a link to the page history. ShareAlike: derived text stays CC BY-SA (O12).
- robots.txt: `Disallow: /w/` (mobileview allowed); the existing /w/api.php calls rely on the same API-host reading (D36).

## 3. Design

### 3.1 place_facts record (built in Phase 5; Phase 1 stores only raw envelopes)
| field | notes |
|---|---|
| place | registry name; never a hold-out |
| fact_type | coordinates, elevation_m, district, heritage_status, population, instance_of, rainfall_mm_month, rainy_days_month, temp_c_month, nearest_airport |
| value, unit | as extracted; no unit conversion without a stated rule |
| as_of | Wikidata: P585 qualifier or null; Open-Meteo: data window; OSM: query date |
| source_revision | Wikidata entity revision id and modified time |
| valid_until | null for static facts; fetch date + 365 days for OSM features |
| source_url | MANDATORY (Special:EntityData URL; Overpass request URL with the query hash stored in the envelope; archive request URL). A row without it is rejected |
| source_type | `wikidata`, `osm`, `open_meteo` (facts only; not chunk source types) |
| retrieval_method | api, or reanalysis for Open-Meteo |
| confidence | high for a direct Wikidata field; medium for a derived value or reanalysis; low for a name match |
| license, attribution | CC0 / ODbL + "OpenStreetMap" / CC-BY |
Derivations are code, one function per fact_type. No model writes, rewrites or fills a value.

### 3.2 Hold-outs
Hold-outs (Chikmagalur, Araku Valley, Thekkady, Thanjavur) are excluded from facts and from OSM queries: no request is centred on a hold-out. **Drop list for names (amended):**
Thanjavur, Tanjore, Thanjai; Chikmagalur, Chikkamagaluru; Araku Valley, Araku; Thekkady, Kumily; Brihadisvara, Brihadeeswarar; and "Periyar" only by the O8 context rule (beside Thekkady,
Kumily, Periyar Lake, Periyar Tiger Reserve, National Park or Wildlife Sanctuary). Any OSM feature whose name tags (`name`, `name:en`, `alt_name`, `old_name`) match is dropped before storing the
derived list. **Distance rule (amended):** any feature within 10 km of a hold-out centre (coordinates from data/registry/core_places.json) is also dropped.
Nearest hold-out per place (great-circle): Munnar 63.7 km (Thekkady), Kodaikanal 83.8, Visakhapatnam 84.1 (Araku Valley), Alappuzha 94.5, Kodagu 99.7 (Chikmagalur), Varkala 103.6, Madurai 109.6,
Kochi 111.4, Mysuru 146.5, Pondicherry 147.8. The raw airport envelope keeps the whole response (unfiltered); the drop happens when derived lists are made. First live result: 48 elements, 1 dropped by name (Thanjavur Air Force Station), 0 by the 10 km rule.

### 3.3 How the facts are used
- `check_season(place, month)`: monthly rainfall, rainy days and temperature normals (when Open-Meteo is run) plus elevation; returns the data window as `as_of`; a missing month is "unknown".
- Planner: nearest airport from the OSM airport list (deduplicated by IATA: the raw list has several elements for HYD, PYB, VDY); railheads and bus links from the Phase 5 travel matrix.
- No fee data comes from this layer.

### 3.4 Requests, cache, log
Wikidata 26 (done), Overpass 1 (done), Open-Meteo 26 (8 done, 18 left): 53 requests if all run, each source as its own run, under the 250 cap. Every call goes through HttpClient (cache first,
request-log line, bounded retries), the envelope records the request URL, time and status. A rerun makes 0 requests. Any 403, 406 or 429 stops the run.

### 3.5 O12 and the ODbL
OSM-derived features stored as place_facts form a database derived from OSM. ODbL text (https://opendatacommons.org/licenses/odbl/1-0/, 2026-10-18): "Any Derivative Database that You Publicly Use must be only under the terms of:"
(section 4.4a); a Produced Work "does not create a Derivative Database for purposes of Section 4.4" (4.5b) but a publicly used one needs a notice ("Contains information from DATABASE NAME ..."). Open for the owner under O12:
(1) whether sharing the facts file with the partner or uploading it to Kaggle is "public use" (UNVERIFIED reading; ask before sharing); (2) the app shows the OpenStreetMap attribution when OSM facts are used;
(3) keep OSM-derived rows in a separate file (data/processed/place_facts_osm.jsonl). Until decided, nothing derived from OSM leaves the machine.

## 4. Decisions
D36: API-host robots exception. D37: `api_hosts` allowlist entries for www.wikidata.org, overpass-api.de and archive-api.open-meteo.com, and the raw facts JSON files (docs/decisions.md).

## 5. Results so far (2026-10-19)
- Wikidata: 26 of 26 envelopes. Open-Meteo: 26 of 26 (each 3,653 days, 0 nulls, HTTP 200; Auroville requested without an elevation parameter). The first resume attempt
  stopped on a read-type timeout at Mysuru (request-log note transport:timeout); the second attempt, 360 s spacing and a 60 s read timeout, completed all 26 (18 network requests, 8 from cache).
- Observation: Wikidata P2044 for Gokarna is 586 m, identical to Badami; Gokarna is coastal, so that value looks wrong (UNVERIFIED) and the request used it.
- No rate-limit header or call weight came back from Open-Meteo, so the real weight is still unknown.
- Overpass: the airport query succeeded on the second try with the 90 s read timeout.

### 5a. Robots evidence and elevation audit (2026-10-18)
- Robots, RFC 9309 matcher on the fetched files: overpass-api.de `Disallow: /api/` (line 2) disallows /api/interpreter; archive-api.open-meteo.com `Disallow: /` (line 2) disallows /v1/archive. The D36 exception applies to both. Terms: UNVERIFIED.
- Gokarna: the ORIGINAL Open-Meteo envelope (elevation 586 m from Wikidata P2044, identical to Badami) is SUSPECT. A second envelope without the elevation parameter is saved beside it (archive_2015_2024_noelev.json; API-echoed
  elevation 2.0 m). Side by side: mean annual precipitation 2991.7 vs 2983.7 mm; January Tmax/Tmin 27.8/17.3 vs 31.5/21.1 C; May 28.5/23.0 vs 32.2/26.8 C. No claim is made about which is correct.
- Wikidata elevations (P2044) for region or centroid places (Kodagu, Wayanad are registry place_type `region`; other place_type values are unset) are UNVERIFIED. Phase 5 must sanity-check every elevation (identical values for
  different places, coastal towns with hill-level values, centroids) before using it as an input or a fact.

### 5b. No-elevation comparison and confidence rules (2026-10-19, D42)
| place | elevation sent in the original | API-echoed elevation without the parameter | precip mm/yr (orig / no-elev) | Jan Tmax/Tmin (orig / no-elev) | May Tmax/Tmin (orig / no-elev) | primary file |
|---|---:|---:|---|---|---|---|
| Gokarna | 586 m | 2.0 m | 2991.7 / 2983.7 | 27.8/17.3 / 31.5/21.1 | 28.5/23.0 / 32.2/26.8 | no-elevation (original SUPERSEDED) |
| Kanniyakumari | 300 m | 36.0 m | 1126.1 / 1126.1 | 29.2/21.9 / 31.0/23.7 | 29.8/24.2 / 31.6/26.0 | no-elevation (original SUPERSEDED, D42 amended) |
| Varkala | 190 m | 46.0 m | 2421.7 / 2421.7 | 29.7/22.3 / 30.8/23.4 | 28.8/24.2 / 29.9/25.3 | no-elevation (original SUPERSEDED, D42 amended) |
No claim is made about which values are correct. Confidence for every Open-Meteo-derived value: `low` for Kodagu, Wayanad (centroid regions) and the hill stations (Munnar, Ooty, Kodaikanal); `medium` otherwise.
Terms for Overpass and Open-Meteo are UNVERIFIED (the owner will paste quotes). The wiki robots.txt files are not cached and were not checked (D36 amendment).

### 5c. Terms record and sources of API-use rules (as returned by the fetch tool in the review chat, 2026-10-19; owner to verify against the live page)
- Open-Meteo: quote "You may only use the free API services for non-commercial purposes." Limits: under 10,000 calls per day, 5,000 per hour, 600 per minute. Licence CC-BY 4.0. Non-commercial examples include
  public research at public institutions and educational content. The page states no call weight and no caching rule (both UNVERIFIED).
- Overpass wiki policy: quote "No parallel running of multiple scripts". Limits: under 10,000 queries per day and 1 GB per day; a User-Agent or Referer that identifies the app; pause 30 s after 429 or 406. The page does not
  mention ODbL attribution (O12 stays open).
- Wikimedia: the cached www.wikidata.org robots.txt is a proxy only. en.wikipedia.org and en.wikivoyage.org robots.txt were never fetched, so they are UNVERIFIED; the exception for their API calls rests on D36 plus the
  Phase 2 owner action on the Wikimedia API terms.
- D42 amended: Kanniyakumari and Varkala no-elevation files are primary (originals superseded, not deleted); Gokarna, Kanniyakumari, Varkala confidence medium; refetch only if the likely elevation error exceeds about 150 m.

### 5d. Secondary terms evidence (as relayed through the review chat, 2026-10-19; owner to verify against the live pages)
- Wikimedia REST API (pageviews host wikimedia.org): quote "Limit your clients to no more than 200 requests/s to this API."; a unique User-Agent that allows contact is required; the Wikimedia Terms of Use apply (a mirror of the mediawiki.org page, not the original).
- Wikidata Query Service (query.wikidata.org): SECONDARY source only (R package documentation quoting the manual; the manual itself was not read): quote "One client is allowed 30 error queries per minute"; 60 s hard query deadline; one client is allowed 60 s of processing per 60 s.
- These two hosts now have secondary evidence. config/sources.json terms_status for them is unchanged (UNVERIFIED) until the owner pastes primary quotes (D43).
