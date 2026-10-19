# Richer-corpus research (DESIGN AND RESEARCH ONLY; nothing collected, built or decided)

Date 2026-10-18. Every figure is an ESTIMATE with its assumptions stated. Licence text was read through the web-fetch tool on
2026-10-18; a quote is verbatim from the fetched text and under 15 words. Where a page could not be read, or only a summary or
search snippet exists, the cell says UNVERIFIED and has no quote. Not legal advice. Nothing was downloaded into the project;
no dump was fetched. Hold-outs (Chikmagalur, Araku Valley, Thekkady, Thanjavur, D17) are excluded everywhere below.

## D1. What the corpus has today (coverage_matrix.py (deleted, D47), read-only, keyword aspect tags = estimate only)
Command: `python coverage_matrix.py (deleted, D47)`. 9 places collected, 17 not collected, 4 hold-outs. Cleaned characters (strict
level) per place and source type, from the run on 2026-10-18:

| place | blog | reference (Wikipedia, Sahapedia) | wikivoyage | publish-dated records | newest publish date |
|---|---|---|---|---|---|
| Auroville | - | 1 (15,115) | 1 (15,420) | 0 of 2 | none |
| Chennai | 1 (2,828) | 1 (50,501) | 1 (51,947) | 1 of 3 | 2015-03-07 |
| Hampi | 2 (15,683) | 1 (42,024) | 1 (34,541) | 2 of 4 | 2012-01-06 |
| Hyderabad | 8 (41,777) | 1 (58,631) | 1 (46,430) | 8 of 10 | 2011-12-26 |
| Kodagu | 5 (45,717) | 1 (12,437) | 1 (14,124) | 5 of 7 | 2022-04-28 |
| Munnar | - | 1 (7,938) | 1 (34,305) | 0 of 2 | none |
| Pondicherry | 1 (3,181) | 1 (16,650) | 1 (32,470) | 1 of 3 | 2005-09-27 |
| Tirupati | 1 (9,789) | 1 (30,848) | 1 (12,376) | 1 of 3 | 2024-11-21 |
| Wayanad | - | 2 (47,169) | 1 (8,531) | 0 of 3 | none |

Reddit, official and youtube columns are empty for every place (official 0 records, Reddit not approved, YouTube off by D27/D30).
Aspect coverage (records with a keyword hit): timing 0 to 2 per place (Auroville, Hampi, Wayanad: 0), closure 1 to 3, price 1 to 3,
safety 0 to 2 (Auroville 0), the stable aspects (scenery, history, transport, season) 2 to 8. The time-sensitive aspects are the
thinnest ones, as expected. 6 of 9 places have at least one publish-dated record; 0 of 9 have one inside the 18 months before
EVAL_REFERENCE_DATE 2026-10-18 (window starts 2025-04-18).

**Proposed per-cell target N (for the 26 collectable places; a proposal, not adopted):**
- Reference family: at least 1 Wikipedia and 1 Wikivoyage record per place (already met for the 9).
- Traveler voice (blog or later reddit): at least 3 records per place, at least 2 with a publish date; none of the 9 places except
  Hyderabad and Kodagu meets this today.
- Time-sensitive aspects (timing, closure, price, safety): at least 3 dated records per aspect for the places that appear in the
  freshness queries, at least 1 for the rest. Today the most any place has is 2 (keyword estimate).
- Stable aspects (scenery, food, history, transport, season): at least 3 records per place (most places already meet this).

## D2. Source candidates
Columns: licence or terms with the URL and date looked up (2026-10-18) and a quote; attribution; storage and redistribution; expected
yield per place (estimate); date type; aspect cells it fills. "Yield" assumptions are in D7.

| # | Source | Licence / terms evidence | Attribution | Store / redistribute | Yield per place (estimate) | Date type | Aspects filled |
|---|---|---|---|---|---|---|---|
| 1 | Wikipedia and Wikivoyage linked articles (sights, temples, parks, districts) | Wikivoyage https://en.wikivoyage.org/wiki/Wikivoyage:Copyleft : "you must give attribution to the creators". Wikipedia https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use : "Share and Reuse our articles and other media under free and open licenses." The CC BY-SA 4.0 and GFDL detail for Wikipedia was only in a model summary: UNVERIFIED verbatim. | Credit authors via the page history link | Copying and redistribution allowed under CC BY-SA 4.0 with ShareAlike (Wikivoyage page, fetched); the Wikipedia terms page has no statement on caching | 4 to 12 chunks per article, cap 8 articles: +32 to +96 per place | Revision date only (O9) | history, scenery, transport, season, food; not timing, closure, price (weak) |
| 2 | OpenStreetMap via Overpass (opening_hours, fee, wheelchair tags for POIs) | https://www.openstreetmap.org/copyright : "as long as you credit OpenStreetMap and its contributors". Share-alike: "you may distribute the result only under the same license" (truncated quote, ODbL 1.0 per the page). Overpass https://wiki.openstreetmap.org/wiki/Overpass_API : "less than 10,000 queries per day" (public instance; caching encouraged) | Required, display notice; ODbL | Storage allowed; derived databases must stay ODbL | 10 to 60 tagged POIs per place, of which perhaps 20% carry opening_hours or fee: a few structured facts, not prose | Per-object edit timestamps exist but are not a publish date (UNVERIFIED how to read them) | timing and price (facts only, as place_facts, D3) |
| 3 | Wikidata | https://www.wikidata.org/wiki/Wikidata:Licensing : "Wikidata requires a CC0 license, which is equivalent to public domain;" | None required | Allowed | 5 to 15 properties (coordinates, elevation, inception, heritage status); already used for the registry | Revision date | history, transport (facts) |
| 4 | Open-Meteo (historical climate API) | https://open-meteo.com/en/terms : "You may only use the free API services for non-commercial purposes." and "Less than 10'000 API calls per day, 5'000 per hour and 600 per minute." The licence page https://open-meteo.com/en/license returned no text: attribution wording UNVERIFIED | UNVERIFIED (terms mention CC-BY 4.0 for paid data) | UNVERIFIED | 12 monthly rows per place (temperature, rainfall) as facts | Reanalysis, not a publish date | season (best-time, monsoon facts for check_season) |
| 5 | IMD climate normals | https://mausam.imd.gov.in/responsive/disclaimer.php : "intended for information to the general public based on the real time data". No reuse licence stated on the pages read; the data portal https://dsp.imdpune.gov.in/ was NOT fetched: UNVERIFIED | None stated | Not stated: treat as not allowed until asked | 12 monthly normals per station, if obtainable | Normals period, not a date | season |
| 6 | Travel Stack Exchange dump | https://archive.org/details/stackexchange : "All user content contributed to the Stack Exchange network is cc-by-sa 4.0 licensed" ; that page lists travel.stackexchange.com (summary part, not verbatim). Stack Overflow terms page could not be fetched: UNVERIFIED | Required: source, link to question, author names, link to profiles (four duties, per the page summary) | CC BY-SA 4.0 allows copy with attribution and ShareAlike. A dump is a bulk download; none was made | India-specific posts are a small fraction: perhaps 0 to 20 per place (guess, unmeasured) | Real post dates (publish) | safety, transport, price, closure (Q and A) |
| 7 | data.gov.in tourism datasets | https://data.gov.in/government-open-data-license-india , /terms-and-conditions and the tourism search returned HTTP 403 to the fetch tool: UNVERIFIED, no dataset titles read | UNVERIFIED | UNVERIFIED | Mostly state-level arrivals counts (guess); little text | Dataset year | price, season (statistics only) |
| 8 | More inditales posts | https://inditales.com/robots.txt : empty `Disallow:` and a `Sitemap:` line (fetched). Sitemap index https://inditales.com/sitemap_index.xml lists post-sitemap.xml, post-sitemap2.xml, page-sitemap.xml, post_format-sitemap.xml: NO category or tag sitemap exists. No terms page; footer "Copyright 2004-2021" then "Anuradha Goyal" (evidence file). Owner accepted the risk 2026-10-18 | Cite URL and author; no licence | Local only (O12) | Only broader slug matching is possible (post sitemaps, 2 files): +0.5 to 2 posts per place, +5 to +20 chunks | Real publish date (2005 to 2024 so far; none in the last 18 months) | food, history, scenery; timing and price only where a post gives them |
| 9 | Other open-licence travel blogs | Search on 2026-10-18 found no CC BY India travel blog (search results were guidebook shops): UNVERIFIED that none exists. Rejects already recorded in docs/sources.md: Tripoto, Holidify, Thrillophilia, Lonely Planet, TripAdvisor, Nativeplanet, global-gallivanting, indiasomeday, theblogchatter, millispotter (terms forbid copying or rules unreadable); ianslife.in and travelogues-of-india.com (robots unreachable). Still ask-first: travelindiasmart.com, lakshmisharath.com (Crawl-delay 20 s), bhushavali.com, wanderingfeet.in | n/a | n/a | 0 until an author says yes | Publish | traveler voice |
| 10 | PIB and state tourism news pages | https://pib.gov.in/PressReleasePage.aspx returned HTTP 403: UNVERIFIED. https://www.keralatourism.org/ (fetched, HTTP 200): "What's New" links and an Events page; the "Latest News" items showed only "Loading..."; no advisories section; no dates in the text. Kerala terms: none found (sources.md) | UNVERIFIED | Not covered | 0 to 3 notices per place | Often undated in the page text | closure, timing (when dated) |
| 11 | Temple notice pages (Tirumala, hrce) | https://tirumala.org/ (fetched 2026-10-18, HTTP 200): home page shows a dated "Day Schedules" block, darshan slots and a "Latest Updates" table with dated entries; news is on http://news.tirumala.org/ (different host, not allowed in config). Robots allow the home page; no terms found; risk accepted. https://hrce.tn.gov.in/ returned HTTP 200 but only the text "HR-CE": nothing readable, no dates | None stated | Not covered (O12) | Tirupati only: a few dated notices per fetch, but they change daily and would be live content, not frozen; hrce: 0 | Page-stated dates (not a publish date of an article) | timing, closure (Tirupati) |
| 12 | Reddit archives (Pushshift-era dumps, community archives) | A search on 2026-10-18 (snippets only, no page read): public Pushshift access reportedly ended in 2023; a community archive listing says no licence specified. UNVERIFIED. No download. The project's own Reddit collector (official API, credentials) is still pending approval | UNVERIFIED | UNVERIFIED; treat as not allowed | n/a | Real post dates | traveler voice, safety, closure, price |

**Permission email drafts (docs/outreach.md, unsent, for the owner's review).** The ASI draft named a hold-out place; that
phrase was removed on 2026-10-18 and the drafts now name no hold-out.

## D3. place_facts design (design only; no code, no file)

> SUPERSEDED IN PART by D32 (decisions.md): curated notes are not possible, so the `curated` retrieval methods and the `curated`
> source_type question below are moot. The full facts-layer design (API-sourced facts only) is in docs/facts-layer.md.
Purpose: small structured values that chunks of prose cannot give reliably (opening hours, fees, best months), used by check_season
and the budget estimate.

Rules: no value without `source_url`; no value written by an LLM; every value copied or derived by a stated rule from a stored
source; hold-out places and their aliases (Thanjavur: Tanjore, Thanjai; Chikmagalur: Chikkamagaluru; Araku Valley, Thekkady; the
"Periyar" context rule of O8) never appear in facts, as for raw data.

| field | type | notes |
|---|---|---|
| place | str | registry name |
| fact_type | enum | opening_hours, entry_fee, best_months, closed_days, altitude, rainfall_month, temperature_month, transport_note |
| value | str or number | as given by the source |
| unit | str | INR, hours, mm, degC, month, ... |
| as_of | date | when the source said it was true or when we observed it |
| valid_until | date or null | when to stop trusting it; mandatory for fees and hours |
| source_url | str | mandatory; a curated note uses the notice URL or the evidence id |
| source_type | enum | same enum as chunks, plus the open question below |
| retrieval_method | enum | api, page_text_rule, curated_visit, curated_call, official_notice |
| confidence | enum | high, medium, low (rule: high only with official notice or direct observation) |
| curator | str or null | initials of the human, null for API values |

Use: `check_season(place, month)` reads best_months, rainfall_month and temperature_month; if a fact is expired (`valid_until` in
the past) it says "unverified" rather than "ok". The budget estimate reads entry_fee and transport_note and marks every number with
its `as_of`. The critic treats a fact without source_url as invalid.

**Decision needed (no change made): the chunk `source_type` "curated".** Curated notes and place_facts rows are neither
reddit, blog, youtube, wikivoyage, official nor reference. Options: (a) add `curated` to the chunk enum by a decisions entry, with
diversity metrics counting it as its own type (as D22 did for reference); (b) map curated notes to `official` only when they quote an
official notice, which would corrupt source diversity; (c) keep curated notes out of the frozen chunk corpus and use them only as
place_facts and live-layer evidence. Recommendation: (a), decided before Phase 2.

## D4. Curated-note template and intake format

> SUPERSEDED by D32 (decisions.md): the team cannot write real dated notes. Kept for the record only; do not implement.
Allowed knowledge sources: the curator visited on a date, phoned the place, or read an official notice at a URL. NEVER copied from a
website or a blog. The model never writes a note, and the template's example holds no real claim.

```yaml
# EXAMPLE ONLY. Every value is a placeholder; this is not a claim about any real place.
note_id: example-0001
place: "<registry name>"
claim: "<one factual sentence in the curator's own words>"
aspect: "<timing|closure|price|scenery|food|safety|history|transport|season>"
how_known: "<visited|phone_call|official_notice>"
observed_on: "YYYY-MM-DD"        # the visit or call date, or the notice date
valid_until: "YYYY-MM-DD"        # mandatory for timing, closure, price, safety
evidence: "<photo file id | receipt id | notice URL>"   # required; kept outside git
season_tags: ["<season>"]
is_warning: false
confidence: "<high|medium|low>"
licence: "<owner grants the project the right to use this note>"
curator: "<initials>"
```
Intake rules: a note with no evidence or no observed_on is rejected; text is checked against the web sources for verbatim overlap
(reject if copied); the note becomes one short chunk whose `date` is observed_on and whose date kind is "observed" (O9).

**How many notes give a usable freshness slice (estimate).** The slice is about 8 queries (plan Phase 4). For a graded judgement a
query needs several relevant dated chunks to separate good from bad rankings. Assumption: at least 4 relevant fresh chunks per
query, drawn from at least 3 places. Minimum: the places named by the 8 queries (about 6 to 8 places) x 4 notes = 24 to 32 notes
across the time-sensitive aspects, none older than 18 months at EVAL_REFERENCE_DATE. A safer figure is 5 per place for 8 places,
about 40 notes. A note is about 100 to 200 characters of claim plus context, so one chunk each; 40 notes add about 40 chunks.
**Choosing the 8 freshness queries:** after the coverage matrix, take only places that have at least 3 dated time-sensitive records
(curated or blog); write the query wording from what those records can answer, never the reverse. No query about a hold-out place,
and the dev/test split is made before the notes are inspected for any tuning (CLAUDE.md rule 3). Today no place qualifies: the
newest publish date is 2024-11-21 (Tirupati) and no record is inside the window.

## D5. Link-following to Wikipedia and Wikivoyage articles (design)
Cap: 8 linked articles per place (Wikipedia) and 4 per hub for Wikivoyage district pages. Selection: links from the place page's
"See" and "Do" sections only, ranked by occurrence; skip lists, categories and years. Exclusions: any article whose title or
alias matches a hold-out (registry names and aliases, case-insensitive); any article outside the five states and Puducherry
(Wikidata chain, D18). O8 context rule applied at fetch and again in Phase 2: a bare "Periyar" does not count; it counts as
Thekkady only beside Thekkady, Kumily, Periyar Lake, Periyar Tiger Reserve, National Park or Wildlife Sanctuary. Articles that
mention a hold-out in the text are dropped or tagged by the Phase 2 decision on O8. Record type: raw source wikipedia or
wikivoyage with the parent place as `place`. Request cost: about 1 link request plus 8 article requests per place, about 230 for 26
places; the polite client and the 250-request cap apply. Revision dates only (O9).

## D6. "Freeze without Reddit" decision text (draft for decisions.md; not entered)
> D32 (draft) The Phase 2 corpus is frozen without Reddit. Reddit access was requested but is not approved; the plan does not
> depend on it for correctness, only for traveler voice and freshness. The frozen corpus, qrels and ablations use the sources
> present at the freeze. The report states that Reddit is absent, that freshness rests on dated blog posts, curated notes (if
> D4 is accepted) and the Phase 7 live layer (D31), and gives the measured freshness share. If Reddit is approved later, it is
> collected and chunked into a SEPARATE file (for example data/processed/extra_reddit.jsonl), never into chunks.jsonl after
> FROZEN exists; it is indexed separately and used only by the application layer or a clearly labelled second evaluation
> with its own pooling round (D5 of decisions.md style), reported apart from the frozen results. The frozen ablation table is
> not re-run or changed.

## D7. Combined estimate (nothing implemented; all labelled ESTIMATES)

> D32 removes curated notes: read the curated row as 0, the totals as 2,800 to 5,900 minus 40 to 100 (about 2,700 to 5,800), and
> read every "curated notes or the live layer" remark as "the live layer".
Assumptions: baseline = D26 range for 26 places (1,842 to 2,763 chunks, 200 to 300-token chunks, 15% overlap, 4 chars/token); option
gains from docs/archive/corpus-growth-options.md and the D2 table above; no Phase 2 dedup or O8 drops applied (so all are upper bounds).

| component | chunks (low to high) | source type | dated by publish date? |
|---|---|---|---|
| Baseline: Wikipedia, Wikivoyage, Sahapedia, inditales (26 places) | 1,842 to 2,763 | by cleaned characters on the 9 collected places: reference 43%, wikivoyage 38%, blog 18% (281,313 / 250,144 / 118,975 of 650,432); the 26-place blog share may be lower, because two places (Hyderabad, Kodagu) supply 87k of the 119k blog characters | blog only |
| Linked articles (D5) | +830 to +2,500 | reference, wikivoyage | no (revision date) |
| More inditales posts (row 8) | +130 to +520 | blog | yes |
| Curated notes (D4, SUPERSEDED by D32: row is 0) | +0 | none | n/a |
| OSM, Open-Meteo, Wikidata facts | not chunks (place_facts) | facts | n/a |
| Reddit (blocked on approval) | 0 now (+400 to +1,500 if approved) | reddit | yes |

Total without Reddit: about 2,800 to 5,900 chunks, which is 28% to 59% of the plan's 10,000 floor (the sum of the low and the high
columns above). Source-type mix if the 9-place blog share holds: reference plus wikivoyage about 81% to 82%, blog about 16% to 17%, curated about 1% to 2%,
official 0% (no official page collected so far). Share of chunks with a publish date: about 18% to 19% (blog plus curated over the total),
but almost all of those dates are old: only curated notes (1% to 2%) or the live layer can be recent. The measured share of
publish-dated RAW records today is 18 of 37 = 49%, none inside 18 months. Places with at least one publish-dated chunk:
6 of 9 today (0 inside 18 months); with 40 curated notes over 8 places, at most 8 places would have a recent dated chunk; for all 26 places
you would need curated notes everywhere (about 3 per place = 78 notes) or the live layer.

**Owner decisions still needed:** (1) the corpus target (accept about 3k to 6k, or widen the place list or caps; the plan says 10k to 20k);
(2) which options to approve: D5 link-following, more inditales posts, curated notes and the `curated` source type (D3); (3) per-source caps
(8 articles per place proposed); (4) the place list (26 collectable now; adding places raises both size and effort); (5) O12, what may be
shared, which also decides whether Kaggle (D1/D2 of decisions.md) can receive the corpus; (6) the Reddit freeze text (D6 draft).
Nothing here has been run or implemented.
