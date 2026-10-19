# Decisions (ADR log). Format: ID, date, decision, reason, status.

D1 (2026-10-18) Enrichment and embedding run as resumable Kaggle notebooks, checkpoint every 200 chunks, output
   saved as a Kaggle dataset. Reason: 10-20k chunks through an ngrok tunnel will drop connections and hit session limits.
   The ngrok Ollama endpoint is used only for live LLM calls (Phase 6 query expansion, Phase 7, Phase 9).
D2 Embeddings for the corpus are computed on Kaggle GPU with bge-m3 via sentence-transformers/FlagEmbedding; the same
   library embeds queries locally. Verify vector parity once on a few chunks.
D3 A null chunk date gives neutral freshness (never "old"); log the share of null dates in corpus stats.
D4 EVAL_REFERENCE_DATE is pinned in .env/config so "last 18 months" is reproducible.
D5 Pooling: Phase 4 pools from BM25, dense, simple fusion. After Phase 6 a second pooling round from the top 10 of the
   final pipeline is judged to reduce pooling bias. Every ablation row reports unjudged@10.
D6 Generator and judge models differ. Defaults in .env.example; revise after hardware is known.
D7 Query set: plan says 30 (10 dev / 20 test). Prefer 40 if labeling time allows. Paired tests (ranx), report effect sizes.
D8 No AI attribution in git history (see CLAUDE.md); enforced by settings.json, a Bash hook and .githooks/commit-msg.
D9 (2026-10-18) Python baseline is 3.10 (>=3.10, ruff target py310), not 3.11: 3.10.11 is what is installed. Code must
   stay 3.10-safe (no StrEnum, tomllib, datetime.UTC). Revisit if 3.11 is installed. Status: accepted.
D10 (2026-10-18) Two chunk models in src/common/schemas.py: BaseChunk (Phase 2 output; Phase 3 fields optional and None,
   context_note "" as in contracts) and EnrichedChunk (Phase 3 onward; every field required). Contract fields unchanged.
   Place registry membership of `place` is validated where the registry is loaded (Phase 3). Status: accepted.
D11 (2026-10-18) LLM cache key uses model and API path, not the host, so rotating ngrok URLs keep cached results.
   JSON-mode results are cached only after schema validation; interstitial 200 bodies are never cached. Status: accepted.
D12 (2026-10-18) .githooks/commit-msg bug fixed: "Claude-Session: <value>" lines were not stripped (regex only matched
   a bare "claude-session:"). Covered by a real-repo test. Status: fixed.
D13 (2026-10-18) PostToolUse lint hook changed to `python -m ruff check --quiet . 1>&2 || exit 2` (ruff is not on PATH
   here and the old `|| true` swallowed every failure). Status: accepted.

Open items (decide before the phase that needs them)
O1 Default query set is 40 (not 30); 10 dev / 30 test or similar. Decide in Phase 4 (refines D7).
O2 Keep qrels in separate qrels_dev.json and qrels_test.json; the loader must refuse test qrels unless --split test.
O3 The corrective-search source for CRAG must be chosen before Phase 6. Live web search contradicts "fully local";
   candidates: a pre-cached search over a second held-out corpus, or Wikivoyage/official pages only.
O4 Cohen's kappa needs a second human labeler on a shared subset. If there is only one person, define an alternative
   (e.g. self-agreement after a delay) and say so in the report.
O5 Phase 3 must map LLM-extracted `place` to registry names via aliases, falling back to rule-based tagging.
O6 Phase 5 travel matrix needs a ghat/hill-road correction to the distance x road-factor model; calibrate separately.
O7 Scraped text is untrusted. In Phase 7 the writer prompt must delimit sources as data and tell the model never to
   follow instructions inside them, and the critic must check outputs. Test with a chunk that contains an injection
   attempt.
D14 (2026-10-18) CLAUDE.md and .claude/ stay tracked in git (user decision). No history rewrite. Status: accepted.
D15 (2026-10-18) The user owns phases 1-5 (git identity Nivithasri A). Status: accepted.
D16 (2026-10-18) LLM calls carry a fixed seed (LLM_SEED, default 42) in options, so it is part of the cache key. Total
   time per call is bounded by LLM_RETRY_BUDGET_S (default 300 s, retries included; the per-attempt read timeout is
   capped by the remaining budget). Empty, non-JSON and structurally wrong 200 bodies are retried and never cached.
   Status: accepted.
D17 (2026-10-18) On-demand place ingestion. The system ingests places on demand and reuses them via
   `ensure_place(place_text, budget) -> PlaceStatus` (docs/contracts.md). Manual inputs are limited to credentials in
   .env and a fixed domain allowlist in config/sources.json. The frozen eval snapshot (data/processed/chunks.jsonl,
   registry data/registry/core_places.json) and the live store (data/live/) are separate; nothing live writes to the
   snapshot. Phase 1 delivers resolve + collect + store only: chunking, enrichment and embedding of live places are
   batch jobs (D1, D2) and are not wired. status "collected" means raw data collected and fresh but not chunked or indexed; "ready" is
   reserved for places whose chunks are indexed (Phases 2-3). chunk_counts stay null until then. Live places have no
   qrels, so reported metrics apply to the snapshot only.
   Core list method (2026-10-18): the core list is explicit, config/core_places_list.json (name, state, reason; hub
   and aliases optional; at most 6 hubs). Every listed place is resolved to a Wikidata item and checked (coordinates,
   scope, district-like or non-place items, wrong state, homonyms); nothing is guessed and no label is set by hand.
   Automatic selection (D19 rules + config/core_overrides.json) is kept only as a reported experiment, and it
   produced poor tourism coverage: Vagamon was selected with 0 See/Do listings and Manthraalayam with 1, and Guntur
   and Kakinada were selected while Kochi, Thanjavur, Kodaikanal, Mahabalipuram, Alappuzha, Gokarna and Thekkady
   were not. The experiment pool still supplies the popular/hidden_gem threshold (D19).
   Hold-outs (replaces the earlier hash-based rules; explicit, chosen by the project owner): Chikmagalur, Araku
   Valley, Thekkady and Thanjavur, marked "holdout": true in config/core_places_list.json and holdout=true in
   data/registry/core_places.json. Reason: the flagship places must stay in the frozen corpus so that realistic test
   queries have judged results; the hold-outs are the places that are not flagship. Hold-outs are NOT collected into
   the snapshot (no data/raw, no chunks); they are used only through ensure_place in the cold-start evaluation.
   Phase 4 constraint: no test or dev query may be about a hold-out place, and no hold-out may be used to design or
   tune anything. Status: accepted.
D18 (2026-10-18, corrected) The Union Territory of Puducherry, including Karaikal, Mahe and Yanam, is in scope
   (plan.md lists it) under its own state code "PY". The state enum becomes TN, KL, KA, AP, TS, PY in
   docs/contracts.md and src/common/schemas.py. Puducherry places are NOT mapped to TN, KL or AP. A place is in
   scope when its Wikidata administrative chain reaches one of the five states or Puducherry; other Union
   Territories and states are rejected. This replaces the first version of D18, which mapped the enclaves to the
   surrounding states against the user's instruction. Status: accepted.
D19 (2026-10-18, revised twice; the formula is frozen after round 4) Core list and popularity. Candidates come from
   the Wikivoyage hierarchy (state -> region -> city, "Cities" and "Other destinations" lists), plus Puducherry places
   from Wikidata/Wikipedia when the hierarchy yields fewer than 4. Tourism filter: a place with a Wikivoyage article
   is kept when it has >= 3 See/Do listings (see/do templates) OR >= 120 words of prose in its See/Do sections (See,
   Do, See and do); otherwise it is dropped. A place without a Wikivoyage article cannot be counted and passes (n/a).
   config/core_overrides.json holds the project owner's includes and excludes: an include skips the filter but must
   resolve to coordinates inside the scope and have a signal; an exclude removes the place; each carries a reason.
   Signal for every place: English Wikipedia 12-month user pageviews. Labels: over the whole filter-passing pool,
   hubs excluded from the calculation, popular = the top 20% by signal (threshold = the signal of the last popular
   place), everything else hidden_gem; hubs are labelled by comparing their signal with the same threshold. `hub`
   (population >= 1M or metropolis/state-capital type, districts excluded) is a flag, never an exclusion, but hubs
   count toward the 30 destinations and at most 5 are selected. Per-state minimum 4, maximum 8, PY included, 30
   destinations. Live places get the label by comparing their signal with the stored popular threshold; the tourism
   filter is not applied to them. If the signal is unavailable the entry has
   `popularity_source: "unavailable"` and the planner treats it as hidden_gem. Optional provenance fields on registry
   entries (hub, holdout, wikipedia_title, wikivoyage_title, see_do_listings, ...); the popularity enum is
   unchanged. Wikipedia sections are collected as raw source "wikipedia" and count with Wikivoyage as one
   reference-wiki family. No subreddit list: Reddit search is site-wide with a place-relevance filter.
   Label balance (agreed 2026-10-18): the rule makes the core list lopsided, about 26 popular to 4 hidden_gem, by
   design of the agreed rule (the threshold is the top 20% of the whole pool and most flagship places clear it).
   Phase 7 should get lesser-known places via ensure_place, not from the core list. Status: accepted.
D20 (2026-10-18) For places ingested on demand, chunks are embedded locally with the same bge-m3 model on CPU, and
   context notes use a deterministic template (place, source type, page title) instead of the 7B LLM. The frozen
   snapshot keeps LLM notes and Kaggle embeddings. Vector parity between the two paths is checked once on a sample.
   Reason: first-request latency. Evaluation metrics apply to the snapshot only. Status: accepted.
D21 (2026-10-18) Kodagu and Wayanad stay as district items. Their Wikidata items are districts of India, so
   they keep district-centroid coordinates and are marked place_type "region" (list file and registry entry).
   Consequence for Phase 5: travel_time to or from a region is district-granular, so it is coarser than for a
   town; the matrix should treat region places as centroids and say so in the sanity checks. Status: accepted.
D22 (2026-10-18) New chunk source_type "reference" for Wikipedia and Sahapedia pages (encyclopedic, rarely dated).
   Wikivoyage keeps "wikivoyage". Contract enum is now reddit, blog, youtube, wikivoyage, official, reference; raw
   record source gains "sahapedia" (maps to reference). Reason: neither is a traveler blog nor an official
   tourism page, and calling them blog would corrupt the source-diversity metrics in Phases 4 and 6. Phase 6
   MMR and the diversity metrics count reference as its own type. Status: accepted.
D23 (2026-10-18) Coverage rule for a core place (coverage_report.py): it passes when it has raw data from >= 2
   distinct chunk source types AND at least one of them is not wikivoyage or reference. Reason: Wikivoyage and
   Wikipedia are one reference-wiki family (D19) and cannot supply traveler experience or freshness on their
   own. A skipped collector (no credentials, not allowed) is reported as skipped, never counted. Status: accepted.
D24 (2026-10-18) Four web domains are scoped in config/sources.json. NONE of these scopings was requested by the
   project owner; the implementer added them when generalising the allowlist ("states" and "places" keys), so that a
   site is only asked for places it can cover. Reversible by deleting the key. Status: accepted, pending owner review.
   - aptourism.gov.in -> states ["AP"]: it is the Andhra Pradesh Tourism Authority; its pages cover AP only.
   - www.keralatourism.org -> states ["KL"]: it is the Kerala board; asking it about Hampi or Chennai cannot succeed.
   - hrce.tn.gov.in -> states ["TN"]: it is the Tamil Nadu temples department.
   - tirumala.org -> places ["Tirupati"]: the Tirumala Tirupati Devasthanams site covers that one place.
   Reason (all four): fewer requests to small public sites with no possible gain. Effect: Munnar and Hampi no longer
   touch aptourism.gov.in; only Tamil Nadu places touch hrce.tn.gov.in (Chennai is its first contact in batch 1).
D25 (2026-10-18) Page discovery stays automatic (D17): sitemaps announced in robots.txt and the standard sitemap paths,
   matched to the place by name in the URL. There are no per-place seed URLs as the primary path. Seed URLs may exist only
   as an optional config file whose URLs the project owner has verified by hand; they are never generated from model
   memory and are excluded from the never-seen-place demo and from any coverage claim about on-demand ingestion.
   Status: accepted.
O8 (Phase 2, hold-out text) Raw pages about one place can name a hold-out place in their text (slice example: the Munnar
   Wikivoyage page names Thekkady 9 times; a Tirupati blog post names Araku once). When the corpus is frozen, a chunk whose
   text mentions Chikmagalur, Araku Valley, Thekkady or Thanjavur (or their aliases) must be dropped or tagged
   (`mentions_holdout`), so no test or dev query can be answered from hold-out content (D17). Decide drop vs tag in Phase 2.
   Not implemented.
O9 (2026-10-18, Phase 2) Wikipedia and Wikivoyage records store only a revision date (`lastmod`), never a publish date
   (contract: lastmod is never used as a publish date). Phase 2 must tag every chunk's date with its KIND (publish date
   vs revision date vs none) so that freshness metrics and the freshness share do not treat reference sources as
   fresh; D3 (null date = neutral) still applies. Requested by the owner; added now (it was missed in the previous round).
D26 (2026-10-18, updated after D27) The plan's target of 10,000 to 20,000 chunks is NOT met by the planned collection.
   Estimate without transcripts (upper bound, estimate_corpus_size.py (deleted, D47)): cleaned text 190,808 characters over the 3
   slice places (strict level 187,504; raw 194,940 characters, 196,309 bytes), about 74 to 75 chunks per place at 250 tokens
   and 1,600 to 2,400 chunks for the 26 collectable places across 200 to 300 token chunks, about 19% of the 10,000 floor.
   Why: (1) the slice places are three popular places and may not represent the rest; (2) one page per place from each
   wiki; (3) web caps of 8 to 15 pages per domain; (4) Sahapedia and the official sites returned 0 pages (their sitemaps do
   not list place pages, D25 keeps discovery automatic); (5) Reddit is not available (credentials pending); (6) YouTube
   transcripts, which were 47% of the earlier text, are dropped (D27); (7) Phase 2 dedup, off-topic filtering and O8
   drops will lower the number further. Caps are unchanged. The report states the corpus size as measured, and any change
   to caps or sources needs a later decision. Status: accepted.
D27 (2026-10-18; stands, confirmed by D30) The YouTube transcript path is DROPPED. Reason: youtube-transcript-api needs POST /youtubei/v1/player
   and GET /api/timedtext, and youtube.com/robots.txt disallows `/youtubei/` and `/api/` for our User-Agent; our policy
   requires robots and terms to allow a fetch, and YouTube's terms on automated access were not reviewed. The owner did
   not accept the risk. Consequences: config/sources.json has `"youtube_transcripts": false`; the YouTube collector
   refuses to run (skipped with "D27", zero requests, no search, no key use) and is not in the default source list;
   the 13 collected transcripts and their 15 cache entries were withdrawn (manifest lines kept, "withdrawn" lines
   added; withdraw_source.py (deleted, D47)); GuardedSession, the real-library transcript tests and mutation M4 are not built.
   The earlier classification fix (403/429/unknown = block) stays in the code as a safeguard. Status: accepted.
O10 (YouTube Data API) The official YouTube Data API could provide video descriptions and comments (documented API,
   own terms, quota). It is NOT enabled, not designed and not decided: listed only. The leftover cached search responses
   in .cache/http (www.googleapis.com) are not read by any code path (the collector returns before searching). Revoking
   YOUTUBE_API_KEY is the owner's decision; .env is untouched.
D29 (2026-10-18) developers.google.com is allowed NARROWLY, for documentation only: config/sources.json -> policy_fetch lists
   the host and three YouTube API policy pages (developer policies, API services terms, quota cost). They are fetched by
   fetch_policy_pages.py (deleted, D47) only, one request per page through HttpClient (robots-checked, logged), to answer the
   owner's questions about storage, retention, scraping and quota. They are not part of web_allowlist, are never
   place-matched or crawled, and nothing from them enters the corpus. Status: accepted.
O11 (CLOSED as moot by D30, 2026-10-18; was: owner decision needed before Phase 2) Which corpus files that contain YouTube-derived chunks may be
   committed, pushed to origin, or uploaded to Kaggle? Chunks are verbatim excerpts of transcripts. Until decided,
   data/processed is treated as local and git-ignored (only .gitkeep is tracked), as are data/raw, data/live and .cache.
D30 (2026-10-18) D27 STANDS: YouTube transcripts (youtube-transcript-api) and the YouTube Data API stay OFF for anything stored.
   The owner reviewed the policy findings (docs/sources.md). Why: (1) the scraping clause: API clients "must not ... scrape
   YouTube Applications" or obtain scraped data, and the key and project we would use sit under it; (2) the 30-day storage
   rule: API data may be stored for at most 30 calendar days, then deleted or refreshed, which is incompatible with a
   frozen corpus; (3) the termination consequences: suspension of credentials and project, and deletion of all API data.
   This supersedes nothing. A decision D28 (re-enable transcripts as an owner-accepted exception) was proposed but never
   written, because the policy findings came first; the number D28 is intentionally unused. D29 (the narrow policy-page fetch
   allowlist) is kept. Consequences: the 3 cached Data API search responses were deleted (30-day rule); the collector stays
   switched off (`youtube_transcripts: false`, guard tests); O11 is moot. Status: accepted.
O10 (updated 2026-10-18) The YouTube Data API stays OFF for stored data. One possible later use is listed as an OPTION for
   the Phase 7 live layer only: a per-request live call whose results are cached for at most 30 days and are NOT part of the
   frozen corpus. Display and attribution requirements for that use are UNVERIFIED (not yet read from the policies).
O8 (amended 2026-10-18) The hold-out text check uses the registry names and aliases (Thanjavur: Tanjore, "Tanjore, Thanjai";
   Chikmagalur: Chikkamagaluru; Thekkady and Araku Valley have no registry aliases) plus a CONTEXT rule, not a bare word
   list. "Periyar" is ambiguous (the social reformer known as Periyar, Periyar University in Salem, the Periyar river) and
   must not count by itself: it counts as a Thekkady mention only next to Thekkady, Kumily, Periyar Lake, Periyar Tiger
   Reserve, National Park or Wildlife Sanctuary. Slice evidence: of 5 bare hits, 1 is the river (Munnar, Wikipedia) and 4 are
   written "Thekkady/Periyar", already caught by the name itself. Not implemented (Phase 2).
D31 (2026-10-18) PROPOSED TO VALINCIA, NOT BINDING UNTIL SHE ACKNOWLEDGES. Live web search is an application-layer feature
   (docs/plan.md Section 2.1, Phase 7 live node, Phase 9 comparison, rule 10). It sits OUTSIDE the frozen corpus, the
   indexes, the qrels and the ablations: retrieve() stays frozen-corpus only and live chunks are never written to
   data/processed or the indexes. It must be free, comply with the source's terms, and use the polite HttpClient (robots
   check, throttle, request log, hard cap). Valincia decides the search API and the details (plan.md lists her open
   decisions). docs/contracts.md carries the matching additions (`live` in the source_type enum, `live_retrieve`) marked as
   proposed. Open item for her: the YouTube Data API as a per-request live source (results cached at most 30 days, never
   frozen; display and attribution terms UNVERIFIED), which refines O10. No live code exists or was written in Phase 1.
   Status: proposed, awaiting Valincia's acknowledgement. Note: the plan.md live-layer text was committed inside 9d86719
   (the D29 allowlist commit), not in its own commit.
O12 (2026-10-18, owner decision needed BEFORE Phase 2) Which corpus files (data/processed chunks, and later index/) may be
   shared with the partner, pushed to origin, or uploaded to Kaggle, given the licences in the docs/sources.md ledger
   (CC BY-SA Wikipedia/Wikivoyage with attribution and ShareAlike; CC BY-NC-SA Sahapedia; blog and official pages with
   no licence, risk accepted for collection only). Until decided, data/processed (and data/raw, data/live, .cache) stay
   gitignored and local. Supersedes the general part of O11 (O11 stays closed as moot, D30).
D32 (2026-10-18) Curated notes are NOT possible: the team cannot write real, dated, evidenced notes, and the model must never write
   them. The frozen corpus has no fresh content (0 of 18 inditales posts inside the 18 months before EVAL_REFERENCE_DATE
   2026-10-18). Freshness evidence therefore comes only from the Phase 7 live layer (D31) and from any reply to the Tamil Nadu,
   Karnataka and ASI permission emails. Valincia's freshness ablation on the frozen corpus is expected to show a null or limited
   result. The Phase 4 freshness-slice queries (the existing `freshness_slice` flag, no schema change) are flagged as live-layer
   queries and reported separately. Supersedes the curated-note parts of docs/archive/corpus-research.md (D3 curated source_type
   question, D4, the D7 curated row). Status: accepted by the owner's instruction, pending Valincia's acknowledgement for the
   Phase 4/6 reporting consequences.
D33 (2026-10-18) After batch 2, D23 is applied ONCE to all 26 collectable places. A place that fails is excluded from the
   frozen corpus by this rule. Its raw data and registry entry are KEPT (travel_time and check_season still need them; nothing is
   deleted). The Phase 7 planner must not recommend a place that has no corpus content. No Phase 4 query may target an excluded
   place. If fewer than 20 places pass, that is recorded as a deviation from the plan's 20 to 30 destinations. Reason: a place with
   only reference-wiki text cannot supply traveler experience, and listing it would corrupt retrieval and diversity metrics.
   Estimate, not a result: without Reddit and official pages a place passes only if it has a blog post, so the count may fall
   below 20. Status: accepted.
D34 (2026-10-18) The plan's 10,000 to 20,000 chunk target is NOT met (D26) and is not pursued by changing the chunk size: chunks
   stay at about 200 to 300 tokens. The only in-policy lever is Wikipedia and Wikivoyage link-following with a per-place cap
   (design: docs/archive/corpus-research.md D5; 8 linked Wikipedia articles per place, 4 Wikivoyage sub-pages per hub). Expected range
   (estimate, before D33 exclusions and Phase 2 dedup): +830 to +2,500 chunks on a 1,842 to 2,763 baseline, so about 2,670 to
   5,260 in total; all of it reference or wikivoyage text with revision dates only (O9), so it adds no freshness. Nothing is
   implemented; implementation needs the owner's go-ahead. Status: accepted (design only).
D35 (2026-10-18) YouTube stays OFF for anything stored (D27, D30). For Valincia's live layer (D31, proposed) the Data API may be
   used per request: search.list, videos.list, commentThreads.list. Quota as recorded in docs/sources.md from the fetched policy
   pages: search.list 100 calls a day in its own bucket, the others 1 unit each. Design guards: store comment text only, no
   usernames; cache for at most 30 calendar days; never freeze. Terms and display or attribution requirements for this use are
   UNVERIFIED. Optional owner action, only if a frozen YouTube snapshot is ever wanted: apply for the YouTube API Compliance Audit.
   Status: proposed to Valincia, not binding until she acknowledges.
O10 (amended 2026-10-18, D35) The per-request live use of the Data API is specified in D35. Still UNVERIFIED: display and
   attribution requirements; whether comment text may be shown. The key stays unused by Phase 1 code.
D31 (amended 2026-10-18, D35) The YouTube Data API question listed above is now specified in D35. D31 stays proposed to Valincia.
D36 (2026-10-18) API-host robots exception (owner choice, design only). Documented programmatic API endpoints may be called with the
   robots check off, as the Wikimedia API calls already are (check_robots=False in src/collect/sources.py and src/places/places.py),
   under these conditions, per host: (1) the host's terms or API documentation were read and the date and a short quote are recorded
   (docs/facts-layer.md); (2) a contact User-Agent from .env, published rate limits honoured, one request at a time; (3) every call
   through HttpClient and the disk cache, counted against the request cap, logged; (4) a 403 or 429 still stops the whole run
   (BlockDetected) with no retry or workaround; (5) web PAGES keep the robots check; (6) a host is added only by a later reviewed
   config change that cites its decision ID. Evidence (robots.txt fetched through HttpClient on 2026-10-18, RFC 9309 matcher, our User-Agent, group `*`; check_robots.py (deleted, D47)):
   overpass-api.de has `Disallow: /api/` (line 2) and the matcher DISALLOWS /api/interpreter; archive-api.open-meteo.com has `Disallow: /` (line 2) and the matcher
   DISALLOWS /v1/archive. The D36 exception therefore applies to both hosts. For www.wikidata.org the matcher ALLOWS Special:EntityData (D41), so it was not needed there [SUPERSEDED by D41 for the Wikidata Special:EntityData path: the earlier False was a urllib.robotparser artifact; the exception still covers /w/api.php];
   the wikis' /w/api.php (`Disallow: /w/` per the fetch-tool report) was not re-checked. Terms for these hosts are UNVERIFIED in this step. Nothing is allowlisted or fetched by this decision; the reserved ID for the facts-layer hosts is D37. If
   a host's terms forbid the use, the exception does not apply. Status: accepted by the owner's choice 2026-10-18.
D38 (2026-10-18) Per-run circuit breaker in HttpClient (replaces the proposal in docs/archive/full-run-plan.md). After 2 consecutive connect
   or timeout failures to a host, further requests to it in the same client are skipped with no network call, no throttle
   wait and no count against the cap; one log line "circuit open" per host; any completed HTTP response resets the count;
   TLS and DNS handling is unchanged; a new run starts closed. Reason: batch 1 lost 10 s to a dead connection and nothing stopped
   repeats. Tests: tests/test_guards.py (9 tests, fake clock); mutation (threshold never reached) failed 5. Batch 2: it did
   not trigger. Status: accepted.
D37 (2026-10-18) API host allowlist for the facts layer, under D36. config/sources.json -> api_hosts lists three hosts, each with a path prefix, terms
   URL, licence, minimum delay, per-run cap and robots_exempt true: www.wikidata.org (/wiki/Special:EntityData/, CC0, 1 s, 30), overpass-api.de (/api/interpreter,
   ODbL, 30 s, 30), archive-api.open-meteo.com (/v1/archive, CC-BY 4.0, free tier non-commercial as resolved by the owner, 2 s, 30). Code in src/facts refuses any
   other host. Raw results are JSON envelopes under data/raw/<source>/<place>/ (entity.json, stations.json [SUPERSEDED by D40: no per-place station files exist; the only Overpass file is airports.json], archive_2015_2024.json; the one airport query under
   data/raw/overpass/_south/), not records.jsonl, so coverage_report, coverage_matrix and the size estimate ignore them; contracts.md notes this. Hold-outs are refused
   with exit code 2; a 403, 406 or 429 stops a run (BlockDetected). Each source is approved to run separately: Wikidata first; Overpass and Open-Meteo (1 place first)
   need the owner's separate approval. No place_facts table before Phase 5. Status: accepted.
D39 (2026-10-19) Freeze rules and recorded deviations (owner instruction).
   (i) D33 is DEFERRED to the Phase 2 freeze. Reason: Reddit access is still pending, so the pass/fail result may change; applying the
   rule now would exclude places before the corpus is final. The rule itself is unchanged: at the freeze, D23 is applied once to all 26
   places, failing places are excluded from the frozen corpus, their raw data and registry entries are kept, and the planner must not
   recommend a place without corpus content.
   (ii) Deviations from the plan, as measured on 2026-10-18/19 (all upper bounds, reproducible with the commands in LOG.md): D23 pass
   count is 18 of 26 (the plan wants 20 to 30 destinations); about 1,943 chunks at 250 tokens (range 1,619 to 2,429) against the plan's
   10,000 to 20,000; 0 records inside the 18-month window (newest publish date 2025-01-30, Ooty; the window starts 2025-04-18);
   Kerala has 1 of 5 places passing D23 (Kochi only; Kerala Tourism yielded 0 records because its pages are client-rendered);
   0 official records; Reddit absent. The report states these numbers as measured.
   (iii) Phase 2 rules: (a) a chunk whose text names a hold-out (names, aliases, Kumily, Araku, Brihadisvara/Brihadeeswarar, and the
   Periyar context rule of O8) is DROPPED, which settles O8 as drop, not tag; (b) every chunk carries date_kind = published | revised
   | none (O9); the freshness share and every freshness metric count published dates only; (c) no dev or test query may be about a place
   outside the frozen set (an excluded place or a hold-out). Status: accepted.
D40 (2026-10-19) Overpass is limited to the ONE airport query. The 26 per-place railway-station and bus-station queries are removed from code
   (src/facts/facts.py, scripts/collect.py, tests) and from docs/facts-layer.md. Reason: not in the plan, ODbL share-alike cost on a larger
   derived database (O12), and railheads come from the Phase 5 travel matrix. The Overpass read timeout is 90 s (the 30 s default timed out once).
   The raw airport envelope is saved unfiltered; the hold-out/10 km filter is applied when derived lists are made. Config purpose string updated; D37's
   stations.json file name no longer applies. Status: accepted.
D41 (2026-10-19) RFC 9309 matcher and the Wikidata robots result. src/facts/facts.py implements RFC 9309 matching (longest match wins, `*`
   wildcard, `$` end anchor, Allow wins a tie; tests/test_facts.py, 8 tests with synthetic robots text including the Wikidata Disallow/Allow pair).
   Applied to the CACHED wikidata.org/robots.txt (retrieved 2026-10-18T18:32:26Z, no new request) for https://www.wikidata.org/wiki/Special:EntityData/Q1361.json,
   group `*` (no group names our product token): the matching rules are Disallow /wiki/Special: (line 156, length 14), Disallow /wiki/Special:EntityData/ (line 435,
   length 25) and Allow /wiki/Special:EntityData/*. (line 436, length 27); the longest is the Allow, so the verdict is ALLOWED. Python's urllib.robotparser, on the same
   text and URL, says False because it treats `*` literally. Conclusion, limited to what the matcher printed: for Special:EntityData on that copy of the file the D36
   exception was NOT needed; the earlier False was a parser artifact. D36 stays in force for the other hosts: overpass-api.de (`Disallow: /api/`) and the Open-Meteo archive
   host (`Disallow: /`) as reported by the fetch tool, which have no cached copy and were not checked with the matcher (UNVERIFIED), and for /w/api.php on the wikis (not cached,
   not checked). The HttpClient robots check and config robots_exempt are unchanged. Status: accepted (information only).
O13 (2026-10-19, Phase 5) The Overpass airport query covers the box lat 8 to 20, lon 74 to 85, which returns airports outside South India. In the 47 kept airports,
   judging from the names and codes only (UNVERIFIED): Sri Lanka (ACJ, JAF, TRR), Maharashtra (IXU, LTU, NDC, SAG, SSE, KLH) and Chhattisgarh/Odisha (JGB, PYB). Phase 5
   needs a region filter (state of the airport, for example by point-in-polygon against the five states and Puducherry, or by the Wikidata administrative chain) before
   nearest_airport is computed, plus deduplication by IATA (HYD, PYB and VDY have two elements each). Example of why: the nearest kept airport to the hold-out Araku Valley is
   PYB Jeypore, 69.2 km away, outside AP.
D37 (amended 2026-10-19, D40 and D41) The file name stations.json in D37 no longer exists: the per-place Overpass station queries were removed (D40), so the only Overpass file is
   data/raw/overpass/_south/airports.json. For www.wikidata.org the robots_exempt flag is kept, but D41 shows the RFC 9309 verdict for Special:EntityData is ALLOWED (Allow on line 436 of the
   cached file); the earlier False from urllib.robotparser was a parser artifact. The Wikidata run therefore did not rely on the D36 exception.
O13 (amended 2026-10-19) Phase 5 uses a curated IATA ALLOWLIST of the 33 in-region codes, to be stored as a data file with a test; NEITHER THE DATA FILE NOR THE TEST EXISTS YET (checked 2026-10-19 in config, src, tests, scripts; Phase 5 task) (33 unique codes, every code present in the
   saved Overpass airport envelope, none of the 11 out-of-region codes). In-region codes: BEP BLR BPM CCJ CDP CJB CNN COK GBI HBX HYD IXE IXG IXM IXX KJB MAA MYQ NVY PNY PUT RJA RMD RQY SXV TCR TIR TRV
   TRZ VDY VGA VTZ WGC. Out-of-region (12 elements, 11 codes): ACJ, IXU, JAF, JGB, KLH, LTU, NDC, PYB (2 elements), SAG, SSE, TRR. ONE entry per IATA; the dedupe rule is written down: among elements with the same
   IATA keep the one with the highest type priority relation > way > node, and on a tie the lowest OSM id (an arbitrary but deterministic convention). Evidence that a distance filter does not work: the
   12 out-of-region elements lie 101.9 km (JAF, nearest place Rameswaram) to 441.4 km (SAG) from the nearest non-hold-out registry place, while the in-region airport farthest from any registry place is GBI at
   160.4 km. Any threshold that keeps every in-region airport (T >= 160.4 km) also keeps JAF (101.9 km) and both PYB elements (153.2 and 153.4 km); only a threshold below 101.9 km would keep none of the
   out-of-region ones, and it would drop 11 in-region codes. So YES: a distance-based filter would have kept an out-of-region airport; use the allowlist.
D36 (amended 2026-10-19) Terms for Overpass and Open-Meteo are UNVERIFIED in this project: no terms page was fetched in the robots-evidence step, and the owner will paste the quotes. No quote is
   invented here. Wiki robots result: en.wikipedia.org/robots.txt and en.wikivoyage.org/robots.txt are NOT in the cache, so they were not checked with the RFC 9309 matcher and no verdict is claimed
   (the fetch-tool report said `Disallow: /w/` with only action=mobileview allowed, UNVERIFIED by the matcher). The API path our wiki collectors use is /w/api.php?action=query (src/collect/sources.py, wv_lists.py (deleted, D47));
   src/places/places.py also calls https://www.wikidata.org/w/api.php and https://query.wikidata.org/sparql. Applied to the cached www.wikidata.org robots.txt (retrieved 2026-10-18T18:32:26Z), the matcher
   DISALLOWS /w/api.php?... by `Disallow: /w/` (line 153), so the registry's Wikidata API calls do rely on this exception; query.wikidata.org/robots.txt is not cached (not checked).
D37 (amended 2026-10-19) Terms for overpass-api.de and archive-api.open-meteo.com are UNVERIFIED (see D36 amendment); the owner will paste quotes. The allowlist entries stay in force on the owner's decision.
D42 (2026-10-19) Open-Meteo elevation handling (owner instruction).
   (i) The `elevation` parameter adjusts the returned temperatures. Observed shifts between the file with the Wikidata elevation and the file without the parameter (API-echoed
   elevations in brackets): Gokarna 586 m sent vs 2.0 m (584 m): January and May Tmax and Tmin 3.7 to 3.8 C warmer without it, which fits about 6.4 C per km (lapse rate from memory, UNVERIFIED);
   Kanniyakumari 300 m vs 36.0 m (264 m): 1.8 C warmer without it in each of the four figures (about 6.8 C per km); Varkala 190 m vs 46.0 m (144 m): 1.1 C (about 7.6 C per km).
   Mean annual precipitation was identical for Kanniyakumari (1126.1 mm) and Varkala (2421.7 mm) and differed slightly for Gokarna (2991.7 vs 2983.7 mm).
   (ii) A place's no-elevation file (archive_2015_2024_noelev.json) is PRIMARY when its Wikidata elevation is implausible for the place. Gokarna: the no-elevation file is primary and the original
   (elevation 586 m) is SUPERSEDED, not deleted. Kanniyakumari and Varkala: both files are kept and the choice of primary is left to the owner's review (no judgement recorded here) [SUPERSEDED: settled by D42(a), the no-elevation files are primary and the originals are superseded, not deleted].
   (iii) Kodagu and Wayanad (centroid regions) and the hill stations (Munnar, Ooty, Kodaikanal) get confidence `low` for every Open-Meteo-derived value; other places keep `medium`.
   (iv) No other Open-Meteo refetches without the owner's review. Status: accepted.
D36 (amended 2026-10-19, terms record; as returned by the fetch tool in the review chat, 2026-10-19; owner to verify against the live page) Open-Meteo terms: the quote "You may only use the free API services for non-commercial purposes." Limits: under 10,000 calls per day, 5,000 per hour and
   600 per minute; licence CC-BY 4.0; non-commercial examples include public research at public institutions and educational content; the page states no call weight and no caching rule. Overpass wiki usage
   policy: the quote "No parallel running of multiple scripts". Limits: under 10,000 queries per day and 1 GB per day; a User-Agent or Referer that identifies the app; pause 30 s after 429 or 406; the page does not
   mention ODbL attribution (O12 stays open). Nothing else is claimed.
D36 (correction 2026-10-19) The cached www.wikidata.org robots.txt is a PROXY, not evidence for en.wikipedia.org or en.wikivoyage.org: those two robots.txt files were never fetched through HttpClient, so their rules
   stay UNVERIFIED. The exception for the wiki API calls rests on D36 plus the Phase 2 owner action on the Wikimedia API terms, not on a matcher result.
D37 (amended 2026-10-19, host audit) A read of data/manifests/request_log.jsonl (414 lines from 2026-10-18T16:01:46Z) against config/sources.json found: covered hosts (web_allowlist incl. www/apex pairs, the three
   api_hosts with their path prefixes, developers.google.com under policy_fetch); NOT covered by any config entry: en.wikipedia.org (23 requests) and en.wikivoyage.org (22), which are hard-coded in src/collect/sources.py and
   called with check_robots=False. Not in the request log (made before logging and hard-coded in src/places/places.py and the bootstrap scripts, so also not in config): query.wikidata.org (SPARQL), wikimedia.org
   pageviews, and www.wikidata.org /w/api.php (outside the api_hosts prefix /wiki/Special:EntityData/). Flagged for the owner; nothing changed.
D42 (amended 2026-10-19) (a) Kanniyakumari and Varkala: the no-elevation files are PRIMARY; the originals are SUPERSEDED, not deleted. (b) Confidence `medium` for Gokarna, Kanniyakumari and Varkala (`low` stays for
   Kodagu, Wayanad, Munnar, Ooty, Kodaikanal). (c) The elevation parameter changed temperature only in two of the three pairs: precipitation was identical for Kanniyakumari (1126.1 mm) and Varkala (2421.7 mm), but for
   Gokarna it differed by 8.0 mm (2991.7 vs 2983.7 mm); recorded as measured, not as "identical in all three". (d) Refetch only when the likely elevation error exceeds about 150 m; no further refetches.
D43 (2026-10-19) Request allowlist for every production client, and five more API hosts (owner instruction, reviewed config change). config/sources.json -> api_hosts gains entries, all
   robots_exempt true, min_delay_s 1.0 (sequential), max_requests_per_run 30, decision D43, with a terms_status field: en.wikipedia.org /w/api.php, en.wikivoyage.org /w/api.php, www.wikidata.org
   /w/api.php (the /wiki/Special:EntityData/ entry stays), query.wikidata.org /sparql, and wikimedia.org /api/rest_v1/metrics/pageviews/per-article/ (host and prefix exactly as in
   src/places/places.py PAGEVIEWS_URL). terms_status for the three /w/api.php hosts: "as returned by the search tool in the review chat, 2026-10-19; owner to verify": Wikimedia Access policy
   quote "The API requires an HTTP User-Agent header for all requests."; API:Etiquette "no hard limit on read requests, be considerate". For query.wikidata.org and the pageviews host:
   UNVERIFIED (the owner will paste a quote). ENFORCEMENT: src/common/http.py builds an Allowlist from the config and every client made by make_http (all collectors, the resolver, the facts
   fetchers, check_robots, the demo) refuses, before any network call and before the throttle, a URL whose host or path is not in web_allowlist domains (matched by the explicit www/bare twin rule
   only: no suffix, subdomain or redirect match), api_hosts (host plus path prefix; /robots.txt is allowed on listed hosts) or the policy_fetch hosts; per-host caps from the config are hard stops
   (RequestBudgetExhausted). This also closes a gap: page URLs listed in a sitemap on another host used to be fetched without a host check. Tests: tests/test_guards.py (a non-listed Wikimedia path
   such as /w/index.php is refused; inditales.com only through the twin rule). Not covered: bootstrap_registry.py [removed, D46], build_core_list.py and gather_source_evidence.py [now under the historical scripts (deleted, D47), D46] build their own HttpClient
   (historical, one-off) and are not restricted. CAP NOTE: scripts/collect.py now defaults to a hard cap of 25 network requests (was 15; Yercaud used 14, 12 of them by the resolver);
   there is no retry. Status: accepted.
D43 (audit findings recorded 2026-10-19) (1) The cached pageviews entry used in the Yercaud demo (kind http, host wikimedia.org, article Yercaud, window 2025100100-2026093000) was written at
   2026-10-18T12:30:03Z, before the request log began (2026-10-18T16:01:46Z), so it cannot be traced to a run id; the only log line is the cache hit of run fe09e288 at 2026-10-19T09:17:14Z. It most
   likely comes from the core-list bootstrap (not logged). (2) The demo's 15 cache hits: hrce.tn.gov.in 3 (robots 1, sitemap 2), inditales.com 5 (sitemap 5), sahapedia.org 3 (robots 1, sitemap 2),
   www.inditales.com 3 (robots 1, sitemap 2), wikimedia.org 1 (pageviews). (3) LIVE-LAYER LIMIT: a small place such as Yercaud yields reference wikis only (Wikipedia and Wikivoyage); web found no page;
   D23 fails. Phase 7's live node therefore cannot rely on fresh web content for such places.
D43 (amended 2026-10-19, guards) (1) tests/test_guards.py pins the exact set of eight api_hosts entries (host, path_prefix, min_delay_s, max_requests_per_run, robots_exempt) and requires plain
   hostnames, so any added, removed or changed entry fails a test and needs a reviewed change. (2) Redirects: tests/test_guards.py proves, with a fake transport, that a redirect to an unlisted host,
   to a path outside an api_hosts prefix, to a URL carrying user information (an allowed name before the at-sign with another host after it, and the reverse) and to a non-http scheme is refused before the follow-up request is sent; the allowlist
   now also rejects any URL with user information. (3) The three historical scripts (bootstrap_registry.py [removed, D46], build_core_list.py (deleted, D47), gather_source_evidence.py (deleted, D47)) exit 2 with a message
   unless --historical is passed, before any HttpClient or settings object is built (src/common/http.py [the former guard.py was merged into it, D46], test_historical_scripts.py (deleted, D47)). WITH --historical THEY BYPASS THE ALLOWLIST: each builds its own
   HttpClient. They were not run.
D43 (amended 2026-10-19, secondary terms evidence; as relayed through the review chat on 2026-10-19; owner to verify against the live pages) Wikimedia REST API (pageviews host wikimedia.org): quote "Limit your clients to no more than 200 requests/s to this API." plus a unique User-Agent that allows contact; the Wikimedia Terms of Use apply (a mirror of the mediawiki.org page, not the original). Wikidata Query Service (query.wikidata.org): SECONDARY source only (R package documentation quoting the manual; the manual itself was NOT read): quote "One client is allowed 30 error queries per minute"; 60 s hard query deadline; one client is allowed 60 s of processing per 60 s. config/sources.json terms_status for these two hosts is UNCHANGED (still UNVERIFIED); both hosts now have secondary evidence, recorded in docs/facts-layer.md 5d. Our 1 s sequential delay is far inside 200 requests/s; the WDQS figures were not checked against the actual SPARQL usage (4 requests in the request log).
D44 (2026-10-19) D33 stays as written (applied once at the Phase 2 freeze, D39). If applied, NO hidden_gem place survives in the frozen corpus: Chikmagalur and Araku Valley are hold-outs (never collected), and Varkala and Bhadrachalam, the only other hidden_gem entries, fail D23. Consequence: popular-versus-hidden diversity cannot be evaluated on the frozen corpus (all 18 remaining places are labelled popular). Phase 4 diversity metrics therefore stay distinct source types and distinct places in the top 10, as in the plan. PROPOSED derived field for Phase 2 or Phase 5 (no schema change now): `popularity_tercile` (T1 high, T2 mid, T3 low) from the stored 12-month Wikipedia pageviews (registry field popularity_signal), computed within the 18 D23 passers; the tercile within all 26 collectable places is shown for comparison. It is a coarse planner signal, not a hidden_gem label. Table (read-only from data/registry/core_places.json and data/manifests/coverage_report.md; terciles split by rank, 26 -> 9/9/8, 18 -> 6/6/6): docs/popularity-terciles.md.
D45 (2026-10-19) Recorded deviations from the Phase 1 plan. (1) YouTube transcripts are dropped: accepted, see D27 and D30. (2) Reddit is DEFERRED, not dropped: no credentials yet; if credentials arrive, run the Reddit collector before the Phase 2 freeze; if they do not, the absence is ACCEPTED at the freeze and the report states 0 Reddit records. The collector's live behaviour and Reddit's current access rules have not been exercised or confirmed. (3) 0 official records (Kerala Tourism is client-rendered, the other sites expose no usable sitemap). (4) D23 passes 18 of 26 places, below the plan's 20 (D39). Status: accepted by the Phase 1 closing instruction; item (2) is conditional on the credentials.
D46 (2026-10-19) Layout cleanup and two dead-code removals (owner instruction; behavior-preserving except the two removals). PATH CHANGES (old -> new): facts/geo.py -> src/facts/facts.py; places/aliases.py -> src/places/places.py; places/overrides.py -> core_list.py (deleted, D47) (its DEFAULT_PATH constant is renamed OVERRIDES_PATH there, because core_list already had a DEFAULT_PATH); collect/matrix.py -> size_estimate.py (deleted, D47); common/guard.py -> src/common/http.py; build_core_list.py and gather_source_evidence.py -> the historical scripts (deleted, D47) (guard unchanged, sys.path bootstrap parents[1] -> parents[2]); corpus-research.md, corpus-growth-options.md and full-run-plan.md -> docs/archive/; kaggle-runbook.md -> kaggle/README.md; the outreach folder (README and three drafts) -> docs/outreach.md (wording unchanged). Test files merged: test_config_optional -> test_config, test_schemas_strict -> test_schemas, test_cache_recovery -> test_cache, test_llm_resilience and test_smoke_helpers -> test_llm, test_matrix -> test_size_estimate, test_ensure_demo -> test_ensure_place, test_aliases -> test_places_resolve, test_http_policy -> test_http (its helpers UA, ok, make renamed POLICY_UA, pok, pmake to avoid name clashes); the two atomic-write tests moved from test_http to test_cache and the size-estimate test from test_http_policy to test_size_estimate. REMOVAL 1 (YouTube, D27/D30): the transcript collector is replaced by a 21-line refusal stub at youtube.py (deleted, D47) (SKIPPED_D27 and a YouTubeCollector that always returns skipped, with no key, client, cache or library use); the orchestrator builds it with no arguments; test_collect_youtube.py, test_collect_blocks::test_transcript_http_403_is_a_block_not_a_transient_error and two collector-level tests in test_youtube_policy are deleted; the orchestrator refusal and no-network tests stay. The youtube-transcript-api dependency in pyproject.toml is now unused and was left in place. REMOVAL 2 (auto-bootstrap experiment, D17/D19): places/bootstrap.py, bootstrap_registry.py and test_bootstrap.py are removed. Helpers still used elsewhere moved into core_list.py (deleted, D47): Candidate, is_hub, label_for, to_entry, POPULAR_SHARE, HUB_POPULATION, HUB_TYPES, SIGNAL. build_core_list.py (deleted, D47) no longer recomputes the popular threshold from the experiment pool: it reads popular_cutoff_views and popular_pool_size from the existing data/registry/core_places.json meta (the D19 formula is frozen; values 69,032 views, pool 220). Eleven bootstrap tests that exercise kept code (wv_lists parsing and filters, overrides loading, is_hub, to_entry) moved to test_core_list.py (deleted, D47); the 11 that exercised removed code are deleted. core_list.py (deleted, D47)'s overrides code now has no caller outside the tests (it was used only by the removed experiment); kept, not deleted. Test count 452 -> 415 (37 deleted, listed in LOG.md). LOG.md entries before this one keep their old paths. Status: accepted.
D47 (2026-10-19) Minimal Phase 1 code base (owner instruction; behavior-preserving except the deletions listed). DELETED (code with no production caller once the one-off tools went): policy_pages.py, fetch_policy_pages.py; withdraw.py, withdraw_source.py (the 26 withdrawn/youtube manifest lines stay in data/); youtube.py (the orchestrator now refuses "youtube" with one entry returning skipped D27, zero requests); size_estimate.py, coverage_matrix.py, estimate_corpus_size.py; the historical scripts (build_core_list.py, gather_source_evidence.py) and therefore the --historical guard (refuse_without_historical had no caller left); check_robots.py (its result is recorded in D36/D41; the RFC 9309 matcher stays). A reachability check from the kept scripts then removed: places/core_list.py and places/wv_lists.py entirely (only the removed registry builder used them, including the overrides code), alias hygiene (clean_aliases, DroppedAlias, _is_latin, MIN_LATIN_ALIAS_LEN), WikidataClient.items_in, entities_by_qids, entities_by_wikivoyage_titles, types_many, parse_point, _POINT, normalize_title, PlaceStatus.to_dict, DiskCache.has, Open-Meteo monthly_normals, is_rainy, RAINY_DAY_MM, Overpass nearest, nearest_airport, point. KEPT although nothing calls them yet: make_chunk_id and EnrichedChunk (contracts, Phases 2-3), OllamaClient.from_settings (CLAUDE.md rule 5 entry point), HoldoutFilter with from_registry and haversine_km (its tests and mutation check are required; the Phase 2 hold-out drop can reuse it); RawRecord._source is a pydantic validator. Phase 5 must re-derive the monthly climate normals, the rainy-day rule and nearest_airport (the bundle and git history keep the old code). MERGED (no logic change): src/common/http.py = redact + requestlog + allowlist + http; src/common/schemas.py = config + schemas; src/places/places.py = registry + wikidata + resolve; src/collect/run.py = types + raw + store + coverage + ensure + orchestrator; src/collect/sources.py = wiki + web + reddit; src/facts/facts.py = robots + holdout + hosts + envelope + wikidata + overpass + openmeteo; scripts/collect.py = collect_core (core), collect_facts (facts), ensure_place_demo (demo), each with its old flags and exit codes. Renames forced by name clashes: MIN_TEXT_CHARS (reddit) -> REDDIT_MIN_TEXT_CHARS; in the facts fetchers SOURCE, URL, ATTRIBUTION, fetch_place, validate_response gained a source prefix (WIKIDATA_*, OVERPASS_*, OPEN_METEO_*, wikidata_fetch_place, open_meteo_fetch_place, overpass_validate_response, open_meteo_validate_response). Structural changes: the orchestrator imports the collectors inside build_collectors (sources.py imports run.py; a top-level import would be a cycle); every __init__.py is removed (namespace packages; pyproject pytest pythonpath ["."]); the redundant src/collect, src/common and kaggle .gitkeep files are removed. Tests: six files (test_common, test_collect, test_places, test_facts, test_hooks_scripts, test_guards); test_guards holds the tests the mutation checks rely on. pyproject: tqdm and youtube-transcript-api removed (no import left). Test count 416 -> 352 (59 test functions, 61 collected instances, deleted with their code; 5 folded into 2: the four YouTube refusal tests became one, and test_parse_point_and_window became test_pageview_window without its parse_point assertions); the list is in LOG.md. data/registry/core_places.json is now written by no code. Status: accepted.
D48 (2026-10-19) Phase 2 cleaning and chunking rules (owner-reviewed plan; no freeze in this step).
   (1) DATES (settles O9). Chunk fields `date` (publish date ONLY, else null), `date_kind` (published | revised | sitemap_lastmod | none,
   required) and `revised_date` (optional). Enforced in BaseChunk: published <=> date set; revised <=> revised_date set. How each source
   gets its kind: blog -> published (the page's own article:published_time meta, JSON-LD datePublished or <time>, extracted in Phase 1;
   src/collect/sources.py); wikipedia and wikivoyage -> revised, revised_date = the latest revision timestamp returned by the MediaWiki
   API (prop=revisions, rvprop=ids|timestamp; revid kept in attribution.json); sahapedia -> sitemap_lastmod (its `lastmod` is the
   site's sitemap <lastmod>, not a content revision; the value is kept in attribution.json, not in the chunk); none when nothing exists.
   The freshness share and every freshness metric count date_kind = published only; D3 (null = neutral) is unchanged.
   (2) TOKENS. count_tokens is pluggable: default regex proxy len(re.findall(r"\w+|[^\w\s]", text)) (no dependency, no network); when
   TOKENIZER_DIR is set (.env, read through src/common/schemas.py) the local bge-m3 tokenizer.json is loaded with `tokenizers` and a
   missing file is an error. corpus_stats records which counter produced a run. The FREEZE REQUIRES the real tokenizer.
   (3) CHUNKS. Sentences packed greedily; target 250, minimum 200, HARD maximum 300 tokens INCLUDING the overlap; overlap = whole trailing
   sentences of the previous chunk, at most 40 tokens; no overlap is carried into a chunk that starts at a section header; a sentence
   over the budget is split on words (counted). A tail under 100 tokens merges into the previous chunk only if the result is <= 300;
   otherwise it stays a short chunk (counted). chunk_id = make_chunk_id(url, character offset in the cleaned body). place and state come
   from the record's registry entry; the other Phase 3 fields stay null; context_note "".
   (4) CLEANING (pure per record). Fixed rules: the inditales share-button header, the "Recommend you to read" related-posts footer,
   whole wiki sections whose header is References, External links, See also, Further reading, Notes, Bibliography, Citations, Sources
   or Gallery, and [n] citation markers. Learned rule file config/boilerplate.json (committed, written only by `scripts/process.py
   learn`): for the hosts it lists (inditales.com only), lines under 120 characters that occur in >= 3 records of the host; parsed
   section headers are exempt. The live path applies the same file and never learns, so one record cleans identically alone or in batch.
   (5) DEDUPE. Exact: batch pre-pass over all places, key = sha256 of the cleaned text; the copy under the first place in registry order
   is kept and the duplicate is logged with both places (the inditales Ooty/Kodaikanal post stays under the earlier registry place; no
   per-chunk reassignment). Near: 5-word shingle Jaccard >= 0.8 over all chunks, earlier (place, url, offset) kept.
   (6) HOLD-OUTS (settles O8 as drop, D39 iii). HoldoutFilter (src/facts/facts.py) gains text_match(text) -> canonical hold-out name.
   A record whose URL (non-alphanumerics read as spaces) or title names a hold-out is dropped whole; otherwise each chunk whose text
   names one is dropped. Word boundaries: a concatenated token such as "Arakuvalley" does NOT match by design; the corpus stats report a
   boundary-free substring scan so such cases stay visible. Every drop is logged in data/processed/drops.jsonl with the matched term.
   (7) D23/D33 is a DRY RUN in Phase 2: the pass/fail list is computed at raw and chunk level and reported; no place is removed.
   (8) Licence/attribution: data/processed/attribution.json keyed by url (title, raw source, licence string from the raw record, revid,
   lastmod). inditales.com content: "no terms found; copyright Anuradha Goyal (personal blog)", collected under the owner's risk
   acceptance of 2026-10-18 recorded in docs/sources.md (no decision ID existed; this entry records it). O12 (sharing) stays open.
   (9) Facts envelopes (data/raw/wikidata, open_meteo, overpass) are never chunked. No chunk may have source_type live (the enum has no
   such value; tested). Status: accepted.
D48 (amendment 2026-10-19, hold-out spellings) A stem scan of the 97 raw bodies (brihad|tanj|thanj|araku|thekk|kumil|chikm|chikk) found two
   hold-out spellings outside the D39 list: "Thekkadi" (Munnar Wikivoyage, "Munnar Thekkadi route") and "Brihadeeswara" (a
   Mahabalipuram inditales footer, "Brihadeeswara Temple, Tanjore"). Both are added to DROP_NAMES (src/facts/facts.py) as Thekkady and
   Thanjavur. The other stem hits are different words (Thekkumbhagom, Chikkadpally, Mathukumilli, Gitanjali, ...). Status: accepted,
   flagged to the owner.
D48 (amendment 2026-10-19, rules added while building; owner review requested)
   (a) Inline-link joins (blog): the Phase 1 page-to-text step put inline link text on its own line ("The whole / Bangalore /
   international airport"). A line starting with a lowercase letter or , . ; : ! ? ) % is joined to the previous one, and so is a line
   of <= 40 characters whose next line starts that way. 244 joins on the snapshot. Without it the learned-line rule picked up
   "Bangalore", "Hyderabad" and "." as boilerplate (they are sentence fragments).
   (b) Image credits (blog, fixed rule): lines that are exactly "/", "Image courtesy" / "Image source", or a Shutterstock credit
   ("Shutterstock", "Shutterstock.com", "Image courtesy Shutterstock"); 228 characters.
   (c) Mid-post "Read more" / "Recommended Read" lines with more than 400 characters after them drop only themselves and the
   following line if it is under 80 characters (the linked post title); 312 characters.
   (d) Learned rule file: candidates are counted once per URL (the Ooty/Kodaikanal post was counted twice). After (a) and (b) the
   rule (>= 3 records, < 120 characters, inditales.com) yields two candidates, "Travel Tips" (4) and "Meditation Hall" (3); both are
   section headings with content after them, so they are listed under `rejected` in config/boilerplate.json and never applied.
   `learn` keeps the rejected list. Applied learned lines on the snapshot: 0. The owner may move them back.
   (e) "km." and "ft." are not abbreviations for the sentence splitter ("10 km." ends a sentence in this text).
   (f) EVAL_REFERENCE_DATE: .env has no EVAL_REFERENCE_DATE (it holds only CONTACT_EMAIL and YOUTUBE_API_KEY, key names checked, no
   values read), so scripts/process.py takes --reference-date; the runs used 2026-10-18 (the value in .env.example, D32, D39).
   load_reference_date() reads only that key, so chunking does not need the Ollama settings.
   (g) Consequence of D48 (5), recorded: the Ooty/Kodaikanal post is Kodaikanal's only blog record, so with the copy kept under Ooty
   (earlier in registry order) Kodaikanal has no blog chunk and FAILS D23 at chunk level (passes at raw level). D23 dry run: 17 of 26
   at chunk level vs 18 of 26 at raw level. The owner chose this rule; changing it (e.g. assigning that post to Kodaikanal) is a
   one-line decision before the freeze.
   (h) Borderline hold-out context, NOT dropped: Madurai Wikipedia "agricultural activity, which is fostered by the Periyar Dam"
   (the Mullaperiyar dam forms Periyar Lake at Thekkady, but "Dam" is not in the O8 context list). Owner decision.
D48 (amendment 2026-10-19, owner review of the Phase 2 draft)
   (i) Kodaikanal: the first-place-in-registry-order rule (D48 5) STAYS. The inditales post "travelogue-ooty-kodaikanal" covers
   BOTH Ooty and Kodaikanal; it is kept under Ooty only, so Kodaikanal is a chunk-level D23 fail (17 of 26 pass at chunk level,
   18 at raw level). Supersedes the open question in amendment (g).
   (ii) How Thekkadi and Brihadeeswara were found: a case-insensitive stem regex over the bodies of the 97 raw records,
   \w*(brihad|tanj|thanj|araku|thekk|kumil|chikm|chikk|chickm)\w*, listed every matching word with its count; each word was read in
   context; "Thekkadi" (Munnar Wikivoyage, "Munnar Thekkadi route") and "Brihadeeswara" (Mahabalipuram blog footer) are hold-out
   spellings outside the D39 list. Accepted by the owner, subject to the independent scan (scripts/phase2_evidence.py holdout).
   (iii) O8 AMENDED: "Periyar" counts as a Thekkady mention also next to Dam (in addition to Lake, Tiger Reserve, National Park,
   Wildlife Sanctuary, Thekkady, Kumily), and "Mullaperiyar" is a hold-out name (the dam that forms Periyar Lake). "Periyar Dam",
   "Periyar Lake" and "Mullaperiyar Dam" drop; "Periyar Bus Stand", "Periyar river" and the social reformer are kept (tests). This
   supersedes amendment (h): the Madurai "Periyar Dam" chunk is now dropped (31 text drops, 1,388 chunks).
   (iv) Reference date: EVAL_REFERENCE_DATE is read from .env when set; --reference-date overrides it; corpus_stats records the value
   and its source. The date is an input to the freshness statistic only; process_place() has no date input and chunks.jsonl is
   byte-identical with --reference-date 2026-10-18 and 2020-01-01 (sha256 ebfdbd51...). Owner action: set EVAL_REFERENCE_DATE=2026-10-18
   in .env (the value of D32/D39 and .env.example); .env was not edited.
   (v) check_corpus() (src/process/process.py) runs before every --write and must run again at the freeze: no source_type live, no
   hold-out mention (HoldoutFilter), unique ids, every chunk valid under BaseChunk.
   (vi) The near-duplicate threshold 0.8 removes nothing on this corpus: the highest 5-word-shingle Jaccard over all 1,388 kept chunks
   is 0.2155 (Bengaluru vs Chennai Wikipedia). The rule is kept for new raw data; it is not evidence of a duplicate-free source.
   (vii) ACCEPTED DEVIATIONS (owner): the plan's 10,000-20,000 chunks are unreachable from this raw volume (1,388 chunks with the
   proxy counter; 1.70 M raw characters over 97 records); source diversity is 3 source types (blog, reference, wikivoyage) with ONE
   blog host (inditales.com) and 0 official, 0 Reddit, 0 YouTube; 0 published chunks fall in the 18-month window 2025-04-18 to
   2026-10-18 (newest blog 2025-01-30). Each is accepted and stated in the report as measured. Status: accepted.
D48 (amendment 2026-10-19, owner decision on the landmark scan; freeze tooling)
   (viii) "Borra Caves" is a hold-out term of Araku Valley (DROP_NAMES and HOLDOUT_OF in src/facts/facts.py). The one chunk that named it,
   500b7e3d23575ed7 (Visakhapatnam Wikipedia), is now dropped; tests: "Borra Caves" drops, a Visakhapatnam sentence without it is kept.
   RESIDUAL RISK: hold-out content can leak through landmark names that are not on the term list (a chunk about a hold-out's sights that
   never names the hold-out). The landmark list used in the scan (Mullayanagiri, Baba Budan, Kemmangundi, Kudremukh, Borra Caves,
   Saraswathi Mahal, Big Temple) was supplied by the owner's reviewer and is UNVERIFIED; it is not a complete gazetteer, and only Borra
   Caves had a hit. The residual risk is accepted and stated in the report.
   (ix) Freeze tooling: scripts/freeze.py (default dry run) checks, with pass/fail: FROZEN absent; outputs match processed_manifest.json;
   check_corpus(); no source_type live; the counter is the real bge-m3 tokenizer (processed_manifest counter "tokenizer.json sha256:<12>"
   equal to the file under TOKENIZER_DIR; the proxy is refused); the reference date came from .env (corpus_stats reference_date_source
   "EVAL_REFERENCE_DATE in .env" and equal to .env); Reddit recorded as resolved; D33 applied. Owner decisions live in config/freeze.json
   (reddit_resolved, d33_mode). --execute writes data/processed/FROZEN (sha256 of chunks.jsonl and attribution.json, chunk count, counter,
   tokenizer directory name, reference date and source, UTC timestamp). D33 is implemented as a SIDECAR only (d33_mode "sidecar":
   data/processed/d33_excluded.json lists the excluded places, chunks.jsonl unchanged, retrieval must filter them); "exclude" (rewriting
   chunks.jsonl) is not implemented; both have the same retrievable effect (17 places, 1,143 chunks, 0 hidden_gem).
   (x) The frozen-file hook (scripts/hooks/guard_frozen.py) now blocks Edit/Write on every path under data/processed/ once
   data/processed/FROZEN exists (was chunks*.jsonl only). Consequence for Phase 3: enriched chunks cannot be written into data/processed
   after the freeze; they need another location (e.g. index/ or a new data/enriched/), decided in Phase 3 via a contracts change.
   Status: accepted.
D48 (amendment 2026-10-19, cleanup round)
   (xi) PHASE 3 LOCATION: enriched chunks are written to data/enriched/chunks.jsonl (gitignored, like data/processed, while O12 is open),
   never to data/processed. check_enriched_matches_frozen(enriched, frozen) (src/process/process.py) fails unless the chunk_id sets
   are equal and chunk_id, text, url, date, date_kind, revised_date, state, place and source_type are identical per chunk_id; tests:
   one passing case, three failing (changed text, missing id, extra id). docs/contracts.md "Locations" updated. Supersedes the open
   question in (x).
   (xii) D33 SIDECAR WORDING (corrects ix; behaviour unchanged): data/processed/d33_excluded.json is a Phase 4 QUERY-ELIGIBILITY list
   (no dev or test query may target a listed place). chunks.jsonl is unchanged and retrieval is NOT filtered, unless the owner records
   otherwise in config/freeze.json. The freeze.py dry-run table and help text say so; "chunks retrievable" was the wrong label.
   (xiii) FROZEN contents (replaces the tokenizer directory name in ix): sha256 of chunks.jsonl and attribution.json, chunk count, counter,
   sha256 of every file the counter loads from TOKENIZER_DIR (TOKENIZER_FILES = tokenizer.json only: Tokenizer.from_file reads no other
   file), the tokenizers library version, the reference date and its source, and the UTC timestamp. The same fields (no text) are
   written to data/manifests/frozen_manifest.json, which is committed at the freeze.
   (xiv) ACCEPTED COST: the chunk-level hold-out rule removes 11 Wikivoyage chunks from Munnar (a D23-failing place) through the Thekkady
   terms (Thekkady 8 incl. one lower-case, Kumily 2, Thekkadi 1); Munnar keeps 28 chunks. Accepted by the owner.
   (xv) Attribution rule for a chunk that names two hold-outs: HoldoutFilter.text_match returns the LEFTMOST name match in the chunk text
   (regex search over all names); the Periyar context rule applies only when no name matches. Only one dropped chunk names two hold-outs
   on the snapshot: 941f36c8ed383169 (Kanniyakumari Wikipedia), Thanjavur at offset 775 and Kumily at 824, attributed to Thanjavur.
   The Madurai Wikipedia and Wikivoyage records each have separate chunks for different hold-outs, not a chunk naming two
   (`python scripts/phase2_evidence.py holdoutdrops`).
