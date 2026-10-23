# South India Intelligent Travel Planner: Project Plan

**Team:** Nivithasri A (22PD25), Valincia John (2PPD39)

A zero-cost Agentic RAG system that turns a traveler's preferences into a realistic, day-by-day South India itinerary, grounded in recent traveler content and official sources. It runs on a frozen local corpus and, on every request, adds a **live web retrieval layer** (free tier only) for current information. It is built as an Information Retrieval project: the retrieval layer over the frozen corpus is what we measure and defend, and itinerary generation, including the live layer, is the application on top.

---

## 1. Who does what

| Owner | Phases | Focus |
|---|---|---|
| **Nivithasri** | **1 to 5** | Data foundation, indexes, test collection and evaluation harness, routing reference data |
| **Valincia** | **6 to 9** | Retrieval engine, place cards and agent pipeline, app, system evaluation |

Each phase has exactly one owner. Nivithasri's half ends at the end of Phase 5, and Valincia's half starts at Phase 6.

**How the load is balanced.** The first half carries the heaviest data work (collection, cleaning, LLM enrichment, indexing) plus the test collection and the routing data. The second half carries the retrieval techniques, the agents, the app (a deliberately small phase) and the evaluation. By our effort estimates the two halves are about equal.

**Each person owns headline techniques for the resume.**
- Nivithasri: contextual enrichment of chunks, LLM metadata extraction, hybrid-ready dual indexes, the test collection with relevance judgments, the evaluation harness with relevance and diversity metrics.
- Valincia: query understanding and expansion, hybrid fusion, aspect-aware and query-aware freshness ranking, cross-encoder reranking, diversity re-ranking (MMR), corrective retrieval, significance testing, planner and critic agents, RAG evaluation.

**Cross-checks (the only shared activities).** Two small checks need a second person so we can report agreement numbers. Each of us has exactly one:
- Valincia independently labels a shared subset of the Phase 4 relevance judgments, so we can report Cohen's kappa.
- Nivithasri independently scores a sample of the Phase 9 itinerary outputs, so we can report scoring agreement.

---

## 2. System overview

**Offline build (everything heavy happens in advance):**
collect, clean, chunk, enrich with a local LLM, extract metadata, embed, index, build the travel-time matrix and season rules, build place cards.

**Online path (fast, interactive):**
1. Parse the user's request into structured fields, a time-sensitivity flag and a few expanded sub-queries (one LLM call).
2. Hybrid retrieval for each sub-query over the prebuilt indexes with filters, fused into one ranking with freshness scoring.
3. Rerank the top candidates, then apply diversity re-ranking so results mix source types and popular and lesser-known places.
4. **Live retrieval (attempted on every request):** in parallel with steps 2 and 3, run a live web search for the same sub-queries, fetch a few permitted pages, and turn them into temporary chunks tagged `source_type: "live"`. Merge them with the corpus results (see Section 2.1).
5. Grade the evidence; run one corrective search only if it is weak.
6. Build the plan with deterministic code (routing and feasibility are code, not LLM calls).
7. Write the itinerary in one streamed LLM call, with a chunk ID on every claim.
8. Critic checks the draft with rules and triggers at most one repair pass.

### 2.1 Live retrieval layer (application layer, outside the frozen corpus)

**Why.** The frozen corpus cannot hold current closures, timings, prices or crowding, and Reddit access is uncertain. A live layer gives every answer some current evidence in addition to the measured corpus.

**Compulsory, but not a hard dependency.** The live step runs on every request. If the search service fails, is rate limited or returns nothing, the system continues with the corpus alone and shows a visible "live search unavailable" note.

**Boundary rules (these protect the evaluation):**
- `retrieve()` stays frozen-corpus only. Live chunks are never added to `data/processed/chunks.jsonl`, the indexes, the test collection or the qrels, and never enter the Phase 6 ablations.
- Live chunks carry `source_type: "live"`, the fetch date, and a publish date only if one can be extracted. Their chunk IDs are marked as live so the critic and the sources panel can tell them apart.
- Live results are cached by query and date, so a rerun on the same day is reproducible.

**Constraints on the design (details are Valincia's to decide):**
- **Zero cost.** Use only a free tier or a free self-hosted option. If no search option meets this and its terms, the live layer is dropped and the decision recorded; it is not a reason to pay.
- **Terms and robots.** Scraping a search engine's result pages is not allowed unless its terms permit it. Use a search API whose terms allow our use, verified at its official page. Fetch result pages only where robots and terms allow it, through the existing polite `HttpClient` (robots check, throttle, request log, hard request cap).
- **Untrusted text.** Live pages are untrusted input to the generator (prompt-injection risk, open item O7). Clean and length-limit the text, and never treat it as instructions.
- **Scope.** Keep results to South India and the chosen destinations.
- **Budgets.** Cap results, pages fetched and total latency per request.

**Open decisions for Valincia (record each in `LOG.md`):** which search API (and its free-tier quota and terms); how many results and pages per request; how live chunks are embedded and ranked against corpus chunks (for example fusing the two lists with RRF before the grader and MMR); and whether the grader's corrective search (Phase 6, item 9) uses the same live path.

**Scope:** South India only: Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, Telangana, including places such as Pondicherry, Hampi, Coorg, Munnar and Wayanad. About 20 to 30 well-covered destinations.

**Stack (no paid APIs; everything runs locally except the live search step):**

| Layer | Choice |
|---|---|
| LLM | Ollama; pick the size by available RAM (for example a 7B/8B model on 16 GB, a 3B/4B model on 8 GB) |
| Embeddings | `bge-m3` through Ollama |
| Sparse search | `bm25s` |
| Vector store | Chroma, Qdrant local, or LanceDB |
| Reranker | `bge-reranker-v2-m3`, or a smaller MiniLM cross-encoder if CPU is slow |
| Orchestration | LangGraph with `langchain-ollama` |
| App | Streamlit |
| Live search (Phase 7) | A free-tier search API or free self-hosted option, chosen by Valincia after checking terms; pages fetched through the polite `HttpClient` |
| Evaluation | `ranx` or `pytrec_eval`, RAGAS with a local judge model |

**Repository layout:**

```
data/raw/          cached raw downloads
data/processed/    cleaned chunks (JSONL)
index/             BM25 and vector indexes, place cards, travel matrix
src/collect/       collectors
src/process/       cleaning, chunking, enrichment
src/retrieval/     bm25, dense, fusion, filters, freshness, rerank, grader
src/agents/        LangGraph nodes
src/tools/         place cards, travel_time, check_season
eval/              queries, qrels, metric scripts, ablation outputs
app/               Streamlit app
LOG.md             decisions and results
```

---

## 3. Working in parallel: contracts and early hand-offs

Valincia's half builds on Nivithasri's outputs, so we agree on formats up front and hand off early.

**Chunk record (JSONL), used everywhere:**

```json
{
  "chunk_id": "...", "text": "...", "context_note": "...",
  "source_type": "reddit|blog|youtube|wikivoyage|official|live",
  "url": "...", "date": "YYYY-MM-DD",
  "state": "...", "place": "...", "place_type": "...",
  "season_tags": ["..."],
  "aspect": "timing|closure|price|scenery|food|safety|history|other",
  "is_warning": false
}
```

**Place registry entry (Phase 1):** `name, aliases, state, region_cluster, popularity ("popular" | "hidden_gem"), lat, lon`. The planner relies on `region_cluster` and `popularity`.

**Function contracts:**

```python
retrieve(query, filters=None, k=10) -> list[dict]   # chunk fields + "score"   (owner: Valincia, Phase 6)
travel_time(place_a, place_b) -> float              # hours                    (owner: Nivithasri, Phase 5)
check_season(place, month) -> dict                  # suitability and notes    (owner: Nivithasri, Phase 5)
get_place_card(place) -> dict                       # tips, warnings, citations (owner: Valincia, Phase 7)
live_retrieve(query, places=None, k=5) -> list[dict] # live chunks, same fields as retrieve(), source_type "live" (owner: Valincia, Phase 7)
```

`source_type: "live"` appears only in live results, never in the frozen corpus.

**Early hand-offs (so nobody waits idle):**
1. **Sample set.** As soon as Nivithasri's pipeline runs end to end on a small slice of data, she shares a small enriched chunk file and its indexes. Valincia develops the Phase 6 components against this sample.
2. **Stubs.** Valincia codes agent nodes against stubs for `travel_time` and `check_season` (a few hard-coded places) until Phase 5 lands, and Nivithasri's `run_eval` works with any function that follows the `retrieve` contract.
3. **Frozen corpus.** Final ablations in Phase 6 run only after the corpus is frozen (end of Phase 2), indexed (Phase 3) and judged (Phase 4).

---

# FIRST HALF: Nivithasri (Phases 1 to 5)

## Phase 1: Data collection and place registry

**Goal:** Gather raw traveler and official content for the target destinations, cached to disk.

**Tasks**
- Build the place registry for 20 to 30 destinations with the fields in Section 3.
- Reddit collector with caching. Confirm the current API access rules first; if access is a problem, consider publicly available dumps after checking their terms.
- Wikivoyage dump extraction for South India pages.
- Web collector for state tourism sites and travel blogs from a fixed source list, with polite delays and respect for site terms.
- YouTube transcript collector (`youtube-transcript-api`), cached.
- Cache every raw response so nothing is fetched twice.
- **Hand-off for the live layer (Section 2.1):** keep `HttpClient` (robots check, throttle, request log, hard cap), the disk cache and the `live` value in `source_type` reusable by Phase 7, and tell Valincia where they are. Do not build live retrieval here.

**Outputs:** `data/raw/` for each source, the place registry file, a counts-per-source table.
**Done when:** every registry destination has raw data from at least two source types.

## Phase 2: Cleaning and chunking

**Goal:** Turn raw data into a clean, frozen chunk corpus.

**Tasks**
- Remove boilerplate and navigation text; remove exact and near duplicates.
- Filter to South India using the registry; drop off-topic content.
- Normalize dates and keep the original publish date, since freshness depends on it.
- Chunk into roughly 200 to 300 tokens with overlap; attach `source_type`, `url` and `date`.
- Produce corpus statistics (chunks per source, per state, per year) and **freeze** the corpus.

**Outputs:** `data/processed/chunks.jsonl` (about 10k to 20k chunks), statistics table.
**Done when:** the corpus is frozen. From this point it never changes, so every metric stays reproducible.

## Phase 3: Enrichment and indexing

**Goal:** Make every chunk findable by keywords and by meaning, with rich metadata.

**Tasks**
- **Contextual enrichment:** a local LLM writes a short note per chunk describing where it fits (place, topic, source). Test on a small sample first and cache every result.
- **Metadata extraction:** the LLM fills `place`, `place_type`, `season_tags`, `aspect`, `is_warning`. Validate the output format and fall back to rule-based tagging when the LLM output is malformed.
- Build the BM25 index over `context_note + text`.
- Build the dense index with `bge-m3` embeddings, in two variants: plain text and contextual text, so Phase 6 can measure the effect of enrichment.
- Share the early sample set described in Section 3 as soon as it exists.

**Outputs:** enriched `chunks.jsonl`, BM25 index, dense indexes (plain and contextual).
**Done when:** a script takes a query and returns ranked chunks from each index.

## Phase 4: Test collection and evaluation harness

**Goal:** Build the ground truth and the measurement tools that make this an IR project.

**Tasks**
- Write 30 realistic queries covering family, backpacker, honeymoon, pilgrimage, monsoon and budget trips. Split into a dev set (10) and a test set (20). Mark about 8 queries as the **freshness slice** (closures, timings, crowding), and store a `time_sensitive` true/false flag for every query. This flag is the reference for the query-sensitivity classifier in Phase 6.
- Pooling script: collect the top 10 results per query from BM25, dense, and a simple fusion of the two.
- Label every pooled result 0 (not relevant), 1 (partly), 2 (highly relevant).
- Build `eval/run_eval.py`: it accepts any function that follows the `retrieve` contract and prints nDCG@10, MRR, Recall@20, P@5, a freshness share (portion of top-k results from the last 18 months), and diversity measures (distinct source types and distinct places in the top 10). Report results overall, per query type, and for the freshness slice.
- Compute Cohen's kappa using Valincia's cross-check labels (Section 1).
- The test collection covers the frozen corpus only. Live results (Section 2.1) are never pooled or judged here, because they change daily and cannot be reproduced.

**Outputs:** `eval/queries.json`, `eval/qrels.json`, `eval/run_eval.py`, kappa score.
**Done when:** one command prints all metrics for any retrieval function.

## Phase 5: Routing reference data

**Goal:** Precompute the travel and season knowledge the planner needs, so nothing is looked up live.

**Tasks**
- Build a travel-time matrix for 40 to 60 destinations. Use distance times a road factor calibrated on about 30 known real routes, or run OSRM locally on an India extract. Avoid hammering the public OSRM demo server.
- Implement `travel_time(a, b)` reading from the matrix.
- Build season knowledge from the registry, the `season_tags` and the warning chunks, and implement `check_season(place, month)`.
- Sanity-check both functions on a handful of well-known cases (for example monsoon access to hill destinations).

**Outputs:** `index/travel_matrix.json`, season data, `src/tools/travel_time.py`, `src/tools/check_season.py`.
**Done when:** both functions return instantly for every registry place.

> **Nivithasri's half ends here.**

---

# SECOND HALF: Valincia (Phases 6 to 9)

## Phase 6: Retrieval engine and ablations

**Goal:** Build the core IR contribution and prove that each component helps.

Add one component at a time and run `eval/run_eval.py` after each. Develop against the early sample set first, then run the final numbers on the frozen corpus.

**Tasks and ablation rows**
1. **Baselines:** BM25 only, dense only.
2. **Hybrid retrieval** with Reciprocal Rank Fusion.
3. **Contextual vs plain** chunks (uses the two dense variants from Phase 3).
4. **Query understanding and expansion:** one LLM call returns structured fields, a time-sensitivity flag, and 3 to 5 sub-queries that turn vague preferences (for example "nature + culture, mid-budget") into concrete searches. Run each sub-query through hybrid retrieval and fuse the lists with RRF. Cache the output for the evaluation queries so ablations are repeatable. Check the flag against the manual `time_sensitive` labels from Phase 4.
5. **Metadata filters:** state, season, place type.
6. **Aspect-aware and query-aware freshness ranking:** the final score combines relevance with a time decay. Use fast decay for aspects that go stale (`closure`, `timing`, `price`, `safety`) and little or none for stable aspects (`scenery`, `history`). Apply the full freshness weight only to time-sensitive queries, because a recency preference helps those and hurts time-stable ones. Tune on the dev set only.
7. **Cross-encoder reranking** of the top 30 candidates.
8. **Diversity re-ranking (MMR):** balance relevance against redundancy so the top results mix source types (Reddit, blogs, video, official) and popular and lesser-known places. Report the relevance-versus-diversity trade-off.
9. **Corrective retrieval (CRAG-style grader):** trigger a corrective search when the top score is below a threshold, there are fewer than 3 distinct sources, or sources contradict each other (for example one says a trail is open and another says closed). Cache fallback results and allow one correction pass at most.
10. **Final ablation table with significance testing:** run a paired significance test across queries between consecutive variants (for example with the `ranx` comparison tools) and report which gains are statistically significant. Then freeze `retrieve()`.

**Live results stay out of this phase.** `retrieve()` and every ablation row use the frozen corpus only. The corrective search in item 9 may be pointed at the live layer in Phase 7, but its effect is reported separately from the ablation table.

**Outputs:** `src/retrieval/` modules, ablation table (CSV and markdown), the final `retrieve()`.
**Done when:** the ablation table is complete and `retrieve()` is frozen at the pipeline the pre-stated keep rule leaves standing (amended by D97: `dense_contextual`, ranked contextual dense search). Filters, freshness, reranking, MMR and the corrective pass exist as measured, non-frozen options behind `RetrieveConfig` (`FULL_CONFIG` is the measured full chain and runs on the eval queries only until a live parser and rewriter are injected).

## Phase 7: Place cards and agent pipeline

**Goal:** Turn a user request into a cited, feasible itinerary.

**Tasks**
- **Place cards:** for each destination, a precomputed summary of recent tips, warnings, best season, and popular and hidden spots, each with chunk citations. Spot-check manually. Implement `get_place_card(place)`.
- **LangGraph pipeline:**
  - *Query node:* reuses the Phase 6 query-understanding component (structured JSON with duration, budget tier, interests, dates, start city and constraints, plus the sub-queries and time-sensitivity flag), with validation and defaults.
  - *Retrieve node:* calls `retrieve()` with filters from the parsed query, applies the grader, at most one repair loop.
  - *Live retrieve node:* calls `live_retrieve()` on every request (Section 2.1), in parallel with the Retrieve node. It searches through a free, terms-compliant search API, fetches a few pages only where robots allows, cleans and chunks them, tags them `source_type: "live"`, and caches by query and date. Its results are merged with the corpus results before the grader and MMR. Failure, quota exhaustion or an empty result degrades to corpus-only with a visible warning.
  - *Plan node (deterministic):* cluster destinations by region, order them using the travel matrix, respect trip length and daily travel limits, and mix popular and hidden places.
  - *Write node:* one streamed LLM call that writes the day-by-day itinerary from the plan and cited chunks. Every claim carries a chunk ID.
  - *Critic node (rules):* check daily travel load, season conflicts via `check_season`, budget fit, and missing citations (live chunk IDs count as citations). On failure, run one repair pass, then finalize.
- Graph wiring, a command-line runner, and per-node timing logs (including the live step).
- Record the live-layer decisions listed in Section 2.1 in `LOG.md`.

**Outputs:** `index/place_cards.json`, `src/agents/`, a CLI that runs a query end to end.
**Done when:** a query returns a cited itinerary end to end, with practical tips and warnings, a rough budget estimate, and backup alternatives, using live evidence when the live step succeeds and degrading cleanly when it does not.

**Amendments recorded in docs/decisions.md D99 to D104 (2026-10-23):**
- Retrieval is the frozen `retrieve()` (dense_contextual); `FULL_CONFIG` is not used (D99).
- The live node uses search snippets only (no result-page fetching: the D43 allowlist refuses unlisted hosts), a same-day cache, minimal stored fields, no personal data in queries, and a terms gate (`terms_status` in `config/live.json`, currently NOT VERIFIED); live results are not reproducible after the day they were fetched (D100).
- The popular/hidden mix is a relative visibility tercile among the 17 eligible destinations (all are labelled popular); excluded places only when named (D101).
- Place cards are extractive, offline, without an LLM, and cannot claim recency (D102).
- Route, season and budget lines are computed by code and tagged with tool provenance; every sentence the model writes carries a chunk alias (D103).
- The grader step of the Retrieve node is a deterministic evidence grade plus one extra search; rules R1 to R7 in the critic (D104).

## Phase 8: App

**Goal:** A simple local demo. This is intentionally a small phase.

**Tasks**
- One Streamlit page: input form (duration, budget tier, style, start city, dates, interests, constraints), streamed itinerary, a sources panel with clickable URLs, warnings and recent items highlighted. Live sources are labelled as live with their fetch date, and the page shows a visible note when live search was unavailable.
- A query cache and a collapsible timing breakdown.

**Outputs:** `app/streamlit_app.py`, README with run instructions.
**Done when:** the demo runs locally. Corpus retrieval and generation need no internet; the live step needs internet and a search API key from `.env`, and the demo still works from the corpus and cache if it is unavailable.

## Phase 9: Evaluation and analysis

**Goal:** Show that the full system beats simple baselines, and explain where it fails.

**Tasks**
- **Itinerary comparison** on 10 queries across three systems: LLM-only, plain RAG (dense only, no extras), and the full system. Score each output 1 to 5 on feasibility, groundedness, freshness and personalization.
- **RAGAS:** faithfulness, answer relevancy, context precision, context recall, using a judge model different from the generator. Treat these as supporting evidence next to the human relevance labels and scores.
- **Citation precision:** sample 50 claims and check that each is supported by its cited chunk.
- **Live layer comparison:** run the full system with and without the live step on the same 10 queries (including the freshness-slice ones), scored on freshness, groundedness and feasibility. Report how often the live step returned results, its latency, and citation precision for live chunks separately. This is an application-level comparison, not part of the pooled-qrels evaluation.
- **Latency table:** per node and total, with and without the live step.
- **Case studies:** 3 to 4 failure cases and 2 to 3 cases where freshness or corrective retrieval changed the answer.
- Collect Nivithasri's cross-check scores and report agreement.

**Outputs:** result tables, case studies, latency table.
**Done when:** all tables and cases are ready for the report.

---

## 4. Dependencies at a glance

- Phase 2 (freeze) must finish before Phase 3, which must finish before the final numbers in Phase 6.
- Phase 4 pooling needs the BM25 and dense indexes from Phase 3. Phase 6 needs `run_eval` and the qrels from Phase 4.
- Phase 7's planner needs `travel_time` and the registry fields (Phases 1 and 5); the critic needs `check_season` (Phase 5). Use stubs until they land.
- Phase 9 needs the wired pipeline from Phase 7.
- The live layer (Section 2.1) depends only on Phase 1's `HttpClient`, cache and request log and on Phase 7's pipeline. It does not block Phases 2 to 6, and Phases 2 to 6 do not use it.

---

## 5. IR concepts covered, and where each is evaluated

This table is the checklist for the IR submission: every concept has a phase and a piece of evidence for the report. It also explains the terms for anyone new to the IR side.

| IR concept | What we do | Phase | Evidence for the report |
|---|---|---|---|
| **Query understanding and expansion** | One LLM call turns a vague request into structured fields, a time-sensitivity flag and 3 to 5 sub-queries, fused with RRF | 6 (reused in 7) | Ablation row; flag accuracy against manual labels |
| **Sparse retrieval (BM25)** | Keyword ranking over context-enriched chunks; strong on exact terms such as place names and "closed" | 3, 6 | Baseline row |
| **Dense retrieval** | Embedding search with `bge-m3`; strong on meaning when wording differs | 3, 6 | Baseline row |
| **Hybrid retrieval and rank fusion** | Combine both rankings with Reciprocal Rank Fusion so each covers the other's weakness | 6 | Ablation row |
| **Contextual enrichment** | Add a short "where this chunk fits" note before indexing so isolated chunks are not ambiguous | 3, 6 | Plain vs contextual row |
| **Metadata and location filtering** | Registry-based South India boundary; filters on state, season, place type | 1 to 3, 6 | Ablation row |
| **Freshness-aware ranking** | Time decay by aspect (fast for closures, timings, prices; none for scenery, history), applied only to time-sensitive queries | 6 | Freshness-slice results and freshness share |
| **Reranking** | Cross-encoder rescoring of the top 30 candidates | 6 | Ablation row |
| **Diversity and multi-source fusion** | MMR re-ranking mixing Reddit, blogs, video, official pages and popular with lesser-known places | 6 | Relevance-vs-diversity trade-off, diversity metrics |
| **Corrective and iterative retrieval** | CRAG-style grader: weak, thin or contradictory evidence triggers one cached corrective search | 6 | Ablation row; case studies |
| **Source attribution** | Chunk ID on every claim, sources panel with links | 7, 8, 9 | Citation precision on 50 sampled claims |
| **Test collection and IR evaluation** | Pooled relevance judgments (0/1/2), nDCG@10, MRR, Recall@20, P@5, Cohen's kappa, paired significance tests | 4, 6 | Ablation tables with significance |
| **RAG generation evaluation** | RAGAS (faithfulness, answer relevancy, context precision, context recall) plus human itinerary scores, against LLM-only and plain-RAG baselines | 9 | Comparison tables |
| **Live (federated) web retrieval** | A free, terms-compliant search API plus a few permitted pages per request, chunked on the fly, tagged `live`, cached by query and date, fused with corpus results; kept outside the frozen corpus and qrels | 7, 9 | With/without comparison on freshness and groundedness; live citation precision; latency |
| **Agentic RAG patterns** | Planning (query decomposition), tool use (travel, season and place-card functions), reflection (critic with one repair pass) | 6, 7 | Pipeline description; case studies |

---

## 6. Rules we both follow

1. Freeze the corpus after Phase 2 and never change it.
2. Measure after every retrieval change with `run_eval.py`.
3. Tune only on the dev set; report only the test set.
4. Cache everything to disk: downloads, LLM enrichment, embeddings, fallback searches.
5. The generator model and the judge model must be different.
6. Every claim in an itinerary must carry a chunk ID; the critic rejects claims without one.
7. Keep scope strictly South India and the chosen destinations.
8. Never cut the frozen corpus, the test collection or the ablation table. They are what make this an IR project.
9. Record decisions and results in `LOG.md`.
10. Live results never enter the frozen corpus, the indexes, the test collection or the ablations. Any live search must be free, comply with the source's terms, and go through the polite `HttpClient`.

**Stretch goals (only if everything above is done):** BGE-M3 sparse vectors as an extra ablation row, agent-chosen tool selection in the retrieve step (letting the model pick between search, place-card and season tools instead of a fixed graph), and a pre-generated itinerary cache for popular trip types.

---

## 7. Final deliverables

- Working local app (Streamlit) that returns a cited itinerary, with a live search layer that degrades cleanly to the corpus.
- Frozen corpus, indexes, place cards, travel matrix.
- Test collection (queries and relevance judgments) with agreement score.
- Retrieval ablation table and itinerary-level evaluation tables with case studies.
- Report and demo recording. Each person writes the sections that match their phases; the introduction, related work and conclusion are written together.