# Phase Flowcharts

Each chart has a **Code map** table under it with clickable links to the code (Ctrl+click in VS Code). The chart nodes also carry `click` links where the viewer allows them (some viewers, including GitHub, disable them; the tables always work). Line numbers are as of the current branch and may drift.

## Phase 1: Data collection and place registry

```mermaid
flowchart TD
    A[Core place list] --> B[Resolve via Wikidata<br/>scope check TN KL KA AP TS PY]
    B --> C[(Place registry<br/>30 places, 4 hold-outs)]
    C --> D1[Wikivoyage / Wikipedia]
    C --> D2[Sahapedia]
    C --> D3[Travel blogs]
    C --> D4[Official sites]
    C --> D5[Facts: Wikidata, Open-Meteo, Overpass]
    D1 --> E[Polite HttpClient<br/>allowlist, robots, throttle, cap]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    E --> F[(Disk cache + request log)]
    F --> G[(data/raw)]
    G --> H{Coverage rule D23<br/>2+ source types, 1 non-wiki}
    H --> I[Coverage report]

    click B href "../src/places/places.py#L331" "resolve_place"
    click C href "../src/places/places.py#L38" "Registry"
    click D1 href "../src/collect/sources.py#L53" "WikiCollector"
    click D3 href "../src/collect/sources.py#L212" "WebCollector"
    click D5 href "../src/facts/facts.py" "facts fetchers"
    click E href "../src/common/http.py#L306" "HttpClient"
    click F href "../src/common/cache.py#L61" "DiskCache"
    click H href "../src/collect/run.py#L236" "passes_rule"
    click I href "../scripts/coverage_report.py" "coverage_report"
```

| Node | Code |
|---|---|
| Resolve via Wikidata, scope check | [resolve_place](../src/places/places.py#L331), [WikidataClient](../src/places/places.py#L223) |
| Place registry | [Registry](../src/places/places.py#L38) |
| Wikivoyage / Wikipedia | [WikiCollector](../src/collect/sources.py#L53) |
| Travel blogs, official sites | [WebCollector](../src/collect/sources.py#L212), [allowlist config loader](../src/collect/sources.py#L125) |
| Reddit (access pending) | [RedditCollector](../src/collect/sources.py#L368) |
| Facts (Wikidata, Open-Meteo, Overpass) | [facts.py](../src/facts/facts.py) |
| Polite HttpClient, allowlist, request log | [HttpClient](../src/common/http.py#L306), [Allowlist](../src/common/http.py#L117), [RequestLog](../src/common/http.py#L73) |
| Disk cache | [DiskCache](../src/common/cache.py#L61) |
| Coverage rule D23 | [passes_rule](../src/collect/run.py#L236), [place_coverage](../src/collect/run.py#L240) |
| Entry points | [collect.py](../scripts/collect.py#L539), [coverage_report.py](../scripts/coverage_report.py#L26), [build_collectors](../src/collect/run.py#L504) |

## Phase 2: Cleaning and chunking

```mermaid
flowchart TD
    A[(data/raw)] --> B[Clean boilerplate,<br/>citations, junk sections]
    B --> C[Exact + near dedupe]
    C --> E[Tag date_kind<br/>published / revised / none]
    E --> F[Chunk 200-300 tokens<br/>bge-m3 tokenizer, overlap]
    F --> G[Corpus stats]
    G --> H[D33 excluded-places list]
    H --> I{freeze.py<br/>8 checks pass?}
    I -->|yes| J[(chunks.jsonl FROZEN<br/>1,696 chunks)]
    I -->|no| B

    click B href "../src/process/process.py#L199" "clean_record"
    click C href "../src/process/process.py#L466" "dedupe_records"
    click E href "../src/process/process.py#L395" "date_fields"
    click F href "../src/process/process.py#L347" "chunk_text"
    click I href "../scripts/freeze.py#L86" "preconditions"
```

| Node | Code |
|---|---|
| Clean boilerplate | [clean_record](../src/process/process.py#L199), [learn_boilerplate](../src/process/process.py#L240) |
| Exact + near dedupe | [dedupe_records](../src/process/process.py#L466), [near_dups](../src/process/process.py#L494) |
| Drop hold-out text | [record_holdout](../src/process/process.py#L406), [HoldoutFilter](../src/facts/facts.py#L133) |
| Tag date_kind | [date_fields](../src/process/process.py#L395) |
| Chunk 200-300 tokens | [chunk_text](../src/process/process.py#L347), [load_counter](../src/process/process.py#L40) |
| Per-place pipeline | [process_place](../src/process/process.py#L427) |
| Corpus checks | [check_corpus](../src/process/process.py#L580) |
| Freeze (8 checks) | [preconditions](../scripts/freeze.py#L86), [run](../scripts/freeze.py#L160), [d33_table](../scripts/freeze.py#L131) |
| Driver | [scripts/process.py](../scripts/process.py#L241) |

## Phase 3: Enrichment and indexing

```mermaid
flowchart TD
    A[(Frozen chunks)] --> C[LLM: context_note + metadata<br/>prompt enrich-v4]
    C --> D{Valid, grounded,<br/>no hold-out, no copy?}
    D -->|yes| E[Model-written row]
    D -->|no| F[Fallback row]
    E --> G[(data/enriched/chunks.jsonl)]
    F --> G
    G --> H1[BM25 plain]
    G --> H2[BM25 contextual]
    G --> H3[bge-m3 dense plain]
    G --> H4[bge-m3 dense contextual]
    H1 --> I[(index/)]
    H2 --> I
    H3 --> I
    H4 --> I
    I --> J[query_index.py<br/>ranked chunks per index]

    click C href "../src/process/enrich.py#L373" "enrich_chunk"
    click G href "../src/process/process.py#L558" "check_enriched_matches_frozen"
    click H1 href "../src/retrieval/build_index.py#L85" "build_bm25"
    click H2 href "../src/retrieval/build_index.py#L85" "build_bm25"
    click H3 href "../src/retrieval/build_index.py#L141" "build_dense"
    click H4 href "../src/retrieval/build_index.py#L141" "build_dense"
    click J href "../scripts/query_index.py#L42" "query_index"
```

| Node | Code |
|---|---|
| LLM enrichment (prompt, validators, fallbacks) | [enrich_chunk](../src/process/enrich.py#L373), driver [scripts/enrich.py](../scripts/enrich.py#L257) |
| LLM client (hosted API, rate limiter) | [OpenAICompatClient](../src/common/llm.py#L710), [RateLimiter](../src/common/llm.py#L528) |
| Enriched file matches frozen | [check_enriched_matches_frozen](../src/process/process.py#L558) |
| BM25 indexes | [build_bm25](../src/retrieval/build_index.py#L85), [bm25_text](../src/retrieval/build_index.py#L60) |
| Dense indexes | [build_dense](../src/retrieval/build_index.py#L141), [dense_text](../src/retrieval/build_index.py#L64), [load_encoder](../src/retrieval/build_index.py#L113) |
| Manifest and verify | [write_manifest](../src/retrieval/build_index.py#L217), [verify_index](../src/retrieval/build_index.py#L234) |
| Index build script | [scripts/build_index.py](../scripts/build_index.py#L65) |
| Query one index | [scripts/query_index.py](../scripts/query_index.py#L42), [Index](../src/retrieval/search.py#L45) |

## Phase 4: Test collection and evaluation harness

```mermaid
flowchart TD
    A[40 queries<br/>13 dev / 27 test] --> B[Pool top 10 from<br/>BM25 x2, dense x2, RRF]
    I[(Indexes)] --> B
    B --> C[Blind judging sheets]
    C --> D[Owner labels 0/1/2]
    C --> E[Partner labels 100 rows]
    D --> F[(qrels_dev / qrels_test)]
    D --> K[Cohen kappa 0.825]
    E --> K
    F --> G[run_eval.py]
    H[Any retrieve function] --> G
    G --> M[nDCG, MRR, Recall, P@5,<br/>freshness, diversity]

    click A href "../eval/queries.json" "queries.json"
    click B href "../eval/pooling.py#L37" "build_pool"
    click C href "../eval/pooling.py#L115" "sheet_rows"
    click F href "../eval/qrels.py#L71" "load_qrels"
    click K href "../scripts/kappa.py#L52" "kappa"
    click G href "../eval/run_eval.py" "run_eval"
    click M href "../eval/metrics.py#L47" "ndcg_at_k"
```

| Node | Code |
|---|---|
| Queries | [eval/queries.json](../eval/queries.json), checker [check_queries.py](../scripts/check_queries.py#L41) |
| Pooling | [build_pool](../eval/pooling.py#L37), [scripts/pool.py](../scripts/pool.py#L58) |
| Blind sheets, label import | [sheet_rows](../eval/pooling.py#L115), [parse_label](../eval/pooling.py#L150), [verify_unchanged](../eval/pooling.py#L157), [import_labels.py](../scripts/import_labels.py#L79) |
| Qrels loaders (test split guarded) | [load_qrels](../eval/qrels.py#L71), [load_qrels_merged](../eval/qrels.py#L94) |
| Cohen's kappa | [kappa](../scripts/kappa.py#L52), [compute](../scripts/kappa.py#L128) |
| Harness | [eval/run_eval.py](../eval/run_eval.py) |
| Metrics | [ndcg_at_k](../eval/metrics.py#L47), [reciprocal_rank](../eval/metrics.py#L59), [recall_at_k](../eval/metrics.py#L76), [freshness_share](../eval/metrics.py#L101), [diversity](../eval/metrics.py#L116) |

## Phase 5: Routing reference data

```mermaid
flowchart TD
    A[(Registry 26 places)] --> C[48 nodes]
    B[(OSM airports<br/>33 IATA allowlist)] --> H[22 hub nodes<br/>no military]
    H --> C
    C --> D[Great-circle km x road factor]
    CAL[30 reference routes] --> D
    D --> E[Two-band speed + ghat surcharge]
    E --> F[(travel_matrix.json)]
    F --> G[travel_time a, b]
    W[(Open-Meteo ERA5 normals)] --> S[Month verdict good/fair/poor]
    T[(Season tags + warning chunks)] --> S
    S --> V[(season_data.json)]
    V --> X[check_season place, month]

    click B href "../config/airports_allowlist.json" "airports_allowlist"
    click CAL href "../config/calibration_routes.json" "calibration_routes"
    click E href "../scripts/build_routing.py#L95" "build_matrix_doc"
    click F href "../index/travel_matrix.json" "travel_matrix.json"
    click G href "../src/tools/travel_time.py#L62" "travel_time"
    click S href "../scripts/build_routing.py#L109" "build_season_doc"
    click V href "../index/season_data.json" "season_data.json"
    click X href "../src/tools/check_season.py#L91" "check_season"
```

| Node | Code |
|---|---|
| Hubs and airport allowlist | [airports_allowlist.json](../config/airports_allowlist.json), [routing_params.json](../config/routing_params.json) |
| Calibration routes | [calibration_routes.json](../config/calibration_routes.json), [cmd_calibrate](../scripts/build_routing.py#L86) |
| Build matrix | [build_matrix_doc](../scripts/build_routing.py#L95), [cmd_matrix](../scripts/build_routing.py#L119) |
| Build season data | [build_season_doc](../scripts/build_routing.py#L109), [cmd_season](../scripts/build_routing.py#L128) |
| Sanity table | [cmd_sanity](../scripts/build_routing.py#L134) |
| Outputs | [travel_matrix.json](../index/travel_matrix.json), [season_data.json](../index/season_data.json) |
| Lookups | [travel_time](../src/tools/travel_time.py#L62), [check_season](../src/tools/check_season.py#L91) |

## Phase 6: Retrieval engine and ablations

```mermaid
flowchart TD
    A[BM25] --> C[Hybrid RRF]
    B[Dense] --> C
    C --> D[Contextual vs plain]
    D --> E[Query expansion]
    E --> F[Metadata filters]
    F --> G[Freshness]
    G --> H[Rerank top 30]
    H --> I[MMR]
    I --> J[Corrective pass]
    J --> K[Round-2 pooling + chat labels]
    K --> L[Significance tests<br/>sign-flip + Holm]
    L --> M{Keep rule<br/>delta >= -0.005}
    M --> N[[Frozen retrieve<br/>dense_contextual]]
    N --> O[Test split run once]

    click A href "../src/retrieval/build_index.py#L102" "bm25_scores"
    click B href "../src/retrieval/search.py#L45" "Index"
    click C href "../src/retrieval/fusion.py#L14" "rrf"
    click E href "../src/retrieval/query_understanding.py#L286" "parse_query"
    click F href "../src/retrieval/filters.py#L130" "partition"
    click G href "../src/retrieval/freshness.py#L49" "freshness rerank"
    click H href "../src/retrieval/rerank.py#L93" "CrossEncoderScorer"
    click I href "../src/retrieval/mmr.py#L69" "apply_mmr"
    click J href "../src/retrieval/grader.py#L74" "fired"
    click K href "../eval/pooling2.py" "pooling2"
    click L href "../eval/significance.py#L40" "randomization_test"
    click N href "../src/retrieval/retrieve.py#L269" "retrieve"
    click O href "../scripts/test_run.py#L141" "test_run"
```

| Node | Code |
|---|---|
| BM25 / dense search | [bm25_scores](../src/retrieval/build_index.py#L102), [Index](../src/retrieval/search.py#L45), [hybrid_ids](../src/retrieval/hybrid.py#L19) |
| Hybrid RRF | [rrf](../src/retrieval/fusion.py#L14) |
| Query expansion | [parse_query](../src/retrieval/query_understanding.py#L286), [validate](../src/retrieval/query_understanding.py#L251), [expanded_ids](../src/retrieval/hybrid.py#L26) |
| Metadata filters | [Filters](../src/retrieval/filters.py#L40), [partition](../src/retrieval/filters.py#L130) |
| Freshness | [freshness_factor](../src/retrieval/freshness.py#L29), [rerank](../src/retrieval/freshness.py#L49) |
| Cross-encoder rerank | [apply_scores](../src/retrieval/rerank.py#L31), [CrossEncoderScorer](../src/retrieval/rerank.py#L93) |
| MMR | [mmr_order](../src/retrieval/mmr.py#L50), [apply_mmr](../src/retrieval/mmr.py#L69), [select_lambda](../src/retrieval/mmr.py#L82) |
| Corrective pass (CRAG-style) | [fired](../src/retrieval/grader.py#L74), [contradiction](../src/retrieval/grader.py#L60), [generate_rewrite](../src/retrieval/grader.py#L125) |
| Ablation driver | [eval/ablation.py](../eval/ablation.py#L151), [variants.py](../eval/variants.py) |
| Round-2 pooling | [eval/pooling2.py](../eval/pooling2.py), [scripts/pool2.py](../scripts/pool2.py#L132) |
| Significance | [randomization_test](../eval/significance.py#L40), [bootstrap_ci](../eval/significance.py#L72), [holm](../eval/significance.py#L88) |
| Frozen retrieve() | [retrieve](../src/retrieval/retrieve.py#L269), [retrieve_with_trace](../src/retrieval/retrieve.py#L258), [frozen_config.json](../eval/frozen_config.json) |
| Equivalence check, test run | [verify_retrieve.py](../scripts/verify_retrieve.py#L90), [test_run.py](../scripts/test_run.py#L141) |

## Phase 7: Place cards and agent pipeline

```mermaid
flowchart TD
    U[Request] --> P[Parse node<br/>one LLM call]
    P --> R[Retrieve node<br/>frozen retrieve per place]
    P --> L[Live node<br/>Tavily snippets]
    L -->|fail or off| N[Corpus-only note]
    R --> G[Merge + grade evidence]
    L --> G
    N --> G
    G -->|thin| X[One extra search]
    X --> G
    G --> PL[Plan node<br/>travel_time, check_season, budget]
    PC[(place_cards.json)] --> PK
    PL --> PK[Pack evidence]
    PK --> W[Write node<br/>streamed LLM, S# L# tags]
    W --> C{Critic R1-R7}
    C -->|pass| F[Finalize]
    C -->|fail| RP[One repair pass]
    RP --> F
    F --> O[Cited itinerary]

    click U href "../scripts/run_pipeline.py#L77" "CLI"
    click P href "../src/agents/nodes.py#L111" "parse_node"
    click R href "../src/agents/nodes.py#L145" "retrieve_node"
    click L href "../src/agents/nodes.py#L198" "live_node"
    click G href "../src/agents/nodes.py#L212" "merge_grade_node"
    click PL href "../src/agents/nodes.py#L245" "plan_node"
    click PC href "../src/tools/place_cards.py#L240" "get_place_card"
    click PK href "../src/agents/evidence.py#L353" "pack"
    click W href "../src/agents/nodes.py#L265" "write_node"
    click C href "../src/agents/critic.py#L139" "check"
    click RP href "../src/agents/critic.py#L261" "deterministic_fix"
    click F href "../src/agents/nodes.py#L322" "finalize_node"
```

| Node | Code |
|---|---|
| Wiring (LangGraph and sequential fallback) | [run_graph](../src/agents/graph.py#L105), [run_local](../src/agents/graph.py#L63), CLI [run_pipeline.py](../scripts/run_pipeline.py#L77) |
| Parse node | [parse_node](../src/agents/nodes.py#L111), [parse_query](../src/retrieval/query_understanding.py#L286) |
| Retrieve node | [retrieve_node](../src/agents/nodes.py#L145), [retrieve](../src/retrieval/retrieve.py#L269) |
| Live node | [live_node](../src/agents/nodes.py#L198), [live_retrieve](../src/live/search.py#L218), [live config and terms gate](../src/live/config.py), [closed-vocabulary queries](../src/live/queries.py) |
| Merge + grade, extra search | [merge_grade_node](../src/agents/nodes.py#L212), [scope.py](../src/agents/scope.py) |
| Plan node | [plan_node](../src/agents/nodes.py#L245), [plan_trip](../src/agents/planner.py#L164), [budget estimate](../src/agents/budget.py#L12), [travel_time](../src/tools/travel_time.py#L62), [check_season](../src/tools/check_season.py#L91) |
| Place cards | [get_place_card](../src/tools/place_cards.py#L240), builder [build_place_cards.py](../scripts/build_place_cards.py#L55) |
| Pack evidence | [pack](../src/agents/evidence.py#L353) |
| Write node (streamed) | [write_node](../src/agents/nodes.py#L265), [write](../src/agents/writer.py#L138), [OpenAICompatClient](../src/common/llm.py#L710) |
| Critic R1-R7, repair | [critic_node](../src/agents/nodes.py#L289), [check](../src/agents/critic.py#L139), [deterministic_fix](../src/agents/critic.py#L261) |
| Finalize | [finalize_node](../src/agents/nodes.py#L322) |
| State and request parsing | [state.py](../src/agents/state.py), [deps.py](../src/agents/deps.py) |
| App (Phase 8) | [streamlit_app.py](../app/streamlit_app.py), [state.py](../app/state.py), [query_cache.py](../app/query_cache.py) |
