# South India Intelligent Travel Planner: Project Flow

An Information Retrieval project (agentic RAG). A traveller's preferences go in; a cited, feasible day-by-day South India itinerary comes out. The retrieval layer over a frozen corpus is what we measure; itinerary generation and the live web layer are the application on top.

Diagrams are Mermaid. They render on GitHub and in the VS Code Markdown preview (with a Mermaid extension).

---

## 1. Big picture

```mermaid
flowchart LR
    A[Offline build<br/>Phases 1-5] --> B[Retrieval engine<br/>Phase 6]
    B --> C[Agent pipeline<br/>Phase 7]
    C --> D[Streamlit app<br/>Phase 8]
    C --> E[Evaluation<br/>Phase 9]
    L[Live web search<br/>Tavily, outside the corpus] -.-> C
```

| Phases | Owner | What happens |
|---|---|---|
| 1-5 | Nivithasri | Collect data, clean and freeze corpus, enrich and index, build test collection, build routing data |
| 6-9 | Valincia | Retrieval ablations, place cards and agent pipeline, app, system evaluation |

---

## 2. Offline build (Phases 1 to 5)

```mermaid
flowchart TD
    subgraph P1[Phase 1: Collect]
        R[Place registry<br/>30 places, 4 hold-outs] --> C1[Wikivoyage / Wikipedia / Sahapedia]
        R --> C2[Travel blogs and official sites]
        C1 --> RAW[(data/raw<br/>disk-cached, polite HttpClient)]
        C2 --> RAW
    end

    subgraph P2[Phase 2: Clean, chunk, freeze]
        RAW --> CL[Remove boilerplate, dedupe,<br/>drop hold-out text]
        CL --> CH[Chunk 200-300 tokens<br/>+ overlap, date_kind tagging]
        CH --> FZ[(data/processed/chunks.jsonl<br/>1,696 chunks, FROZEN)]
    end

    subgraph P3[Phase 3: Enrich and index]
        FZ --> EN[LLM adds context_note + metadata<br/>place_type, season_tags, aspect, is_warning<br/>rule-based fallback if output is bad]
        EN --> EC[(data/enriched/chunks.jsonl)]
        EC --> BM[BM25 indexes<br/>plain + contextual]
        EC --> DN[bge-m3 dense indexes<br/>plain + contextual]
    end

    subgraph P4[Phase 4: Test collection]
        Q[40 queries<br/>13 dev / 27 test] --> POOL[Pool top 10 from 5 systems]
        BM --> POOL
        DN --> POOL
        POOL --> LAB[Human relevance labels 0/1/2<br/>kappa with second labeller]
        LAB --> QR[(qrels_dev / qrels_test)]
        QR --> RE[run_eval.py<br/>nDCG, MRR, Recall, P@5, diversity]
    end

    subgraph P5[Phase 5: Routing data]
        R --> TM[(travel_matrix.json<br/>travel_time a,b)]
        EC --> SD[(season_data.json<br/>check_season place,month)]
    end
```

**Key rule:** after the freeze, `data/processed/` never changes, so every metric stays reproducible.

---

## 3. Retrieval engine and ablations (Phase 6)

Each component was added one at a time and scored on **dev**; the **test** split ran once at the end.

```mermaid
flowchart TD
    B1[BM25 only] --> H[Hybrid: RRF of BM25 + dense]
    B2[Dense only] --> H
    H --> CTX[Contextual vs plain text]
    CTX --> QE[Query understanding + expansion]
    QE --> FI[Metadata filters]
    FI --> FR[Aspect-aware freshness]
    FR --> RR[Cross-encoder rerank top 30]
    RR --> MMR[MMR diversity]
    MMR --> CR[Corrective retrieval, 1 pass]
    CR --> SIG[Significance tests<br/>sign-flip + Holm]
    SIG --> KEEP{Keep rule<br/>delta nDCG >= -0.005}
    KEEP --> FROZEN[[Frozen retrieve<br/>= dense_contextual]]
```

Result: contextual notes helped; hybrid, expansion, filters and freshness did not beat plain dense search on dev, so `retrieve()` is frozen as **dense search over the contextual text**. The full chain is kept behind `RetrieveConfig` as a measured, non-frozen option.

---

## 4. Online pipeline (Phase 7): one request, end to end

```mermaid
flowchart TD
    U[User request<br/>days, budget, party, month,<br/>start city, interests, constraints] --> PARSE

    PARSE[1. Parse node<br/>one LLM call to structured fields,<br/>validated, registry-mapped]
    PARSE --> RET
    PARSE --> LIVE

    RET[2. Retrieve node<br/>frozen retrieve per place,<br/>blogs excluded, excluded places dropped]
    LIVE[2b. Live node, optional<br/>Tavily snippets only,<br/>closed-vocabulary queries, no personal data]

    RET --> GRADE
    LIVE -->|fails or off| NOTE[Corpus-only + visible note]
    LIVE -->|ok| GRADE
    NOTE --> GRADE

    GRADE[3. Merge and grade evidence<br/>thin evidence triggers one extra search<br/>live items kept only for plan stops or named places]
    GRADE --> PLAN

    PLAN[4. Plan node, deterministic code<br/>cluster by travel_time hours,<br/>respect daily limit, check_season,<br/>day trips, backups, rough budget]
    PLAN --> PACK

    PACK[5. Pack evidence<br/>3 items per night, interest match,<br/>one warning slot, month check]
    PACK --> WRITE

    WRITE[6. Write node<br/>one streamed LLM call,<br/>every bullet carries a source tag S# or L#]
    WRITE --> CRITIC

    CRITIC{7. Critic, rules R1-R7<br/>travel load, season, budget,<br/>citations, scope, untrusted text,<br/>can't-confirm line}
    CRITIC -->|pass| FIN
    CRITIC -->|fail| REP[One repair pass<br/>strip uncited / out-of-scope bullets,<br/>fill empty days, one LLM repair]
    REP --> FIN

    FIN[8. Finalize<br/>expand tags to chunk ids,<br/>status: passed / repaired / failed_rules]
    FIN --> OUT[Cited itinerary + sources + warnings + timings]
```

What is code and what is LLM:

| Step | Who does it |
|---|---|
| Parse | LLM (cached) |
| Retrieve, grade, plan, pack, critic | Deterministic code |
| Route, season and budget lines | Code, tagged with tool provenance |
| Day-by-day wording | LLM (streamed), restricted to cited sources |

---

## 5. App (Phase 8)

```mermaid
flowchart LR
    F[Streamlit form] --> V{Valid cities?<br/>privacy checks}
    V -->|no| ERR[Clear error]
    V -->|yes| CK{Cache hit?<br/>corpus-only only}
    CK -->|yes| SHOW
    CK -->|no| W[Worker thread<br/>runs the LangGraph pipeline]
    W -->|pieces via queue| ST[Live streaming text]
    W --> FINAL[Final answer]
    ST --> SHOW
    FINAL --> SHOW[Itinerary with S#/L# markers<br/>sources table, notices,<br/>timing panel, details expander]
```

Live-on answers are never written to disk. A "terms not verified, owner accepted risk" notice appears whenever the live step ran.

---

## 6. Evaluation (Phase 9)

```mermaid
flowchart TD
    REQ[10 trip requests] --> S1[LLM-only]
    REQ --> S2[Plain RAG<br/>dense top 12, no plan, no critic]
    REQ --> S3[Full system<br/>live off]
    REQ --> S4[Full system<br/>live on, one UTC day]

    S1 --> SH[Blind shuffled sheet]
    S2 --> SH
    S3 --> SH
    S4 --> SH
    SH --> SC[Human scores 1-5:<br/>feasibility, groundedness,<br/>freshness, personalization]
    SC --> AG[Second scorer on 10 rows<br/>agreement and kappa]
    SC --> CMP[Paired comparisons<br/>sign-flip + Holm]

    S3 --> CIT[50 sampled claims<br/>citation precision]
    S3 --> JG[Lean LLM judge<br/>faithfulness, relevancy, context precision]
    S3 --> LAT[Latency per node]
    CMP --> REP[Result tables and case studies]
    CIT --> REP
    JG --> REP
    LAT --> REP
    AG --> REP
```

Headline findings (details in `docs/decisions.md`, D130):

- Full system and plain RAG are far more grounded than LLM-only (by construction: LLM-only has no sources).
- The full system does **not** beat the baselines on feasibility or personalization.
- Live search scored **lower** than corpus-only on groundedness and freshness in this implementation (causes listed in D130).

---

## 7. Boundaries that protect the evaluation

```mermaid
flowchart LR
    FC[(Frozen corpus<br/>+ indexes + qrels)] --- RT[retrieve]
    RT --> PIPE[Pipeline]
    LV[Live results] -.never enters.-> FC
    LV --> PIPE
    PIPE --> ANS[Answer]
```

- `retrieve()` uses the frozen corpus only; live chunks are never added to the corpus, indexes, qrels or ablations.
- Dev queries tune; test queries are only reported.
- Generator and judge models differ.
- Scope is South India only (TN, KL, KA, AP, TS, PY). Hold-out places are never collected or recommended.
- Every LLM call, download, embedding and search goes through the disk cache.

---

## 8. Repository map

| Path | Contents |
|---|---|
| `data/raw`, `data/processed`, `data/enriched` | Raw downloads, frozen chunks, enriched chunks |
| `index/` | BM25 and dense indexes, travel matrix, season data, place cards |
| `src/collect`, `src/process` | Collectors, cleaning, chunking, enrichment |
| `src/retrieval` | `retrieve()`, fusion, filters, freshness, rerank, MMR, grader |
| `src/agents` | LangGraph nodes: parse, retrieve, plan, write, critic |
| `src/tools` | `travel_time`, `check_season`, `get_place_card` |
| `src/live` | `live_retrieve()` (Tavily) |
| `eval/` | Queries, qrels, metrics, ablation outputs, Phase 9 results |
| `app/` | Streamlit page |
| `docs/` | plan, contracts, decisions, handoff |
