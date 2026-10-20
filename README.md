# South India Intelligent Travel Planner
Agentic RAG itinerary planner built as an Information Retrieval project. The plan is in
[docs/plan.md](docs/plan.md); the current state of each phase is in [docs/phase-status.md](docs/phase-status.md).

## Install
Python 3.10 (see D9 in [docs/decisions.md](docs/decisions.md)).

```sh
pip install -e .[dev]
git config core.hooksPath .githooks      # or: sh scripts/setup_git.sh
cp .env.example .env                     # then fill in the values (never commit .env)
```

`.env` needs LLM_API_BASE_URL, LLM_API_KEY, GENERATOR_MODEL, JUDGE_MODEL (must differ from the generator), the four
LLM_RPM/TPM/RPD/TPD limits (D59), EVAL_REFERENCE_DATE, TOKENIZER_DIR, EMBED_LOCAL_DIR (the local bge-m3 folder) and
CONTACT_EMAIL (sent in the User-Agent of every request). OLLAMA_BASE_URL is only for the legacy Ollama client (smoke_ollama.py).
REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and YOUTUBE_API_KEY are optional. Never print or commit keys or the API URL.

## Test and lint
`ruff` and `pytest` may not be on PATH, so call them through Python:

```sh
python -m pytest -q
python -m ruff check .
```

## Run (Phase 1 tools)
Everything that touches the network prints its plan first and only runs with `--execute`. Every request goes
through the polite, cached, allowlisted HTTP client (`src/common/http.py`, `config/sources.json`).

| Command | What it does |
|---|---|
| `python scripts/collect.py core --places <names>` | plan a collection run for core places; add `--execute` to run it |
| `python scripts/collect.py facts --source <wikidata\|open_meteo\|overpass> --places <names>` | facts-layer envelopes; add `--execute` |
| `python scripts/collect.py demo --place <name>` | never-seen-place demo into `data/live/` only; add `--execute` |
| `python scripts/coverage_report.py` | D23 coverage per place (offline) |
| `python scripts/smoke_ollama.py` | check the live Ollama endpoint (run before Phase 3) |

`python scripts/collect.py <subcommand> --help` lists the flags. Exit codes: 2 hold-out or bad input, 3 block,
4 request cap reached, 5 (facts) a fetch error stopped the run.

## Run (Phase 3 tools: enrichment and indexes)
The frozen corpus (`data/processed`) is read-only. Enriched chunks go to `data/enriched/`, indexes to `index/` (both gitignored, O12).

| Command | What it does |
|---|---|
| `python scripts/enrich.py --full --eligible-only` | enrich the 1,100 non-blog chunks of the 17 eligible places with the hosted API; resumable (checkpoint + disk cache), so re-running after a quota stop (exit 5) continues where it stopped. `--max-wait`, `--tpd-share`, `--trust-provider` tune the rate-limit handling (D65-D67); `--include-blog` stays off |
| `python scripts/enrich.py --report --full --eligible-only` | counts, fallback reasons and the call reconciliation (offline) |
| `python scripts/enrich.py --assemble --eligible-only` | write `data/enriched/chunks.jsonl` and `provenance.jsonl` from the checkpoint (offline) |
| `python scripts/build_index.py --model-dir models/bge-m3-full` | build BM25 (contextual, plain) and dense (plain, contextual) indexes on CPU (about 75 min); `--verify` checks them against the manifest |
| `python scripts/query_index.py "<query>"` | print the top 10 from the four indexes (needs EMBED_LOCAL_DIR in `.env` or `--model-dir`) |
| `python scripts/label_sample.py` / `--score` | write the 30-chunk hand-label sheet / score your labels against the model (offline) |
| `python scripts/mutation_checks.py` | apply 19 one-line mutations and confirm a test fails for each; `git diff --stat` must be empty afterwards |

## Layout
`src/common/{http,cache,llm,schemas}.py` (HTTP client with allowlist and request log; disk cache; Ollama client;
settings and contract models), `src/collect/sources.py` (collectors) and `src/collect/run.py` (raw and live
stores, coverage, ensure_place, orchestrator), `src/places/places.py` (registry, Wikidata, resolver),
`src/facts/facts.py` (facts layer); `tests/` has six files. `src/process`, `src/retrieval`, `src/agents`, `src/tools`,
`eval`, `app`, `index` are filled in later phases. `data/raw` and `data/processed` are gitignored; the frozen
corpus rules are in [CLAUDE.md](CLAUDE.md). Results and the commands that reproduce them are in [LOG.md](LOG.md);
decisions in [docs/decisions.md](docs/decisions.md); contracts in [docs/contracts.md](docs/contracts.md);
the session handoff in [docs/HANDOFF.md](docs/HANDOFF.md).
