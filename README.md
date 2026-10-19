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

`.env` needs OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL (must differ from the generator), EMBED_MODEL,
EVAL_REFERENCE_DATE and CONTACT_EMAIL (sent in the User-Agent of every request). REDDIT_CLIENT_ID,
REDDIT_CLIENT_SECRET and YOUTUBE_API_KEY are optional.

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

## Layout
`src/common/{http,cache,llm,schemas}.py` (HTTP client with allowlist and request log; disk cache; Ollama client;
settings and contract models), `src/collect/sources.py` (collectors) and `src/collect/run.py` (raw and live
stores, coverage, ensure_place, orchestrator), `src/places/places.py` (registry, Wikidata, resolver),
`src/facts/facts.py` (facts layer); `tests/` has six files. `src/process`, `src/retrieval`, `src/agents`, `src/tools`,
`eval`, `app`, `index` are filled in later phases. `data/raw` and `data/processed` are gitignored; the frozen
corpus rules are in [CLAUDE.md](CLAUDE.md). Results and the commands that reproduce them are in [LOG.md](LOG.md);
decisions in [docs/decisions.md](docs/decisions.md); contracts in [docs/contracts.md](docs/contracts.md);
the session handoff in [docs/HANDOFF.md](docs/HANDOFF.md).
