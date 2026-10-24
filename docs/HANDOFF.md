# HANDOFF (rewritten 2026-10-24 at the end of the Phase 9 run; the next session reads this first)

Read this file, LOG.md, docs/phase-status.md, then `git status` and `git log --oneline -10`. Trust these and git, not memory.
The old per-phase and enrichment-routine notes are in the git history of this file (the commit before this rewrite).

## Where the project stands
| Phase | State |
|---|---|
| 1 to 5 (data, corpus, enrichment and indexes, test collection, routing data) | CLOSED. Phase 5 parameters are PROVISIONAL until about 8 calibration routes are spot-checked (D77). |
| 6 (retrieval engine and ablations) | CLOSED. retrieve() is FROZEN as dense_contextual (D96 amended). |
| 7 (place cards, agent pipeline, live layer) | CLOSED. |
| 8 (Streamlit app) | BUILT and hand-run by the owner twice on 2026-10-24 (live on): works end to end; findings D113 to D122 and D129 fixed. `/phase-review 8` was NOT run; no tag. |
| 9 (evaluation) | RUN and saved (D123 to D130, results in eval/phase9/results/). `/phase-review 9` NOT run; no tag. |
Local tags: phase-1, 4, 5, 6 and 7-done. phase-2-done and phase-3-done do NOT exist (older notes said they did). phase-8-done and phase-9-done are the owner's call.

## Phase 9 in one paragraph (details: D130 and D130 amended, docs/phase9_cases.md, eval/phase9/results/summary.md)
Three systems (LLM-only, plain RAG, the full pipeline) on 10 approved requests, scored 1 to 5 by the owner on four dimensions, with an independent second scorer on 10 rows
(pooled kappa 0.60: weak on feasibility 0.21 and personalization 0.33, good on groundedness 0.72 and freshness 0.80). Means: llm_only 2.7 / 1.0 / 2.2 / 3.8, plain_rag 2.6 / 3.5 / 2.2 / 2.8,
full 2.3 / 3.8 / 4.5 / 2.6 (feasibility / groundedness / freshness / personalization). The full system wins on groundedness and freshness (partly by construction) and does NOT beat the
baselines on feasibility or personalization: 12 of 43 days are empty (21 of 43 with live search). Live search made the answers significantly WORSE on groundedness (-0.9) and
freshness (-1.9). Citation precision 43 of 50. Latency: 56.5 s median, 52.0 s of it rate-limiter wait. The prompt and live-layer fixes are listed in docs/HANDOVER.md.

## Tests and checks (the state at the last commit; rerun to confirm)
`.venv/Scripts/python.exe -m ruff check .` clean; `.venv/Scripts/python.exe -m pytest -q`: 1,490 passed, 12 skipped (the skips need data/raw or the reranker folder).
Mutation checks (`scripts/mutation_checks.py`): the last FULL run was 212 rows, 211 caught, only M32 alive (its tests skip without data/raw); it predates D129, the Phase 9 harness
and the later fixes. The FULL suite was NOT rerun after them (owner decision, time): the groups for the new code ran and were all caught (`--critic` M208 to M212, `--phase9` M213 to M251
including the D129 group, `--evidence` M241 to M248). A killed rerun left one mutation in src/common/llm.py; it was restored with git checkout and every tracked code file was verified identical to HEAD.
Run the full suite again on a quiet machine (about 85 minutes, nobody using the app or the pipeline) before relying on it.

## How to run (use the .venv interpreter; PowerShell needs the leading .\)
- App: `.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py` (live search is a tick box, off by default; a live-on request spends searches and tokens).
- Pipeline CLI: `scripts/run_pipeline.py --query "..." --days 4 --month 12 --start Chennai --no-live`. Evaluation: `scripts/phase9.py` (subcommands in its docstring and docs/HANDOVER.md).
- Retrieval evaluation: `python eval/run_eval.py --retriever <module:function> --split dev|test` (the test split was run once, D96).

## Environment (this machine, 2026-10-24)
.venv on Python 3.12.7 with the D94 pins; the lint hook calls bare `python` (ruff is also installed in the base interpreter) and the guard hooks call `python3` (a real python3.exe sits next to Python 3.12).
Missing and expected: data/raw, models/bge-reranker-v2-m3, data/processed/FROZEN (the freeze ran on the partner's machine; frozen_manifest.json is committed). Commits are authored as the owner's GitHub
identity through GIT_AUTHOR_* and GIT_COMMITTER_* environment variables (no git config change; the address itself is deliberately not written in the repo, rule 11).

## The frozen corpus (verify with `sha256sum data/processed/*`)
| file | sha256 |
|---|---|
| chunks.jsonl (1,696 chunks) | ee11755106409fd0d6ae8656453fc1ec70560593a9c25348572f01ec804a0003 |
| drops.jsonl | 51fc0cd78423533c43a5f482d4d04ffc163e83dad80f2eb35304c2372046b2f9 |
| attribution.json (96 urls) | 0b5b73bd28b3f278aaaa8d532574dc765e194f85cd4ddc3821745d62551ee7bf |
| d33_excluded.json (committed copy: data/manifests/d33_excluded.json) | a712d7b6cef25c9c5b0a0d932e45ef12e4088353640cdf60f20a251ee61b3041 |
data/enriched/chunks.jsonl sha256 e4885c84031e871a2765b706dbdd9f4c0780c1b6f8ee1fb98479e9be18185135. 17 eligible places, 9 D33-excluded places (query-eligibility list; retrieval is unfiltered), 0 hidden_gem.
Accepted deviations: 1,696 chunks (plan 10k to 20k), 3 source types, one blog host, no chunk dated inside the 18-month window (freshness only from the live layer).

## Known limits (do not forget)
Routing is PROVISIONAL (D77); budget figures are ASSUMED (D101); place-card items are about 58% useful (D106); the evaluated corpus has one Kerala place; the live layer runs under the owner's
recorded risk acceptance with terms NOT VERIFIED (D109) and, as built, lowers quality (D130 amended); round-2 relevance labels were produced by a chat assistant (D95, D98); only 10 requests and 13/27 dev/test
queries, so only large differences are detectable; the hosted model is not fully deterministic.

## Open owner items
1. `/phase-review 8` and `/phase-review 9`, then the tags (phase-8-done, phase-9-done). 2. The full mutation suite on a quiet machine. 3. Optional Phase 9 extras: the lean judge has 7 plain_rag answers left
(`phase9.py judge`, needs token room), the with-live latency run (`latency --yes --with-live`, 20 searches). 4. The prompt and live-layer fixes (docs/HANDOVER.md). 5. O12 (what may be shared or pushed
beyond code and committed numbers), the provider and Tavily terms, spot-checking the calibration routes. 6. Never push anything under .cache, eval/sheets, data/processed, data/enriched, index or models.

## Hooks and working rules
Keep `git config core.hooksPath .githooks`. No AI attribution anywhere (CLAUDE.md). A STOP rule means stop BEFORE acting. Never push or tag unless the owner asks.
