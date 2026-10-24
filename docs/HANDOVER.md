# Handover: Phase 9 results and what to improve next

## State (2026-10-24)
Phases 1 to 8 are built, tested and documented; Phase 9 (evaluation) is RUN and saved but not closed. Read docs/HANDOFF.md, LOG.md, docs/phase-status.md
and decisions D123 to D130 first. Numbers: eval/phase9/results/summary.md and *.json (scores and counts only). Nothing in the repo holds answer text,
source excerpts, keys or personal addresses; keep it that way (CLAUDE.md rules 1 to 11).

## The result in one paragraph
The full system is the most grounded and the most honest about current status, but it is NOT better on feasibility or personalization: 12 of 43 days
(28%) come out empty, 21 of 43 (49%) when live search is on. Only three comparisons are significant after Holm, all partly by construction (D130), and live search made the answers significantly worse on groundedness and freshness (D130 amended).
The prompt and the evidence handling are what to improve; retrieval is frozen (D96) and should not change.

## What to improve (each is small, test it with fakes and add a mutation)
1. **Empty days.** Assign evidence items to specific days in the plan block (for example "Day 2: Mahabalipuram [use S4, S5, S6]") and add a writer
   rule: each day uses only its listed tags. Files: src/agents/writer.py (plan_block, SYSTEM, which has a 1,900-character limit), src/agents/evidence.py.
2. **Non-breaking hyphens.** Normalise U+2011 and similar characters in the model output before the critic (src/agents/critic.py, src/agents/writer.py),
   so the no-details line is recognised and the status is not spuriously "repaired".
3. **Live place resolution.** Resolve a live item's place from its URL and title first, then by the most mentions (src/live/search.py; D119 limit 3).
4. **Warning flags.** The model-written is_warning flag is only about 61% right (D106); one warning per stop is now allowed first (D129). Consider
   ranking by aspect and dropping history-only text, as D106 suggests.
5. **Rubric.** Say how an empty day is scored (eval/phase9/rubric.md); the two scorers disagreed most on feasibility and personalization.
6. **Live layer (it scored worse than corpus-only, D130 amended).** (a) Make R7 require live support for each current-status claim: keep the
   cannot-confirm line unless a live item backs the claim, even when live status is ok (src/agents/critic.py). (b) Filter live sources: no social media,
   no booking aggregators, require the page to name the place (src/live). (c) Forbid "as of <fetch date>" in the writer rule: the fetch date is not the
   page's date (src/agents/writer.py SYSTEM). Then rerun the live day; the harness and the baseline are in place.
After any change: `ruff check .`, `pytest -q`, the mutation groups, then rerun `scripts/phase9.py` against the saved baseline (the records carry the git sha).
A prompt change invalidates the cached writer replies once (a token cost) and every score that was given to outputs of the old prompt.

## How to rerun Phase 9
See docs/HANDOFF.md "PHASE 9 STEPS". In short, with `.venv/Scripts/python.exe scripts/phase9.py`: `check-requests`; `run --arm llm_only|plain_rag|full_corpus`
(10 requests, about 50,000 tokens in all); `sheet --set main`; score by eval/phase9/rubric.md; `score --set main --filled <csv>`; `crosscheck`; `citations`;
`judge`; `run --arm full_live` (one UTC day, 20 searches); `latency --yes`; `report`. Records and sheets are gitignored; only results are committed.

## Setup on a new machine
- Python 3.12 and a venv; pin the D94 versions (sentence-transformers 6.1.0, torch 2.14.1 CPU, numpy 2.5.4, transformers 5.20.2, bm25s 0.3.14, PyStemmer 3.1.0,
  ruff 0.17.1) plus streamlit, langgraph, scipy, then `pip install -e .[dev]`; do not install torchvision. `git config core.hooksPath .githooks`.
- Gitignored data you need (you built most of it): data/processed, data/enriched, index/ (the dense, BM25 and place-card files), models/bge-m3-full
  (the local embedding model), optionally models/bge-reranker-v2-m3 (only for the retrieval ablations), data/raw (only the Phase 5 tests and mutation M32).
- `.env`: copy .env.example and fill in YOUR OWN values. Variable names: LLM_API_KEY, LLM_API_URL (or LLM_API_BASE_URL), GENERATOR_MODEL, JUDGE_MODEL,
  LLM_RPM, LLM_TPM, LLM_RPD, LLM_TPD, LIVE_SEARCH_API_KEY, EMBED_LOCAL_DIR, RERANK_LOCAL_DIR, TOKENIZER_DIR, EVAL_REFERENCE_DATE, EMBED_MODEL, CONTACT_EMAIL.
  Never commit .env or paste keys anywhere. The free LLM quota is per organisation: two people on one key share one daily budget.
- The live search (Tavily) was used under the first author's own recorded risk acceptance (D109); if you use it, decide that yourself and record it.

## Do not share or commit
`.env`, keys, `.cache/`, `eval/sheets/` (they hold source excerpts and live-derived text, O12, D100), any file with an email address, and anything that
says an AI tool wrote code or commits (CLAUDE.md rule 1).
