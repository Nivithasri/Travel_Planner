# OBSOLETE (D59, 2026-10-20): the Kaggle account is banned. Do not follow this runbook.

Enrichment now uses a hosted OpenAI-compatible API (see docs/decisions.md D59, D60 and scripts/enrich.py) and
embeddings are computed locally on CPU with sentence-transformers. The text below is kept only as history.

# Kaggle runbook (written by the user; notebook details are filled in during Phase 3)

Phase 3 amendments (D49, D51, D56): enrichment runs LOCALLY through the tunnel (not in a notebook); the notebook only embeds. Dataset and notebook are PRIVATE (D49). [SUPERSEDED by D59]

## A. Enrichment + embedding notebook (resumable)
1. Upload data/enriched/chunks.jsonl (after enrichment) as a PRIVATE Kaggle dataset. Never data/processed (frozen, D48 xi).
2. Notebook (GPU): start Ollama or use transformers; read chunks, process in batches, checkpoint every 200 chunks
   to /kaggle/working/checkpoints/, skip already-done chunk_ids on restart.
3. Output: enriched chunks.jsonl + embeddings (plain and contextual) as .npy + id list. Save as a dataset version.
4. Download the .npy and id files into index/ (never data/processed/); `build_index.py --verify` checks counts, id order and the chunks.jsonl sha256 the notebook recorded.

## B. Live Ollama via ngrok (for query-time LLM calls only)
1. Start Ollama in a notebook, pull the generator and judge models.
2. Expose with ngrok; put the https URL in .env as OLLAMA_BASE_URL. Tunnel URL changes each session.
3. All client calls go through src/common/llm.py (retries, cache). Expect the tunnel to drop; never rely on it for batch jobs.
