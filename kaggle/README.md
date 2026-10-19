# Kaggle runbook (written by the user; notebook details are filled in during Phase 3)

## A. Enrichment + embedding notebook (resumable)
1. Upload data/processed/chunks.jsonl as a private Kaggle dataset.
2. Notebook (GPU): start Ollama or use transformers; read chunks, process in batches, checkpoint every 200 chunks
   to /kaggle/working/checkpoints/, skip already-done chunk_ids on restart.
3. Output: enriched chunks.jsonl + embeddings (plain and contextual) as .npy + id list. Save as a dataset version.
4. Download, place in data/processed/ and index/, run integrity checks (counts, no empty notes, id alignment).

## B. Live Ollama via ngrok (for query-time LLM calls only)
1. Start Ollama in a notebook, pull the generator and judge models.
2. Expose with ngrok; put the https URL in .env as OLLAMA_BASE_URL. Tunnel URL changes each session.
3. All client calls go through src/common/llm.py (retries, cache). Expect the tunnel to drop; never rely on it for batch jobs.
