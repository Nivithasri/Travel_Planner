# Cross-encoder reranking (Nogueira & Cho, "Passage Re-ranking with BERT", 2019)

WRITTEN FROM MEMORY, not from the paper. Verify before relying on it.

- Method: a first-stage retriever returns top-N candidates; a cross-encoder then scores each (query, passage) PAIR jointly through one transformer (query and passage concatenated, [CLS] -> relevance score) and the candidates are re-sorted by that score. Unlike a bi-encoder (two separate embeddings, cosine), the cross-encoder attends across query and passage together, so it is more accurate but cannot be precomputed and is run only on the shortlist.
- Purpose: a precision rerank of a cheap recall-oriented first stage.
- Phase that uses it: 6 (rerank row 7). First stage = the row-3 hybrid; we rerank the top 30 (plan.md item 7). Model: BAAI/bge-reranker-v2-m3 (owner's choice, D88), run locally on CPU, offline, scores cached per (query, chunk).
- Parameters that must be right: rerank depth = 30; the score is the model's logit (we map it through sigmoid only for MMR relevance, D89); ranks 31-100 keep their order below the reranked 30; the cache key includes the weight-file hashes so other weights never reuse scores.
- Caveat: a cross-encoder gives a per-pair score, not a calibrated probability; thresholds on it (CRAG tau, D90) are tuned on dev, not assumed.
