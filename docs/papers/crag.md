# Corrective RAG (Yan et al., 2024)

WRITTEN FROM MEMORY, not from the paper. Verify before relying on it.

- Method: a lightweight retrieval evaluator grades the retrieved documents for a query as Correct / Incorrect / Ambiguous. On Correct, keep and refine; on Incorrect, discard and fall back to an external (web) search; on Ambiguous, combine both. A knowledge-refinement ("decompose-then-recompose") step strips irrelevant strips from kept documents.
- Purpose: stop the generator from grounding on weak or wrong retrieval by detecting it and correcting before generation.
- Phase that uses it: 6 (corrective row 9), adapted. OUR DEVIATIONS, stated because they matter for the report:
  - No learned evaluator and NO LLM grading of chunks: owner rule says only query text goes to the LLM in Phase 6 (D90), so we cannot send chunk text to a judge. We use deterministic triggers instead: top cross-encoder score below tau, fewer than 3 distinct source URLs in the top 5, or an open/closed contradiction among same-place top-5 chunks.
  - The "external search" fallback is NOT web in Phase 6 (retrieve() is frozen-corpus only, D31); the corrective pass relaxes filters, widens depth 100->300 and adds one cached LLM query rewrite. The live-layer version is Phase 7 (O3).
  - At most one corrective pass (plan.md item 9).
- Parameters that must be right: tau (tuned on dev, D90); exactly one pass; no corpus text in any prompt.
