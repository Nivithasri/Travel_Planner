# Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)

WRITTEN FROM MEMORY, not from the paper (no copy is in the repo). Verify the constant and the formula before relying on them.

- Method: combine several ranked lists by scoring each document `RRF(d) = sum_over_lists 1 / (k + rank_in_that_list(d))`, rank counted from 1, then sort by that score descending. A document absent from a list contributes nothing from it.
- The constant `k` damps the influence of top ranks so a single list cannot dominate; the paper uses `k = 60` and reports it is not sensitive around that value.
- Why it works here: BM25 (exact terms, place names, "closed") and dense bge-m3 (meaning) fail on different queries; RRF needs only the ranks, not comparable scores, so it fuses them without score normalisation.
- Phase that uses it: 6 (hybrid row 2, sub-query fusion row 4; already in eval/adapters.rrf and eval/pool with k=60, depth 100).
- Parameter that must be right: `k = 60`; ranks from 1; ties broken by chunk_id for reproducibility (our addition, not the paper's).
