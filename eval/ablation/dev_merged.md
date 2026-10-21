# Dev rows re-scored with the merged judgments (round 1 human + round 2 chat)

Round 2 is labelled by a chat assistant (D95); 4 of its labels were corrected on the owner's instruction. Dev, 13 queries: descriptive, not significant (D92).

## All rows

| row | nDCG@10 round 1 | nDCG@10 merged | cond. merged | MRR | P@5 | R@10 | unjudged@10 r1 | unjudged@10 merged | places@10 |
|---|---|---|---|---|---|---|---|---|---|
| bm25_plain | 0.542 | 0.524 | 0.524 | 0.724 | 0.600 | 0.288 | 0.000 | 0.000 | 5.385 |
| dense_plain | 0.701 | 0.673 | 0.673 | 0.949 | 0.769 | 0.369 | 0.000 | 0.000 | 4.462 |
| hybrid_plain | 0.659 | 0.645 | 0.645 | 0.904 | 0.677 | 0.357 | 0.062 | 0.000 | 4.692 |
| hybrid_contextual | 0.769 | 0.768 | 0.768 | 1.000 | 0.846 | 0.403 | 0.069 | 0.000 | 4.000 |
| expanded_hybrid | 0.581 | 0.717 | 0.717 | 0.923 | 0.831 | 0.400 | 0.323 | 0.000 | 4.077 |
| row7_rerank | 0.589 | 0.771 | 0.771 | 0.962 | 0.846 | 0.417 | 0.346 | 0.000 | 4.154 |
| row8_mmr | 0.579 | 0.772 | 0.772 | 0.962 | 0.862 | 0.410 | 0.362 | 0.000 | 5.154 |
| row9_crag | 0.578 | 0.778 | 0.778 | 0.923 | 0.862 | 0.419 | 0.346 | 0.000 | 5.000 |
| hybrid_filters | 0.717 | 0.726 | 0.726 | 0.923 | 0.831 | 0.391 | 0.177 | 0.000 | 3.615 |
| hybrid_rerank | 0.612 | 0.729 | 0.729 | 0.885 | 0.815 | 0.405 | 0.254 | 0.000 | 4.000 |
| row5_filters | 0.536 | 0.654 | 0.654 | 0.885 | 0.754 | 0.371 | 0.369 | 0.000 | 3.692 |
| row6_fresh_w10 | 0.536 | 0.654 | 0.654 | 0.885 | 0.754 | 0.371 | 0.369 | 0.000 | 3.692 |
| row6_fresh_w20 | 0.532 | 0.650 | 0.650 | 0.885 | 0.754 | 0.371 | 0.369 | 0.000 | 3.615 |
| row6_fresh_w30 | 0.532 | 0.650 | 0.650 | 0.885 | 0.754 | 0.371 | 0.369 | 0.000 | 3.615 |

## Keep rules re-applied (D84: delta >= -0.005 over the predecessor)

| component | from -> to | delta round 1 | delta merged | verdict (merged) |
|---|---|---|---|---|
| row 3: contextual notes | hybrid_plain -> hybrid_contextual | +0.110 | +0.123 | KEEP |
| row 4: query expansion | hybrid_contextual -> expanded_hybrid | -0.188 | -0.051 | DROP |
| row 5: metadata filters | expanded_hybrid -> row5_filters | -0.045 | -0.063 | DROP |
| row 5 on the row-3 base (diagnostic) | hybrid_contextual -> hybrid_filters | -0.052 | -0.042 | DROP |
| row 7: cross-encoder rerank | expanded_hybrid -> row7_rerank | +0.008 | +0.054 | KEEP |
| row 7 on the row-3 base (diagnostic) | hybrid_contextual -> hybrid_rerank | -0.156 | -0.039 | DROP |
| row 8: MMR at the frozen lambda | row7_rerank -> row8_mmr | -0.009 | +0.002 | by the lambda rule: lambda 0.9 (MMR on) |
| row 9: corrective pass at the frozen tau | row8_mmr -> row9_crag | -0.001 | +0.005 | KEEP |

## Parameters: what the pre-stated rules pick on the merged judgments

* freshness weight w (D87, highest dev nDCG@10, ties to the smaller): {0.0: 0.654, 0.1: 0.654, 0.2: 0.6505, 0.3: 0.6505} -> picks 0.0 (frozen: 0, freshness off)
* MMR lambda (D89): {1.0: 0.7705, 0.9: 0.7722, 0.8: 0.7461, 0.7: 0.7318, 0.5: 0.7089} -> picks 0.9 (frozen: 0.9)
* corrective tau (D90): {0.0611: 0.7899, 0.1104: 0.7794, 0.6314: 0.7775} -> picks 0.0611 (frozen: 0.6314)

The frozen files eval/mmr_lambda.json and eval/corrective_tau.json were NOT changed. A different pick here is information for the owner, not an automatic change.

## Paired nDCG@10 significance between consecutive chain rows (merged)

| comparison | n | delta | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| hybrid_plain -> hybrid_contextual | 13 | +0.123 | [0.029, 0.235] | 0.0354 | 0.1770 |
| hybrid_contextual -> expanded_hybrid | 13 | -0.051 | [-0.105, 0.002] | 0.0977 | 0.2930 |
| expanded_hybrid -> row7_rerank | 13 | +0.054 | [0.003, 0.097] | 0.0562 | 0.2246 |
| row7_rerank -> row8_mmr | 13 | +0.002 | [-0.041, 0.039] | 0.9492 | 1.0000 |
| row8_mmr -> row9_crag | 13 | +0.005 | [-0.029, 0.037] | 0.7969 | 1.0000 |
