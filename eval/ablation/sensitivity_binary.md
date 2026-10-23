# Sensitivity analysis, run after the test split was seen; the frozen retrieve() choice is unchanged

Grade 2 is collapsed into 1 for ALL judgments (round 1 human and round 2 chat), then nDCG@10, MRR, P@5 and Recall@10 are recomputed from the saved top-10 rankings (no retrieval re-run). MRR, P@5 and Recall use grade >= 1 and are identical by construction; only nDCG@10 can move. MRR here is computed on the saved top 10, whereas the final table's MRR comes from the top-20 run, so a row whose first relevant chunk sits at rank 11 to 20 shows a slightly lower MRR here (bm25_plain on test: 0.692 against 0.699); the original/collapsed pair is identical either way. Recall@20 is not recomputed (only top-10 rankings are saved); it uses the same threshold, so it is unchanged too. Column `chat share` is the share of top-10 slots filled by a round-2 (chat-labelled) pair.

## test (27 queries)

| row | nDCG@10 original | nDCG@10 collapsed | delta | MRR | P@5 | Recall@10 | chat share |
|---|---|---|---|---|---|---|---|
| bm25_plain | 0.436 | 0.491 | +0.055 | 0.692 / 0.692 | 0.519 / 0.519 | 0.279 / 0.279 | 0.000 |
| dense_plain | 0.590 | 0.664 | +0.074 | 0.877 / 0.877 | 0.733 / 0.733 | 0.415 / 0.415 | 0.000 |
| bm25_contextual | 0.537 | 0.583 | +0.046 | 0.796 / 0.796 | 0.600 / 0.600 | 0.356 / 0.356 | 0.000 |
| dense_contextual (FROZEN) | 0.685 | 0.733 | +0.047 | 0.883 / 0.883 | 0.778 / 0.778 | 0.467 / 0.467 | 0.000 |
| hybrid_plain | 0.609 | 0.676 | +0.067 | 0.896 / 0.896 | 0.726 / 0.726 | 0.393 / 0.393 | 0.067 |
| hybrid_contextual | 0.711 | 0.763 | +0.052 | 0.898 / 0.898 | 0.822 / 0.822 | 0.479 / 0.479 | 0.048 |
| rrf4 | 0.639 | 0.705 | +0.066 | 0.880 / 0.880 | 0.748 / 0.748 | 0.425 / 0.425 | 0.000 |
| dense_control | 0.683 | 0.742 | +0.058 | 0.920 / 0.920 | 0.756 / 0.756 | 0.473 / 0.473 | 0.111 |
| bm25_control | 0.501 | 0.548 | +0.047 | 0.733 / 0.733 | 0.563 / 0.563 | 0.337 / 0.337 | 0.015 |
| hybrid_control | 0.695 | 0.756 | +0.061 | 0.888 / 0.888 | 0.770 / 0.770 | 0.474 / 0.474 | 0.056 |
| expanded_hybrid | 0.713 | 0.773 | +0.060 | 0.963 / 0.963 | 0.807 / 0.807 | 0.464 / 0.464 | 0.196 |
| row5_filters | 0.685 | 0.741 | +0.056 | 0.895 / 0.895 | 0.756 / 0.756 | 0.447 / 0.447 | 0.326 |
| row6_fresh_w10 | 0.681 | 0.739 | +0.057 | 0.895 / 0.895 | 0.756 / 0.756 | 0.444 / 0.444 | 0.330 |
| row6_fresh_w20 | 0.681 | 0.738 | +0.057 | 0.895 / 0.895 | 0.763 / 0.763 | 0.444 / 0.444 | 0.333 |
| row6_fresh_w30 | 0.681 | 0.738 | +0.057 | 0.895 / 0.895 | 0.756 / 0.756 | 0.444 / 0.444 | 0.337 |
| hybrid_filters | 0.658 | 0.702 | +0.045 | 0.869 / 0.869 | 0.726 / 0.726 | 0.425 / 0.425 | 0.270 |
| row7_rerank | 0.819 | 0.874 | +0.056 | 1.000 / 1.000 | 0.904 / 0.904 | 0.543 / 0.543 | 0.204 |
| hybrid_rerank | 0.791 | 0.846 | +0.055 | 1.000 / 1.000 | 0.874 / 0.874 | 0.522 / 0.522 | 0.193 |
| row8_mmr | 0.791 | 0.859 | +0.068 | 1.000 / 1.000 | 0.904 / 0.904 | 0.527 / 0.527 | 0.233 |
| row9_crag | 0.803 | 0.873 | +0.069 | 1.000 / 1.000 | 0.911 / 0.911 | 0.541 / 0.541 | 0.237 |

### test: paired nDCG@10 comparisons, original vs collapsed (100,000 Monte Carlo flips, Holm over 11)

| comparison | delta orig | p_holm orig | delta collapsed | p_holm collapsed |
|---|---|---|---|---|
| bm25_plain -> dense_plain | +0.154 | 0.0568 | +0.173 | 0.0341 |
| dense_plain -> dense_contextual | +0.096 | 0.0325 | +0.069 | 0.1145 |
| hybrid_plain -> hybrid_contextual | +0.102 | 0.0084 | +0.087 | 0.0378 |
| dense_plain -> hybrid_plain | +0.019 | 1.0000 | +0.012 | 1.0000 |
| dense_contextual -> hybrid_contextual | +0.026 | 1.0000 | +0.030 | 1.0000 |
| bm25_contextual -> dense_contextual | +0.148 | 0.0324 | +0.150 | 0.0341 |
| dense_contextual -> row9_crag | +0.118 | 0.0057 | +0.140 | 0.0058 |
| hybrid_contextual -> expanded_hybrid | +0.002 | 1.0000 | +0.010 | 1.0000 |
| expanded_hybrid -> row7_rerank | +0.106 | 0.0041 | +0.101 | 0.0277 |
| row7_rerank -> row8_mmr | -0.027 | 0.4194 | -0.015 | 1.0000 |
| row8_mmr -> row9_crag | +0.012 | 1.0000 | +0.014 | 1.0000 |

Significant (p_holm < 0.05), original: ['bm25_contextual -> dense_contextual', 'dense_contextual -> row9_crag', 'dense_plain -> dense_contextual', 'expanded_hybrid -> row7_rerank', 'hybrid_plain -> hybrid_contextual']
Significant (p_holm < 0.05), collapsed: ['bm25_contextual -> dense_contextual', 'bm25_plain -> dense_plain', 'dense_contextual -> row9_crag', 'expanded_hybrid -> row7_rerank', 'hybrid_plain -> hybrid_contextual']
Rank order by nDCG@10, original: row7_rerank, row9_crag, hybrid_rerank, row8_mmr, expanded_hybrid, hybrid_contextual, hybrid_control, dense_contextual, row5_filters, dense_control, row6_fresh_w10, row6_fresh_w30, row6_fresh_w20, hybrid_filters, rrf4, hybrid_plain, dense_plain, bm25_contextual, bm25_control, bm25_plain
Rank order by nDCG@10, collapsed: row7_rerank, row9_crag, row8_mmr, hybrid_rerank, expanded_hybrid, hybrid_contextual, hybrid_control, dense_control, row5_filters, row6_fresh_w10, row6_fresh_w20, row6_fresh_w30, dense_contextual, rrf4, hybrid_filters, hybrid_plain, dense_plain, bm25_contextual, bm25_control, bm25_plain

## dev (13 queries)

| row | nDCG@10 original | nDCG@10 collapsed | delta | MRR | P@5 | Recall@10 | chat share |
|---|---|---|---|---|---|---|---|
| bm25_plain | 0.524 | 0.561 | +0.037 | 0.724 / 0.724 | 0.600 / 0.600 | 0.288 / 0.288 | 0.000 |
| dense_plain | 0.673 | 0.744 | +0.071 | 0.949 / 0.949 | 0.769 / 0.769 | 0.369 / 0.369 | 0.000 |
| bm25_contextual | 0.551 | 0.588 | +0.037 | 0.769 / 0.769 | 0.646 / 0.646 | 0.288 / 0.288 | 0.000 |
| dense_contextual (FROZEN) | 0.793 | 0.863 | +0.070 | 1.000 / 1.000 | 0.862 / 0.862 | 0.431 / 0.431 | 0.000 |
| hybrid_plain | 0.645 | 0.701 | +0.056 | 0.904 / 0.904 | 0.677 / 0.677 | 0.357 / 0.357 | 0.062 |
| hybrid_contextual | 0.768 | 0.826 | +0.058 | 1.000 / 1.000 | 0.846 / 0.846 | 0.403 / 0.403 | 0.069 |
| rrf4 | 0.703 | 0.767 | +0.064 | 1.000 / 1.000 | 0.769 / 0.769 | 0.380 / 0.380 | 0.000 |
| dense_control | 0.727 | 0.781 | +0.054 | 1.000 / 1.000 | 0.815 / 0.815 | 0.393 / 0.393 | 0.115 |
| bm25_control | 0.538 | 0.572 | +0.034 | 0.814 / 0.814 | 0.631 / 0.631 | 0.274 / 0.274 | 0.023 |
| hybrid_control | 0.716 | 0.755 | +0.039 | 0.891 / 0.891 | 0.785 / 0.785 | 0.390 / 0.390 | 0.108 |
| expanded_hybrid | 0.717 | 0.789 | +0.072 | 0.923 / 0.923 | 0.831 / 0.831 | 0.400 / 0.400 | 0.323 |
| row5_filters | 0.654 | 0.728 | +0.074 | 0.885 / 0.885 | 0.754 / 0.754 | 0.371 / 0.371 | 0.369 |
| row6_fresh_w10 | 0.654 | 0.728 | +0.074 | 0.885 / 0.885 | 0.754 / 0.754 | 0.371 / 0.371 | 0.369 |
| row6_fresh_w20 | 0.650 | 0.727 | +0.077 | 0.885 / 0.885 | 0.754 / 0.754 | 0.371 / 0.371 | 0.369 |
| row6_fresh_w30 | 0.650 | 0.727 | +0.077 | 0.885 / 0.885 | 0.754 / 0.754 | 0.371 / 0.371 | 0.369 |
| hybrid_filters | 0.726 | 0.783 | +0.056 | 0.923 / 0.923 | 0.831 / 0.831 | 0.391 / 0.391 | 0.177 |
| row7_rerank | 0.771 | 0.811 | +0.040 | 0.962 / 0.962 | 0.846 / 0.846 | 0.417 / 0.417 | 0.346 |
| hybrid_rerank | 0.729 | 0.777 | +0.048 | 0.885 / 0.885 | 0.815 / 0.815 | 0.405 / 0.405 | 0.254 |
| row8_mmr | 0.772 | 0.813 | +0.040 | 0.962 / 0.962 | 0.862 / 0.862 | 0.410 / 0.410 | 0.362 |
| row9_crag | 0.778 | 0.826 | +0.048 | 0.923 / 0.923 | 0.862 / 0.862 | 0.419 / 0.419 | 0.346 |

### dev: paired nDCG@10 comparisons, original vs collapsed (exact, Holm over 11)

| comparison | delta orig | p_holm orig | delta collapsed | p_holm collapsed |
|---|---|---|---|---|
| bm25_plain -> dense_plain | +0.149 | 0.2637 | +0.183 | 0.2900 |
| dense_plain -> dense_contextual | +0.120 | 0.0161 | +0.119 | 0.0430 |
| hybrid_plain -> hybrid_contextual | +0.123 | 0.2832 | +0.125 | 0.3477 |
| dense_plain -> hybrid_plain | -0.028 | 1.0000 | -0.043 | 1.0000 |
| dense_contextual -> hybrid_contextual | -0.025 | 1.0000 | -0.037 | 1.0000 |
| bm25_contextual -> dense_contextual | +0.242 | 0.0537 | +0.275 | 0.0464 |
| dense_contextual -> row9_crag | -0.016 | 1.0000 | -0.037 | 1.0000 |
| hybrid_contextual -> expanded_hybrid | -0.051 | 0.5859 | -0.037 | 1.0000 |
| expanded_hybrid -> row7_rerank | +0.054 | 0.3931 | +0.022 | 1.0000 |
| row7_rerank -> row8_mmr | +0.002 | 1.0000 | +0.002 | 1.0000 |
| row8_mmr -> row9_crag | +0.005 | 1.0000 | +0.013 | 1.0000 |

Significant (p_holm < 0.05), original: ['dense_plain -> dense_contextual']
Significant (p_holm < 0.05), collapsed: ['bm25_contextual -> dense_contextual', 'dense_plain -> dense_contextual']
Rank order by nDCG@10, original: dense_contextual, row9_crag, row8_mmr, row7_rerank, hybrid_contextual, hybrid_rerank, dense_control, hybrid_filters, expanded_hybrid, hybrid_control, rrf4, dense_plain, row5_filters, row6_fresh_w10, row6_fresh_w20, row6_fresh_w30, hybrid_plain, bm25_contextual, bm25_control, bm25_plain
Rank order by nDCG@10, collapsed: dense_contextual, hybrid_contextual, row9_crag, row8_mmr, row7_rerank, expanded_hybrid, hybrid_filters, dense_control, hybrid_rerank, rrf4, hybrid_control, dense_plain, row5_filters, row6_fresh_w10, row6_fresh_w20, row6_fresh_w30, hybrid_plain, bm25_contextual, bm25_control, bm25_plain

## Does any conclusion change?

* test: significant pairs CHANGED; rank order of all 20 rows CHANGED.
  * only original: ['dense_plain -> dense_contextual']; only collapsed: ['bm25_plain -> dense_plain']
  * positions that differ (position, original, collapsed): [(3, 'hybrid_rerank', 'row8_mmr'), (4, 'row8_mmr', 'hybrid_rerank'), (8, 'dense_contextual', 'dense_control'), (10, 'dense_control', 'row6_fresh_w10'), (11, 'row6_fresh_w10', 'row6_fresh_w20'), (13, 'row6_fresh_w20', 'dense_contextual'), (14, 'hybrid_filters', 'rrf4'), (15, 'rrf4', 'hybrid_filters')]
* dev: significant pairs CHANGED; rank order of all 20 rows CHANGED.
  * only original: none; only collapsed: ['bm25_contextual -> dense_contextual']
  * positions that differ (position, original, collapsed): [(2, 'row9_crag', 'hybrid_contextual'), (3, 'row8_mmr', 'row9_crag'), (4, 'row7_rerank', 'row8_mmr'), (5, 'hybrid_contextual', 'row7_rerank'), (6, 'hybrid_rerank', 'expanded_hybrid'), (7, 'dense_control', 'hybrid_filters'), (8, 'hybrid_filters', 'dense_control'), (9, 'expanded_hybrid', 'hybrid_rerank'), (10, 'hybrid_control', 'rrf4'), (11, 'rrf4', 'hybrid_control')]

## Reading (computed from the tables above)

* test: frozen row to full chain, delta +0.118 (p_holm 0.0057) originally, +0.140 (p_holm 0.0058) collapsed. Mean nDCG@10 gain from collapsing: +0.055 over the 7 rows with under 5% chat-labelled slots, +0.058 over the 10 rows with 15% or more (descriptive; those rows also differ in other ways).
* dev: frozen row to full chain, delta -0.016 (p_holm 1.0000) originally, -0.037 (p_holm 1.0000) collapsed. Mean nDCG@10 gain from collapsing: +0.052 over the 6 rows with under 5% chat-labelled slots, +0.061 over the 10 rows with 15% or more (descriptive; those rows also differ in other ways).

## Round-2 label counts (final labels after the recorded corrections; counts only)

| split | round-2 pairs | label 0 / 1 / 2 | label 1 in top 10 of frozen row | label 1 in top 10 of full chain |
|---|---|---|---|---|
| dev | 162 | 78 / 80 / 4 | 0 | 31 |
| test | 273 | 140 / 128 / 5 | 0 | 42 |

## Owner vs chat spot-check, binary view (computed from the confusion table)

Confusion (rows owner, columns chat): [[8, 3, 0], [1, 6, 0], [1, 11, 0]]. Relevant-or-not (grade >= 1) agreement 0.833 (25 of 30), kappa 0.634; 2x2 [[both 0, owner 0 / chat >= 1], [owner >= 1 / chat 0, both >= 1]] = [[8, 3], [2, 17]].

