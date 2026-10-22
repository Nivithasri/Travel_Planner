# Phase 6 TEST run (27 queries, one run; round 1 human + round 2 chat judgments)

commit 0dd35c9d9bb037d1f70490c5ad14123a0cfadada; frozen pipeline: dense_contextual. Descriptive: 27 queries detect differences of about 0.1 nDCG only.

| row | nDCG@10 test | dev (merged) | named | free | MRR | P@5 | R@10 | unjudged@10 | excluded@10 | places@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_plain | 0.436 | 0.524 | 0.423 | 0.464 | 0.699 | 0.519 | 0.279 | 0.000 | 0.115 | 4.852 |
| dense_plain | 0.590 | 0.673 | 0.623 | 0.522 | 0.877 | 0.733 | 0.415 | 0.000 | 0.126 | 3.852 |
| bm25_contextual | 0.537 | 0.551 | 0.531 | 0.549 | 0.796 | 0.600 | 0.356 | 0.000 | 0.070 | 4.185 |
| dense_contextual (FROZEN) | 0.685 | 0.793 | 0.759 | 0.538 | 0.883 | 0.778 | 0.467 | 0.000 | 0.104 | 2.852 |
| hybrid_plain | 0.609 | 0.645 | 0.595 | 0.637 | 0.896 | 0.726 | 0.393 | 0.000 | 0.122 | 4.111 |
| hybrid_contextual | 0.711 | 0.768 | 0.747 | 0.640 | 0.898 | 0.822 | 0.479 | 0.000 | 0.078 | 3.259 |
| rrf4 | 0.639 | 0.703 | 0.646 | 0.624 | 0.880 | 0.748 | 0.425 | 0.000 | 0.096 | 3.778 |
| dense_control | 0.683 | 0.727 | 0.760 | 0.529 | 0.920 | 0.756 | 0.473 | 0.000 | 0.074 | 3.000 |
| bm25_control | 0.501 | 0.538 | 0.520 | 0.464 | 0.733 | 0.563 | 0.337 | 0.000 | 0.093 | 4.481 |
| hybrid_control | 0.695 | 0.716 | 0.744 | 0.597 | 0.888 | 0.770 | 0.474 | 0.000 | 0.093 | 3.296 |
| expanded_hybrid | 0.713 | 0.717 | 0.709 | 0.720 | 0.963 | 0.807 | 0.464 | 0.000 | 0.137 | 2.815 |
| row5_filters | 0.685 | 0.654 | 0.676 | 0.703 | 0.895 | 0.756 | 0.447 | 0.000 | 0.178 | 2.519 |
| row6_fresh_w10 | 0.681 | 0.654 | 0.670 | 0.703 | 0.895 | 0.756 | 0.444 | 0.000 | 0.178 | 2.519 |
| row6_fresh_w20 | 0.681 | 0.650 | 0.670 | 0.703 | 0.895 | 0.763 | 0.444 | 0.000 | 0.178 | 2.481 |
| row6_fresh_w30 | 0.681 | 0.650 | 0.670 | 0.703 | 0.895 | 0.756 | 0.444 | 0.000 | 0.178 | 2.481 |
| hybrid_filters | 0.658 | 0.726 | 0.679 | 0.615 | 0.869 | 0.726 | 0.425 | 0.000 | 0.111 | 2.852 |
| row7_rerank | 0.819 | 0.771 | 0.831 | 0.793 | 1.000 | 0.904 | 0.543 | 0.000 | 0.144 | 2.741 |
| hybrid_rerank | 0.791 | 0.729 | 0.809 | 0.757 | 1.000 | 0.874 | 0.522 | 0.000 | 0.111 | 2.889 |
| row8_mmr | 0.791 | 0.772 | 0.821 | 0.732 | 1.000 | 0.904 | 0.527 | 0.000 | 0.100 | 3.593 |
| row9_crag | 0.803 | 0.778 | 0.826 | 0.757 | 1.000 | 0.911 | 0.541 | 0.000 | 0.111 | 3.519 |

## Pre-registered paired comparisons (nDCG@10; sign-flip, 100,000 flips; Holm over these 11)

| comparison | n | delta | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|
| bm25_plain -> dense_plain | 27 | +0.154 | [0.049, 0.264] | 0.0095 | 0.0568 |
| dense_plain -> dense_contextual | 27 | +0.096 | [0.035, 0.156] | 0.0046 | 0.0325 |
| hybrid_plain -> hybrid_contextual | 27 | +0.102 | [0.051, 0.156] | 0.0009 | 0.0084 |
| dense_plain -> hybrid_plain | 27 | +0.019 | [-0.068, 0.103] | 0.6747 | 1.0000 |
| dense_contextual -> hybrid_contextual | 27 | +0.026 | [-0.039, 0.089] | 0.4370 | 1.0000 |
| bm25_contextual -> dense_contextual | 27 | +0.148 | [0.059, 0.239] | 0.0040 | 0.0324 |
| dense_contextual -> row9_crag | 27 | +0.118 | [0.058, 0.184] | 0.0006 | 0.0057 |
| hybrid_contextual -> expanded_hybrid | 27 | +0.002 | [-0.046, 0.050] | 0.9491 | 1.0000 |
| expanded_hybrid -> row7_rerank | 27 | +0.106 | [0.058, 0.155] | 0.0004 | 0.0041 |
| row7_rerank -> row8_mmr | 27 | -0.027 | [-0.059, 0.000] | 0.0839 | 0.4194 |
| row8_mmr -> row9_crag | 27 | +0.012 | [-0.009, 0.038] | 0.3855 | 1.0000 |

## time_sensitive flag, predicted vs eval/queries.json, TEST (27 queries)

TP 7 FP 0 FN 1 TN 19, accuracy 0.963; wrong: ['Q25']

frozen engine vs its ablation row (top 10 ids, in order): 27 of 27 queries identical
