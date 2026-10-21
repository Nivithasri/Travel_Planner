# Phase 6 ablation (dev; dev tuning only, D84/D92)

| row | n | 0rel | ndcg@10 | cond_ndcg@10 | mrr | p@5 | recall@10 | unjudged@10 | excluded@10 | src_types@10 | places@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_contextual | 13 | 0 | 0.769 | 0.803 | 1.000 | 0.846 | 0.505 | 0.069 | 0.100 | 2.462 | 4.000 |
| hybrid_rerank | 13 | 0 | 0.612 | 0.793 | 0.857 | 0.600 | 0.420 | 0.254 | 0.123 | 2.308 | 4.000 |
| expanded_hybrid | 13 | 0 | 0.581 | 0.752 | 0.879 | 0.692 | 0.396 | 0.323 | 0.138 | 2.462 | 4.077 |
| row5_filters | 13 | 0 | 0.536 | 0.672 | 0.827 | 0.600 | 0.379 | 0.369 | 0.169 | 2.308 | 3.692 |
| row6_fresh_w10 | 13 | 0 | 0.536 | 0.672 | 0.827 | 0.600 | 0.379 | 0.369 | 0.169 | 2.308 | 3.692 |
| row6_fresh_w20 | 13 | 0 | 0.532 | 0.671 | 0.827 | 0.600 | 0.379 | 0.369 | 0.169 | 2.385 | 3.615 |
| row6_fresh_w30 | 13 | 0 | 0.532 | 0.670 | 0.827 | 0.600 | 0.379 | 0.369 | 0.169 | 2.385 | 3.615 |

## Paired nDCG@10 significance between consecutive rows (D92)

| comparison | n | delta | 95% CI | p | p (Holm) | method |
|---|---|---|---|---|---|---|
| hybrid_contextual -> hybrid_rerank | 13 | -0.156 | [-0.279, -0.050] | 0.0183 | 0.1099 | exact |
| hybrid_rerank -> expanded_hybrid | 13 | -0.031 | [-0.188, 0.118] | 0.7039 | 1.0000 | exact |
| expanded_hybrid -> row5_filters | 13 | -0.045 | [-0.102, 0.005] | 0.1494 | 0.7471 | exact |
| row5_filters -> row6_fresh_w10 | 13 | +0.000 | [0.000, 0.000] | 1.0000 | 1.0000 | exact |
| row6_fresh_w10 -> row6_fresh_w20 | 13 | -0.004 | [-0.011, 0.000] | 1.0000 | 1.0000 | exact |
| row6_fresh_w20 -> row6_fresh_w30 | 13 | +0.000 | [0.000, 0.000] | 1.0000 | 1.0000 | exact |

unjudged@10 > 0 means the row retrieved chunks round 1 never pooled; those count as 0 in ndcg@10 until round 2 is labelled (D91). cond_ndcg@10 removes unjudged chunks from each ranking before the cut, so a row is not charged for labels nobody has made yet; read the two columns together. nDCG on 13 dev queries is descriptive, not significant (D92).
