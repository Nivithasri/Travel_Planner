# Phase 6 ablation (dev; dev tuning only, D84/D92)

| row | n | 0rel | ndcg@10 | cond_ndcg@10 | mrr | p@5 | recall@10 | unjudged@10 | excluded@10 | src_types@10 | places@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25_plain | 13 | 0 | 0.542 | 0.542 | 0.724 | 0.600 | 0.378 | 0.000 | 0.177 | 2.462 | 5.385 |
| dense_plain | 13 | 0 | 0.701 | 0.701 | 0.949 | 0.769 | 0.476 | 0.000 | 0.054 | 2.308 | 4.462 |
| hybrid_plain | 13 | 0 | 0.659 | 0.684 | 0.904 | 0.677 | 0.455 | 0.062 | 0.108 | 2.538 | 4.692 |
| hybrid_contextual | 13 | 0 | 0.769 | 0.803 | 1.000 | 0.846 | 0.505 | 0.069 | 0.100 | 2.462 | 4.000 |
| expanded_hybrid | 13 | 0 | 0.581 | 0.752 | 0.879 | 0.692 | 0.396 | 0.323 | 0.138 | 2.462 | 4.077 |

## Paired nDCG@10 significance between consecutive rows (D92)

| comparison | n | delta | 95% CI | p | p (Holm) | method |
|---|---|---|---|---|---|---|
| bm25_plain -> dense_plain | 13 | +0.159 | [0.044, 0.282] | 0.0278 | 0.0835 | exact |
| dense_plain -> hybrid_plain | 13 | -0.042 | [-0.164, 0.065] | 0.5215 | 0.5215 | exact |
| hybrid_plain -> hybrid_contextual | 13 | +0.110 | [0.015, 0.220] | 0.0576 | 0.1152 | exact |
| hybrid_contextual -> expanded_hybrid | 13 | -0.188 | [-0.314, -0.075] | 0.0103 | 0.0410 | exact |

unjudged@10 > 0 means the row retrieved chunks round 1 never pooled; those count as 0 in ndcg@10 until round 2 is labelled (D91). cond_ndcg@10 removes unjudged chunks from each ranking before the cut, so a row is not charged for labels nobody has made yet; read the two columns together. nDCG on 13 dev queries is descriptive, not significant (D92).
