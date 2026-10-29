# Key results 

## 1. Our frozen retriever (`dense_contextual`) on the test set

| Metric | Value |
|---|---|
| nDCG@10 | **0.685** |
| MRR | **0.883** |
| P@5 | **0.778** |
| Recall@10 | **0.767** |

On average the first relevant chunk sits at about rank 1 to 2 (MRR 0.883), and about 4 of the top 5 results are relevant (P@5 0.778).

---

## 2. Adding the context note helps, and the gain is significant

| Comparison (test, 27 queries) | nDCG@10 before | after | Gain | 95% CI | p (Holm) |
|---|---|---|---|---|---|
| dense plain -> dense contextual | 0.590 | 0.685 | **+0.096** | [0.035, 0.156] | **0.033** |
| hybrid plain -> hybrid contextual | 0.609 | 0.711 | **+0.102** | [0.051, 0.156] | **0.008** |
| BM25 plain -> BM25 contextual (descriptive, no test) | 0.436 | 0.537 | +0.101 | n/a | n/a |

The context note improved all three retrievers by about 0.10 nDCG.

---

## 3. Dense retrieval beats keyword search

| Comparison (test) | nDCG@10 | Gain | 95% CI | p (Holm) |
|---|---|---|---|---|
| BM25 contextual (0.537) -> dense contextual (0.685) | 0.537 -> 0.685 | **+0.148** | [0.059, 0.239] | **0.032** |

Dense also leads on MRR (0.883 vs 0.796) and P@5 (0.778 vs 0.600).

---

## 4. Cross-encoder reranking gives a large gain

| Comparison (test) | nDCG@10 | Gain | 95% CI | p (Holm) |
|---|---|---|---|---|
| expanded hybrid (0.713) -> + rerank of top 30 (0.819) | 0.713 -> 0.819 | **+0.106** | [0.058, 0.155] | **0.004** |

The reranked row reaches **MRR 1.000** (the first result was relevant for every query) and **P@5 0.904**.

---

## 5. Best observed pipeline (full chain), labelled as post-hoc

| Comparison (test) | nDCG@10 | Gain | 95% CI | p (Holm) |
|---|---|---|---|---|
| frozen dense_contextual (0.685) -> full chain (0.803) | 0.685 -> 0.803 | **+0.118** | [0.058, 0.184] | **0.006** |

The full chain is expansion, rerank of 30, MMR and one corrective pass. It reaches MRR 1.000, P@5 0.911 and Recall@10 0.841.


