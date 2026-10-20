# Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998)

WRITTEN FROM MEMORY, not from the paper. Verify the formula before relying on it.

- Method: build the result list greedily. At each step pick the document that maximises
  `MMR = lambda * rel(d, query) - (1 - lambda) * max_over_already_selected_s sim(d, s)`.
  `lambda = 1` is pure relevance; `lambda = 0` is pure diversity (anti-redundancy).
- Purpose: reduce redundancy among top results, so a result set covers more of the information need instead of repeating the single best angle.
- Phase that uses it: 6 (diversity row 8). Our rel = sigmoid of the cross-encoder logit; our sim = 0.6*cos(dense_contextual) + 0.2*[same source_type] + 0.2*[same place], with the place term off for place-named queries (D89).
- Parameters that must be right: `lambda` (tuned on dev over {1.0,0.9,0.8,0.7,0.5}, selection rule in D89); the similarity must be the document-document similarity, not document-query; selection is greedy from the already-ranked candidates (our reranked top 30).
- Caveat for this project: the popular/hidden-gem axis cannot be evaluated (0 hidden_gem in the frozen corpus, D44); only source-type and place diversity are measured.
