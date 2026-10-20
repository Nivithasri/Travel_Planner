# Time-based / recency ranking (Li & Croft, "Time-Based Language Models", CIKM 2003)

WRITTEN FROM MEMORY, not from the paper. Verify before relying on it.

- Idea: for recency-sensitive needs, bias ranking toward newer documents with an exponential decay on document age (a prior that favours recent documents), rather than treating all documents as equally timely.
- Our adaptation (Phase 6, freshness row 6, D87): `final = rel_norm*(1-w_q) + w_q*f`, with `f = 0.5 ** (age_days / half_life(aspect))` for published-dated chunks and a neutral `f = 0.5` otherwise (a null/revised date is never treated as old, D3/D48). The decay is ASPECT-AWARE (fast half-life for closure/timing/price/safety, slow for food/other, none for scenery/history) and QUERY-AWARE (`w_q = w` only when the parsed query is time_sensitive, else 0).
- Parameters that must be right: half-lives fixed a priori (closure/timing/price/safety 180 d; food/other 730 d; scenery/history neutral); w tuned on dev; the reference date is EVAL_REFERENCE_DATE (D4), not today.
- MEASURED CAVEAT for this corpus: the only published dates are blog chunks, all rule_only, and the newest (2025-01-30) is outside the 18-month window (D32), so this row is mechanism-only and cannot show a freshness gain; fresh@10 stays 0. Freshness evidence is expected only from the Phase 7 live layer (D31, D32).
