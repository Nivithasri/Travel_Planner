# Significance testing for IR (notes; Smucker, Allan & Carterette 2007; Bassani, ranx 2022)

WRITTEN FROM MEMORY, not from the papers. Verify before relying on it.

- Why a paired test: two retrievers are run on the SAME queries, so compare them per query (paired), not as two independent samples. The per-query metric here is nDCG@10.
- Randomization (permutation) test, paired: under H0 the two systems are exchangeable per query, so for each query the sign of the difference may be flipped. The p-value is the fraction of all sign-flip assignments whose mean |difference| is at least the observed one (two-sided). Smucker et al. found the paired randomization and bootstrap tests more appropriate for IR than the paired t-test or the sign/Wilcoxon tests.
  - Dev (13 queries): exact enumeration, 2^13 = 8,192 assignments.
  - Test (27 queries): 2^27 is too many, so 100,000 seeded Monte Carlo flips.
- Effect size: the mean per-query difference with a 95% paired bootstrap CI (resample queries with replacement). Reported next to the p-value; a significant p with a tiny effect is still reported as tiny.
- Multiple comparisons: several consecutive-row comparisons inflate the false-positive rate, so apply Holm-Bonferroni (step-DOWN: sort p ascending, compare the i-th to alpha/(m-i+1), stop at the first non-rejection). Show raw and adjusted p.
- Phase that uses it: 6 (final table, D92). Own implementation (ranx not installed); validated in tests against exact enumeration and against scipy.stats.ttest_rel on the same pairs.
- Parameters/pitfalls that must be right: the flip is PER QUERY (paired), never a shuffle across queries; queries with no relevant chunk are dropped pairwise; with 27 test queries only differences near 0.1 nDCG are detectable, and cumulatively built rows are not independent.
