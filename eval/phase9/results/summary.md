# Phase 9 summary (generated; scores and counts only, no answer text)

## main_scores
| system | n | feasibility | groundedness | freshness | personalization |
|---|---|---|---|---|---|
| full | 10 | 2.3 | 3.8 | 4.5 | 2.6 |
| llm_only | 10 | 2.7 | 1.0 | 2.2 | 3.8 |
| plain_rag | 10 | 2.6 | 3.5 | 2.2 | 2.8 |

| first | second | dimension | n | mean_diff | ci | p | p_holm |
|---|---|---|---|---|---|---|---|
| full | plain_rag | feasibility | 10 | -0.3 | [-0.9, 0.2] | 0.5312 | 1.0 |
| full | plain_rag | groundedness | 10 | 0.3 | [-0.5, 1.0] | 0.6172 | 1.0 |
| full | plain_rag | freshness | 10 | 2.3 | [1.9, 2.8] | 0.002 | 0.0234 |
| full | plain_rag | personalization | 10 | -0.2 | [-0.9, 0.4] | 0.7812 | 1.0 |
| full | llm_only | feasibility | 10 | -0.4 | [-1.0, 0.2] | 0.3594 | 1.0 |
| full | llm_only | groundedness | 10 | 2.8 | [2.0, 3.5] | 0.0039 | 0.0391 |
| full | llm_only | freshness | 10 | 2.3 | [1.4, 3.1] | 0.0078 | 0.0703 |
| full | llm_only | personalization | 10 | -1.2 | [-1.8, -0.7] | 0.0078 | 0.0703 |
| plain_rag | llm_only | feasibility | 10 | -0.1 | [-0.9, 0.7] | 1.0 | 1.0 |
| plain_rag | llm_only | groundedness | 10 | 2.5 | [1.9, 3.0] | 0.002 | 0.0234 |
| plain_rag | llm_only | freshness | 10 | 0.0 | [-0.7, 0.7] | 1.0 | 1.0 |
| plain_rag | llm_only | personalization | 10 | -1.0 | [-1.8, -0.2] | 0.0898 | 0.6289 |


## live_scores
| system | n | feasibility | groundedness | freshness | personalization |
|---|---|---|---|---|---|
| full_corpus | 10 | 2.4 | 3.8 | 4.0 | 2.7 |
| full_live | 10 | 1.9 | 2.9 | 2.1 | 2.5 |

| first | second | dimension | n | mean_diff | ci | p | p_holm |
|---|---|---|---|---|---|---|---|
| full_live | full_corpus | feasibility | 10 | -0.5 | [-0.9, -0.1] | 0.125 | 0.25 |
| full_live | full_corpus | groundedness | 10 | -0.9 | [-1.3, -0.5] | 0.0156 | 0.0469 |
| full_live | full_corpus | freshness | 10 | -1.9 | [-2.7, -1.1] | 0.0078 | 0.0312 |
| full_live | full_corpus | personalization | 10 | -0.2 | [-0.5, 0.0] | 0.5 | 0.5 |


## live arm (counts)
```
{
 "requests": 10,
 "utc_dates": [
  "2026-10-24"
 ],
 "same_day": true,
 "status": {
  "ok": 9,
  "empty": 1
 },
 "totals": {
  "searches": 20,
  "results": 100,
  "in_scope": 57,
  "in_plan": 54,
  "dropped_excluded": 1,
  "merged": 34,
  "cited": 10,
  "not_cited": 24
 },
 "requests_with_live_cited": 5
}
```

## latency
| arm | node | n | median_s | p90_s |
|---|---|---|---|---|
| latency_corpus | critic | 10 | 0.0 | 0.01 |
| latency_corpus | finalize | 10 | 0.0 | 0.0 |
| latency_corpus | limiter_wait | 10 | 51.98 | 55.61 |
| latency_corpus | live | 10 | 0.0 | 0.0 |
| latency_corpus | merge_grade | 10 | 0.01 | 0.02 |
| latency_corpus | parse | 10 | 1.82 | 2.0 |
| latency_corpus | plan | 10 | 0.0 | 0.02 |
| latency_corpus | retrieve | 10 | 0.33 | 0.39 |
| latency_corpus | total | 10 | 56.48 | 60.62 |
| latency_corpus | write | 10 | 54.21 | 58.77 |

