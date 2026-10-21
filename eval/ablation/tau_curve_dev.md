# Row 9: tau on dev (rule in LOG.md; row 8 = no correction)

| arm | triggered | nDCG@10 | cond nDCG@10 | MRR | P@5 | unjudged@10 | places@10 |
|---|---|---|---|---|---|---|---|
| row 8 (no correction) | - | 0.579 | 0.791 | 0.853 | 0.615 | 0.362 | 5.154 |
| row 9, tau 0.0611 | 5/13 | 0.574 | 0.793 | 0.808 | 0.600 | 0.369 | 5.154 |
| row 9, tau 0.1104 | 7/13 | 0.572 | 0.785 | 0.801 | 0.585 | 0.362 | 5.077 |
| row 9, tau 0.6314 | 10/13 | 0.578 | 0.784 | 0.801 | 0.600 | 0.346 | 5.000 |

chosen tau by the rule (highest dev nDCG@10, ties to the smaller): 0.6314

## Per query at the chosen tau 0.6314 (triggered queries only)

| query | triggers | rewrite | nDCG@10 before | after |
|---|---|---|---|---|
| Q01 | low_confidence | Mysuru family-friendly attractions for kids 3 days itinerary | 0.400 | 0.510 |
| Q03 | low_confidence | South India toddler-friendly beach destinations safety tips | 0.348 | 0.427 |
| Q06 | few_sources | budget backpacking accommodation Hampi hostel guesthouse cheap lodging and transport options | 0.872 | 0.919 |
| Q09 | low_confidence | offbeat tourist destinations Karnataka away from crowds 2024 travel guide | 0.473 | 0.472 |
| Q16 | few_sources | Tirupati Venkateswara Temple darshan timings, queue management tips, best time to visit, temple schedule 2024 | 0.729 | 0.729 |
| Q28 | low_confidence | South India two-week train and bus itinerary budget travel plan | 0.499 | 0.467 |
| Q31 | low_confidence | mid-budget one week nature and culture South India itinerary | 0.095 | 0.078 |
| Q33 | low_confidence | History lover South India best heritage sites with local cuisine recommendations | 0.318 | 0.114 |
| Q37 | low_confidence | Vivekananda Rock ferry schedule latest timings 2024 | 0.737 | 0.737 |
| Q39 | low_confidence | South Indian beaches currently closed or unsafe for swimming 2026 safety status | 0.514 | 0.519 |

row 9 minus row 8 at tau 0.6314: -0.001 nDCG@10 (D84 keep rule: >= -0.005)
