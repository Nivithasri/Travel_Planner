# Rows 5 and 6 on dev: which filters fired and what they cost

condensed nDCG@10 (unjudged removed before the cut) of row 4 (expanded) and row 5 (+ filters), per dev query.

| query | filters from the parse | rejected of 100 (state/season/place_type) | exempt | row4 | row5 | delta |
|---|---|---|---|---|---|---|
| Q01 | {'state': ['KA'], 'place_type': ['city']} | 18/0/46 | 14 | 0.684 | 0.472 | -0.211 |
| Q03 | {'place_type': ['beach']} | 0/0/34 | 57 | 0.653 | 0.665 | +0.012 |
| Q06 | {'state': ['KA'], 'place_type': ['temple', 'heritage', 'region']} | 16/0/12 | 18 | 0.732 | 0.731 | -0.001 |
| Q09 | {'state': ['KA'], 'place_type': ['heritage', 'region']} | 29/0/7 | 52 | 1.000 | 1.000 | +0.000 |
| Q11 | {'state': ['KA'], 'place_type': ['hill_station']} | 27/0/30 | 42 | 0.715 | 0.729 | +0.014 |
| Q16 | {'state': ['AP'], 'place_type': ['temple']} | 46/0/35 | 13 | 0.873 | 0.648 | -0.225 |
| Q18 | {'state': ['TN'], 'place_type': ['temple']} | 29/0/29 | 40 | 1.000 | 0.870 | -0.130 |
| Q21 | {'state': ['KL'], 'season': 'monsoon', 'place_type': ['beach', 'city', 'heritage']} | 32/6/18 | 9 | 0.852 | 0.794 | -0.058 |
| Q28 | none | 0/0/0 | 0 | 0.861 | 0.861 | +0.000 |
| Q31 | {'place_type': ['heritage', 'hill_station', 'wildlife']} | 0/0/50 | 43 | 0.290 | 0.110 | -0.180 |
| Q33 | {'place_type': ['heritage']} | 0/0/48 | 47 | 0.532 | 0.294 | -0.238 |
| Q37 | none | 0/0/0 | 0 | 0.953 | 0.953 | +0.000 |
| Q39 | {'place_type': ['beach']} | 0/0/46 | 39 | 0.630 | 0.604 | -0.026 |

mean condensed nDCG@10: row 4 0.752, row 5 0.672

## Each filter kind alone (explicit filters taken from the parse), on row 4

| arm | cond nDCG@10 | queries where it fired |
|---|---|---|
| state only | 0.705 | 7 of 13 |
| season/month only | 0.747 | 1 of 13 |
| place_type only | 0.672 | 11 of 13 |

## Diagnostics (not frozen variants)

| arm | cond nDCG@10 | note |
|---|---|---|
| 5b: place_type also on rule-based rows | 0.688 | uses RULE-BASED place_type (D69) |
| filters on the plain hybrid base | 0.777 | row 4 is provisional |
| hybrid base without filters | 0.803 | = hybrid_contextual |
