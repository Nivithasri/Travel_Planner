# Corpus statistics (Phase 2, NOT frozen)

Command: `python scripts/process.py --write` (offline; reads data/raw records only).
Token counter: **regex-proxy**. Parameters: {"max_incl_overlap": 300, "min": 200, "near_dup_threshold": 0.8, "overlap_max": 40, "shingle": 5, "tail_merge_below": 100, "target": 250}.

## Reconciliation

- records: 97 in = 96 kept + 1 exact duplicates + 0 hold-out (url/title)
- chunks: 1419 created = 1389 out + 30 hold-out (text) + 0 near duplicates

## Size: 1389 chunks

Estimate 1943 (range 1619 to 2429, D39 (ii), LOG.md 2026-10-19); difference -554.
Tokens per chunk: min 55, p50 256, p95 278, max 300; under 200: 54; over 300: 0. Chunking: {"merged_tails": 16, "oversize_splits": 0, "short_tails": 10}.

## Chunks per source_type

| value | chunks |
|---|---|
| blog | 242 |
| reference | 558 |
| wikivoyage | 589 |

## Chunks per state

| value | chunks |
|---|---|
| AP | 111 |
| KA | 438 |
| KL | 263 |
| PY | 73 |
| TN | 343 |
| TS | 161 |

## Chunks per published year

| value | chunks |
|---|---|
| 2004 | 9 |
| 2005 | 9 |
| 2006 | 9 |
| 2007 | 21 |
| 2008 | 16 |
| 2011 | 52 |
| 2012 | 8 |
| 2013 | 6 |
| 2015 | 3 |
| 2018 | 41 |
| 2019 | 5 |
| 2021 | 20 |
| 2022 | 21 |
| 2023 | 7 |
| 2024 | 8 |
| 2025 | 7 |
| none | 1147 |

## Chunks per revised year (wiki API revision date)

| value | chunks |
|---|---|
| 2026 | 1104 |
| none | 285 |

## Chunks per date_kind

| value | chunks |
|---|---|
| published | 242 |
| revised | 1104 |
| sitemap_lastmod | 43 |

## Chunks per place

| place | chunks | by source_type |
|---|---|---|
| Alappuzha | 43 | {"reference": 22, "wikivoyage": 21} |
| Auroville | 27 | {"reference": 12, "wikivoyage": 15} |
| Badami | 32 | {"blog": 18, "reference": 7, "wikivoyage": 7} |
| Bengaluru | 131 | {"blog": 29, "reference": 48, "wikivoyage": 54} |
| Bhadrachalam | 9 | {"reference": 1, "wikivoyage": 8} |
| Chennai | 90 | {"blog": 3, "reference": 41, "wikivoyage": 46} |
| Gokarna | 47 | {"blog": 20, "reference": 8, "wikivoyage": 19} |
| Hampi | 81 | {"blog": 14, "reference": 35, "wikivoyage": 32} |
| Hyderabad | 124 | {"blog": 36, "reference": 46, "wikivoyage": 42} |
| Kanniyakumari | 39 | {"blog": 13, "reference": 9, "wikivoyage": 17} |
| Kochi | 125 | {"blog": 5, "reference": 73, "wikivoyage": 47} |
| Kodagu | 62 | {"blog": 39, "reference": 11, "wikivoyage": 12} |
| Kodaikanal | 32 | {"reference": 12, "wikivoyage": 20} |
| Madurai | 76 | {"blog": 6, "reference": 31, "wikivoyage": 39} |
| Mahabalipuram | 28 | {"blog": 3, "reference": 6, "wikivoyage": 19} |
| Munnar | 28 | {"reference": 7, "wikivoyage": 21} |
| Mysuru | 85 | {"blog": 10, "reference": 27, "wikivoyage": 48} |
| Ooty | 48 | {"blog": 13, "reference": 14, "wikivoyage": 21} |
| Pondicherry | 46 | {"blog": 3, "reference": 13, "wikivoyage": 30} |
| Rameswaram | 30 | {"reference": 21, "wikivoyage": 9} |
| Srisailam | 8 | {"reference": 3, "wikivoyage": 5} |
| Tirupati | 43 | {"blog": 8, "reference": 24, "wikivoyage": 11} |
| Varkala | 22 | {"reference": 4, "wikivoyage": 18} |
| Visakhapatnam | 60 | {"blog": 6, "reference": 33, "wikivoyage": 21} |
| Warangal | 28 | {"blog": 16, "reference": 12} |
| Wayanad | 45 | {"reference": 38, "wikivoyage": 7} |

Null-date share: 0.8258. Freshness (published dates only, window 2025-04-18 to 2026-10-18): 0 chunks, share 0.0.

## Removed text (characters per rule)

- blocked_section: 17744
- citation_marker: 74
- empty_header: 308
- image_credit: 228
- inline_link_joins: 244
- notes_bibliography: 7895
- read_more_link: 312
- related_footer: 5620
- share_header: 2016

## Drops

- chunk: 30
- exact_duplicate: 1

Exact duplicates:
- https://inditales.com/travelogue-ooty-kodaikanal/ under Kodaikanal (kept under Ooty)

Record-level hold-out drops (url/title):

Hold-out drops per hold-out, source and level:

| hold-out | source | level | n |
|---|---|---|---|
| Araku Valley | blog | chunk | 4 |
| Araku Valley | wikipedia | chunk | 1 |
| Araku Valley | wikivoyage | chunk | 3 |
| Thanjavur | wikipedia | chunk | 5 |
| Thanjavur | wikivoyage | chunk | 3 |
| Thekkady | wikipedia | chunk | 1 |
| Thekkady | wikivoyage | chunk | 13 |

Boundary-free stem scan of the OUTPUT chunks (case-insensitive substring; hits are other words or concatenations, listed for review):
- brihad: 0
- tanj: 1
- thanj: 0
- araku: 1
- thekk: 5
- kumil: 1
- chikm: 0
- chikk: 7
- periyar: 17

## D23 / D33 dry run (nothing removed)

| place | types in chunks | passes (chunks) | passes (raw) |
|---|---|---|---|
| Hyderabad | blog, reference, wikivoyage | True | True |
| Bengaluru | blog, reference, wikivoyage | True | True |
| Chennai | blog, reference, wikivoyage | True | True |
| Pondicherry | blog, reference, wikivoyage | True | True |
| Visakhapatnam | blog, reference, wikivoyage | True | True |
| Kochi | blog, reference, wikivoyage | True | True |
| Auroville | reference, wikivoyage | False | False |
| Mysuru | blog, reference, wikivoyage | True | True |
| Ooty | blog, reference, wikivoyage | True | True |
| Madurai | blog, reference, wikivoyage | True | True |
| Tirupati | blog, reference, wikivoyage | True | True |
| Hampi | blog, reference, wikivoyage | True | True |
| Kanniyakumari | blog, reference, wikivoyage | True | True |
| Rameswaram | reference, wikivoyage | False | False |
| Srisailam | reference, wikivoyage | False | False |
| Kodaikanal | reference, wikivoyage | False | True |
| Kodagu | blog, reference, wikivoyage | True | True |
| Munnar | reference, wikivoyage | False | False |
| Mahabalipuram | blog, reference, wikivoyage | True | True |
| Wayanad | reference, wikivoyage | False | False |
| Alappuzha | reference, wikivoyage | False | False |
| Badami | blog, reference, wikivoyage | True | True |
| Gokarna | blog, reference, wikivoyage | True | True |
| Warangal | blog, reference | True | True |
| Varkala | reference, wikivoyage | False | False |
| Bhadrachalam | reference, wikivoyage | False | False |

D23 at chunk level: 17 of 26 pass; 9 would be excluded at the freeze (D33).
