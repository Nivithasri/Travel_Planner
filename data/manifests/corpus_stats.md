# Corpus statistics (Phase 2, NOT frozen)

Command: `python scripts/process.py --write` (offline; reads data/raw records only).
Token counter: **tokenizer.json sha256:21106b6d7dab**. Parameters: {"max_incl_overlap": 300, "min": 200, "near_dup_threshold": 0.8, "overlap_max": 40, "shingle": 5, "tail_merge_below": 100, "target": 250}.

## Reconciliation

- records: 97 in = 96 kept + 1 exact duplicates + 0 hold-out (url/title)
- chunks: 1726 created = 1696 out + 30 hold-out (text) + 0 near duplicates

## Size: 1696 chunks

Estimate 1943 (range 1619 to 2429, D39 (ii), LOG.md 2026-10-19); difference -247.
Tokens per chunk: min 52, p50 259, p95 287, max 300; under 200: 55; over 300: 0. Chunking: {"merged_tails": 14, "oversize_splits": 0, "short_tails": 17}.

## Chunks per source_type

| value | chunks |
|---|---|
| blog | 283 |
| reference | 696 |
| wikivoyage | 717 |

## Chunks per state

| value | chunks |
|---|---|
| AP | 142 |
| KA | 524 |
| KL | 333 |
| PY | 89 |
| TN | 416 |
| TS | 192 |

## Chunks per published year

| value | chunks |
|---|---|
| 2004 | 11 |
| 2005 | 10 |
| 2006 | 9 |
| 2007 | 25 |
| 2008 | 18 |
| 2011 | 60 |
| 2012 | 9 |
| 2013 | 7 |
| 2015 | 3 |
| 2018 | 47 |
| 2019 | 6 |
| 2021 | 25 |
| 2022 | 26 |
| 2023 | 9 |
| 2024 | 11 |
| 2025 | 7 |
| none | 1413 |

## Chunks per revised year (wiki API revision date)

| value | chunks |
|---|---|
| 2026 | 1358 |
| none | 338 |

## Chunks per date_kind

| value | chunks |
|---|---|
| published | 283 |
| revised | 1358 |
| sitemap_lastmod | 55 |

## Chunks per place

| place | chunks | by source_type |
|---|---|---|
| Alappuzha | 55 | {"reference": 29, "wikivoyage": 26} |
| Auroville | 33 | {"reference": 15, "wikivoyage": 18} |
| Badami | 38 | {"blog": 21, "reference": 9, "wikivoyage": 8} |
| Bengaluru | 152 | {"blog": 33, "reference": 56, "wikivoyage": 63} |
| Bhadrachalam | 12 | {"reference": 2, "wikivoyage": 10} |
| Chennai | 106 | {"blog": 3, "reference": 49, "wikivoyage": 54} |
| Gokarna | 57 | {"blog": 25, "reference": 10, "wikivoyage": 22} |
| Hampi | 100 | {"blog": 17, "reference": 45, "wikivoyage": 38} |
| Hyderabad | 148 | {"blog": 42, "reference": 57, "wikivoyage": 49} |
| Kanniyakumari | 49 | {"blog": 16, "reference": 12, "wikivoyage": 21} |
| Kochi | 155 | {"blog": 6, "reference": 92, "wikivoyage": 57} |
| Kodagu | 74 | {"blog": 45, "reference": 13, "wikivoyage": 16} |
| Kodaikanal | 41 | {"reference": 16, "wikivoyage": 25} |
| Madurai | 94 | {"blog": 7, "reference": 37, "wikivoyage": 50} |
| Mahabalipuram | 33 | {"blog": 3, "reference": 8, "wikivoyage": 22} |
| Munnar | 37 | {"reference": 9, "wikivoyage": 28} |
| Mysuru | 103 | {"blog": 12, "reference": 32, "wikivoyage": 59} |
| Ooty | 54 | {"blog": 13, "reference": 16, "wikivoyage": 25} |
| Pondicherry | 56 | {"blog": 4, "reference": 16, "wikivoyage": 36} |
| Rameswaram | 39 | {"reference": 27, "wikivoyage": 12} |
| Srisailam | 10 | {"reference": 4, "wikivoyage": 6} |
| Tirupati | 56 | {"blog": 11, "reference": 32, "wikivoyage": 13} |
| Varkala | 27 | {"reference": 5, "wikivoyage": 22} |
| Visakhapatnam | 76 | {"blog": 7, "reference": 42, "wikivoyage": 27} |
| Warangal | 32 | {"blog": 18, "reference": 14} |
| Wayanad | 59 | {"reference": 49, "wikivoyage": 10} |

Null-date share: 0.8331. Freshness (published dates only, window 2025-04-18 to 2026-10-18; reference date from EVAL_REFERENCE_DATE in .env): 0 chunks, share 0.0. The reference date is used only here; it is in no chunk.

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
| Thanjavur | wikivoyage | chunk | 2 |
| Thekkady | wikipedia | chunk | 2 |
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
- periyar: 13

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
