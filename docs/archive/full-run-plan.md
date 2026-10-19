# Full-run plan (PROPOSAL, not executed, needs the owner's approval)

Status 2026-10-18: nothing in this plan has been run. The three slice places (Munnar, Hampi, Tirupati) are done.
The YouTube transcript path is stopped pending the owner's answer (robots.txt disallows `/youtubei/` and `/api/`,
see docs/sources.md), so the commands below use `--sources wikivoyage,wikipedia,web` and leave `youtube` out.

## Batches
Batch 1, the kinds not yet tested: Wayanad and Kodagu (district items), Pondicherry and Auroville (PY),
Hyderabad and Chennai (hubs).
Batch 2, only after the owner has reviewed batch 1 (17 places): Bengaluru, Visakhapatnam, Kochi, Mysuru, Ooty,
Madurai, Kanniyakumari, Rameswaram, Srisailam, Kodaikanal, Mahabalipuram, Alappuzha, Badami, Gokarna, Warangal,
Varkala, Bhadrachalam. (26 collectable = 3 slice + 6 + 17; the 4 hold-outs are never collected.)

## First contacts in batch 1
- `www.keralatourism.org` is contacted for the first time by **Wayanad** (its robots.txt is cached from the evidence
  round; `/sitemap.xml` is cached from the slice, so the cost is the page fetches that match the name, if any).
- `hrce.tn.gov.in` is contacted for the first time by **Chennai** (robots.txt 404 means no rules; its sitemap
  paths have never been fetched, so about 2 sitemap requests plus any matching pages).
- Hyderabad and Pondicherry do NOT contact a host known to fail TLS. Their web domains are only `sahapedia.org` and
  `www.inditales.com` (plus the two wikis). The hosts that failed in the evidence round (`telanganatourism.gov.in`,
  `tourism.telangana.gov.in`, `tourism.ap.gov.in`, `tourism.py.gov.in`) are not in `web_allowlist`, and `pondytourism.in`
  is robots-blocked and not listed either. They are in `unreachable_not_retested`, so even a future allowlist mistake
  would not contact them.

## TLS and DNS failures (implemented, tested in tests/test_common.py)
Recorded once per host, never retried, never treated as a block (no HTTP status, so `check_block` ignores them),
not counted against the request cap, and the host is not contacted again in the run. Config entries behind it:
`config/sources.json` -> `unreachable_not_retested` (6 hosts, preloaded as unreachable with `unreachable_reason`,
never contacted, 0 requests) and `run_limits.max_requests_per_run` (the cap; TLS/DNS attempts are excluded from it by
`HttpClient._once`, counted separately as `unreachable_attempts`). Note: `tourism.gov.in` and `wikitravel.org` are
in that list because of HTTP 403 on robots.txt in the evidence round, not TLS.

## Request cap and estimates (ESTIMATES, from the slice, labelled as such)
- Per place, without transcripts: 2 wiki requests, 0 to 3 matching pages per web domain, plus sitemap requests the
  first time a host is seen (cached 7 days). Batch 1 about 30 to 40 network requests; batch 2 about 90 to 110.
- Transcripts would add about 3 library requests per video, up to 5 videos per place: about 15 per place
  (estimate; the library does not report counts yet). Only after the owner decides.
- Proposed cap: batch 1 keeps `max_requests_per_run` = 250 (config value, `--max-requests` can only lower it).
  Batch 2: 250 is enough without transcripts; with transcripts it would be raised to 450 by a reviewed config commit.

## Commands (to run only after approval)
```
python scripts/collect.py --places Wayanad,Kodagu,Pondicherry,Auroville,Hyderabad,Chennai --sources wikivoyage,wikipedia,web
python scripts/collect.py --places Wayanad,Kodagu,Pondicherry,Auroville,Hyderabad,Chennai --sources wikivoyage,wikipedia,web --execute
python scripts/coverage_report.py
python estimate_corpus_size.py (deleted, D47)
```
The first line prints the plan only. Repeat the key-prefix leak scan and the hold-out check after the run.

## Abort behaviour and what survives
| Event | Exit | What happens |
|---|---|---|
| HTTP 403 or 429 from any site, or a transcript block | 3 | `BlockDetected`, whole run stops at once, no retry |
| Request cap reached | 4 | `RequestBudgetExhausted` (not a FetchError, no collector can swallow it), run stops |
| A hold-out place asked for | 2 | refused before any request |
| Deadline | 0 | place status `ingesting`, resumable |
| TLS/DNS failure | continues | host recorded unreachable once, skipped afterwards, not a block, not counted |
| Redirect to another site, from https to http, or to a robots-disallowed URL | continues | not followed, logged as `redirect-refused` |

On disk after any abort: every raw record already written (append-only, de-duplicated by url + content hash);
the status file of each collector that finished (the interrupted one is not recorded, so a rerun repeats it, mostly from
cache at 0 requests); the HTTP cache; the manifest lines; and `data/manifests/request_log.jsonl` up to the last request
(host, path without query, kind, status, from_cache, attempt, source). Nothing is rolled back.

## Per-host circuit breaker (IMPLEMENTED as D38, 2026-10-18; text below is the original proposal)
Today a run does NOT skip a host after one connection error: `HttpClient._once` marks a host unreachable only for TLS or DNS
failures, not for a connect timeout (batch 1: `www.keralatourism.org` /districts/wayanad/ failed with `transport:connect`
after the 10 s connect timeout, and nothing stopped the next request to it). Proposal: after 2 consecutive connect/timeout
failures on a host within one run, skip further requests to that host (no network, not counted against the cap, logged once as
"circuit-open"), reset at the next run, optionally try one probe after a delay. Effect on batch 2: `www.keralatourism.org`
serves Kochi, Alappuzha and Varkala (up to 8 pages each); if the host is down this saves up to about 24 requests at 10 s each
(about 4 minutes) and up to 24 cap units; the risk is a false trip on a transient fault, which the delayed probe limits.
