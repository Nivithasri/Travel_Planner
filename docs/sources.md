# Sources: domain allowlist for the web collector

Status (2026-10-18): **allowlist in config/sources.json: sahapedia.org and aptourism.gov.in (approved) plus Kerala
Tourism, inditales.com, Tirumala and hrce.tn.gov.in (owner accepted the risk on 2026-10-18, see the update at the end).
TN Tourism, Karnataka Tourism, ASI and Incredible India stay ask-first. The web collector is built and has run on a
3-place slice.** Original draft text follows. Nothing from any
domain below has been collected into the corpus; only its robots.txt, its homepage and its terms
pages were read, to decide whether collecting from it is acceptable.

Evidence file: `data/manifests/source_evidence.json` (retrieved 2026-10-18 13:19 UTC).
Regenerate it with `python gather_source_evidence.py (deleted, D47)` (live network; needs `CONTACT_EMAIL`
in `.env`). The quotes below are copied from that file. This is a technical and policy reading by
the project, not legal advice.

## How the evidence was gathered

- One request at a time per host, at least 1 s apart, through `src/common/http.py`. The
  User-Agent is `TravelPlannerResearch/0.1` plus the contact address from `.env`.
- robots.txt rules are applied to that User-Agent (it falls under `User-agent: *`). Following the
  robots standard: a missing robots.txt (404) means no rules; a robots.txt that cannot be fetched
  means disallowed; a page is only fetched if robots.txt allows it.
- Terms pages were found by following homepage links that mention terms, conditions, disclaimer,
  copyright or legal. Where none was found, usual paths were tried. Sentences about crawling,
  copying, commercial use and licences were extracted by keyword, so a clause phrased unusually
  could be missed. That is why the verdict for a site with no terms found is "ask first", never
  "include".
- Nothing was done to get around a block (no other User-Agent, no retries after a 403).

## Decision rule

| Verdict | Meaning |
|---|---|
| **INCLUDE** | robots.txt allows our User-Agent, and the terms either grant a licence that covers this use or contain no restriction on copying or automated access. |
| **ASK FIRST** | robots.txt allows it, but the terms restrict copying or no terms could be found. Not collected until the owner has written permission (or a terms page is found). |
| **EXCLUDE** | robots.txt or the terms forbid it, or the site could not be verified from this network. |

Our use: a non-commercial university research project; text stored locally in git-ignored folders
(`data/raw`, `data/live`), never redistributed; itineraries cite the source URL.

## Summary

| Domain | Verdict | Why, in one line |
|---|---|---|
| `sahapedia.org` | **INCLUDE** | Content shared under CC BY-NC-SA 4.0 (non-commercial fits); robots allows. |
| `aptourism.gov.in` | **INCLUDE** | No robots.txt (no rules); terms are only a disclaimer, no copying restriction. |
| `www.tamilnadutourism.tn.gov.in` | ASK FIRST | robots allows all; terms: personal, non-commercial use, "will not be copied or posted on any network computer". |
| `www.karnatakatourism.org` | ASK FIRST | robots allows all, has a sitemap; terms: "Unauthorized use or reproduction of any materials is prohibited." |
| `www.keralatourism.org` | ASK FIRST | robots allows (admin paths only blocked); no terms page exists at the usual paths. |
| `asi.nic.in` | ASK FIRST | No robots.txt; terms: reproduction free "after taking proper permission by sending a mail". |
| `www.incredibleindia.gov.in` | ASK FIRST | robots allows; terms forbid reproducing or publishing content without express authorisation. |
| `tirumala.org` | ASK FIRST | robots allows all; no terms page found. Valuable for darshan timings. |
| `hrce.tn.gov.in` | ASK FIRST | No robots.txt; no terms page found. Temple department of Tamil Nadu. |
| `www.inditales.com` | ASK FIRST | robots allows all; no terms page; footer "Copyright ... Anuradha Goyal" (a personal blog). |
| `whc.unesco.org` | EXCLUDE | Terms prohibit web scraping by automated systems; robots blocks 12 AI crawlers. |
| `pondytourism.in` | EXCLUDE | robots.txt is `User-agent: *` / `Disallow: /`. |
| `telanganatourism.gov.in`, `tourism.telangana.gov.in`, `tourism.ap.gov.in`, `tourism.py.gov.in` | EXCLUDE (unverifiable) | Connection timeouts or TLS failures from this network; robots.txt and terms could not be read. |
| `tourism.gov.in`, `wikitravel.org` | EXCLUDE (unverifiable) | HTTP 403 even for robots.txt. |
| `periyartigerreserve.org` | EXCLUDE | A ticket-booking portal; no terms page (every guessed path returns the same booking shell). |
| `www.tripoto.com` | EXCLUDE | Terms forbid downloading and commercial use without written consent; 20 crawlers blocked. |
| `www.holidify.com` | EXCLUDE | No terms page can be found (all usual paths are 404); commercial site. |
| `www.thrillophilia.com` | EXCLUDE | Copyright policy forbids copying and republishing; about 90 crawlers blocked. |
| `www.lonelyplanet.com` | EXCLUDE | Terms: no commercial use, text and data mining prohibited, only search-engine crawlers allowed. |
| `www.tripadvisor.in` | EXCLUDE | HTTP 403 to our User-Agent; 621 Disallow rules; AI crawlers blocked. |
| `www.thehindu.com` | EXCLUDE | News publisher; 96 Disallow rules, 14 crawlers (several AI) blocked; no terms page found. |
| `www.nativeplanet.com` | EXCLUDE | 20 crawlers blocked; terms and privacy paths return HTTP 410. |

The allowlist this proposes is therefore **two domains**, plus eight that need permission first.

## Evidence per domain

### INCLUDE

**sahapedia.org** (heritage and culture encyclopedia)
- robots.txt (200): 34 Disallow rules under `*`, all CMS/admin paths (`/core/`, `/profiles/`,
  `/README.md`, ...); no crawler is fully blocked; no Crawl-delay; no sitemap line. `/`,
  `/sitemap.xml` and the content paths we would use are allowed.
- Terms, https://sahapedia.org/terms-use: "Users may submit, upload and post ... Content, data or
  information ... under a CC-BY-NC-SA 4.0 licence." "User Content will be stored by Sahapedia and
  may be shared and distributed by Sahapedia, as well as other Users of the Platform, under a
  CC-BY-NC-SA 4.0 licence." "Sahapedia has complete editorial discretion over the Content that it
  creates, shares and distributes, and may make changes ... as permitted by the terms of the
  CC-BY-NC-SA 4.0 licence."
- Reading: content is distributed under CC BY-NC-SA 4.0. Non-commercial research use with
  attribution fits; ShareAlike matters only if we redistribute derived text, which we do not.
- Limits: the terms page speaks about "Content" in general; individual articles may carry their
  own licence line, which the collector should record per page (and skip pages that differ).
  The project must stay non-commercial. Its articles are heritage and culture, rarely dated, so
  it helps `history` and `scenery` aspects, not freshness.

**aptourism.gov.in** (Andhra Pradesh Tourism Authority)
- robots.txt: HTTP 404, so no rules apply. Homepage 200.
- Terms, https://aptourism.gov.in/terms-and-conditions/ (3,647 chars): accuracy disclaimer,
  liability, links to outside sites, governing law. The only copyright sentence is "APTA, cannot
  authorise the use of copyrighted materials contained in linked websites." Nothing restricts
  copying or automated access.
- Reading: the terms are silent, which is not the same as permission. This is a public body
  promoting tourism, and a low-volume polite crawl (one request per second, a few pages per place)
  is low risk. A courtesy email to APTA is still recommended.
- Limits: covers AP places only (Visakhapatnam, Tirupati, Araku Valley, Srisailam).

### ASK FIRST

**www.tamilnadutourism.tn.gov.in** (Tamil Nadu Tourism). robots.txt (200): no rules, no sitemap.
Terms https://www.tamilnadutourism.tn.gov.in/terms-of-use: "Use of the software and information from
this site is subject to the End-User License Agreement, is for informational and non-commercial or
personal use only, and will not be copied or posted on any network computer or broadcast, or
published in any media ... Use for any other purpose is prohibited by law." Storing pages in a local
index is copying, so written permission is needed. The homepage URL without `www` returned HTTP 500
on one run.

**www.karnatakatourism.org** (Karnataka Tourism). robots.txt (200): no rules, 1 sitemap. Terms
https://www.karnatakatourism.org/en/terms-conditions/: "All content (text, images, videos, logos,
etc.) is the property of Karnataka Tourism Department or licensed third parties. Unauthorized use
or reproduction of any materials is prohibited." Technically easy (sitemap), but needs permission.

**www.keralatourism.org** (Kerala Tourism). robots.txt (200): 16 Disallow rules for admin and
internal paths (`/Administration/`, `/wts/`, `/admin/`, `/inc/`, ...); no sitemap line; no crawler
fully blocked. No terms page: `/terms-of-use` and `/disclaimer` return HTTP 200 with the same
1,281-character "Page not found" text, `/terms-and-conditions` returns HTTP 500. The privacy policy
(8,645 chars) has no reproduction clause. Silence of a missing page is not permission.

**asi.nic.in** (Archaeological Survey of India; protects Hampi, Mahabalipuram, the Chola temples).
robots.txt returns an HTML page, so no rules. Terms https://asi.nic.in/pages/Term-&-Conditions/:
"Material featured on this website may be reproduced free of charge after taking proper permission
by sending a mail to us." "Wherever the material is being published or issued to others, the source
must be prominently acknowledged." Copyright Policy: "This contents of this website may not be
reproduced partially or fully, without due permission from Archaeological Survey of India". The
route is clear: one email, then attribute the source.

**www.incredibleindia.gov.in** (Ministry of Tourism). robots.txt (200): 1 Disallow rule
(`/content/dam/incredible-india-content-hub/`), 1 sitemap. Terms (25,060 chars): "Unless you are
expressly authorized by us, or otherwise by law, you must not ... sell, reproduce, publish,
distribute, communicate to the public ... or otherwise use any of the Content". Needs authorisation.

**tirumala.org** (Tirumala Tirupati Devasthanams; darshan timings and closures). robots.txt (200):
no rules, no sitemap. Homepage 200. No terms page found by link discovery. High value for
time-sensitive queries about Tirupati; permission or a terms page should be sought.

**hrce.tn.gov.in** (Tamil Nadu temples department). robots.txt: HTTP 404 (no rules). Homepage 200.
No terms page found.

**www.inditales.com** (heritage-travel blog). robots.txt (200): no rules, 1 sitemap. No terms or
disclaimer page (404); the privacy page footer reads "Copyright 2004-2021 Anuradha Goyal". Normal
copyright applies, so ask the author.

### EXCLUDE

**whc.unesco.org** (UNESCO World Heritage; Hampi, Mahabalipuram, the Great Living Chola Temples).
robots.txt: 36 Disallow rules under `*`; 12 crawlers fully blocked (AI2Bot, Amazonbot,
Applebot-Extended, Bytespider, CCBot, ClaudeBot, FacebookBot, GPTBot, Meta-ExternalAgent,
cohere-training-data-crawler, omgili, omgilibot). Terms https://www.unesco.org/en/terms-use define
"Web Scraping" as "the automated extraction, harvesting or copying of Materials from a Site" and say
"Any access to or use of a Site or Material accessible through it by any person or Automated System
that is contrary to the UNESCO Terms of Use ... is unauthorized and prohibited." The syndication
page: "Sections that are not made available for syndication ... may not be 'scraped' or otherwise
syndicated in any manner." Wikipedia and Wikivoyage already cover these sites under CC BY-SA.

**pondytourism.in** (Puducherry tourism). robots.txt, served over HTTP only (HTTPS fails with an SSL
error): `User-agent: *` / `Disallow: /`. A blanket block.

**telanganatourism.gov.in, tourism.telangana.gov.in, tourism.ap.gov.in, tourism.py.gov.in.** These
resolve in DNS but the connection fails from this network: connect timeout (telanganatourism.gov.in,
both ports), SSL error (tourism.telangana.gov.in, tourism.ap.gov.in), connection error
(tourism.py.gov.in). robots.txt could not be read, so by the rule above they are disallowed.
Re-testing from another network might change this; Telangana and Puducherry have no usable official
source until then.

**tourism.gov.in, wikitravel.org.** HTTP 403 for robots.txt itself.

**periyartigerreserve.org.** robots.txt allows (14 Disallow rules for login/profile paths,
Crawl-delay 1, 1 sitemap), but the site is a ticket-booking portal. Every guessed terms path
(`/terms-and-conditions`, `/Disclaimer`, `/privacy-policy`) returns the same 527-character booking
page, so there are no terms and almost no travel text. Thekkady is a hold-out place anyway.

**www.tripoto.com.** Terms: "The Website grants you a limited sub-license to access and make personal
use of the content therein and not to download (other than page caching) or modify it ... except
with express written consent of the Company." "... may not be reproduced, duplicated, copied, sold,
resold, visited, or otherwise exploited for any commercial purpose without express written
consent." robots: 16 Disallow rules; 20 crawlers fully blocked (GPTBot, ClaudeBot, CCBot,
anthropic-ai, ...).

**www.holidify.com.** robots allows (7 Disallow rules, no crawler blocked) but no terms page can be
found (homepage links and `/terms-of-service`, `/terms`, `/terms-and-conditions`,
`/privacy-policy` are all HTTP 404). A commercial site whose rules cannot be read.

**www.thrillophilia.com.** Copyright policy https://www.thrillophilia.com/copyright-policies: "No
visitor or third party may copy, reproduce, distribute, modify, republish, upload, transmit,
perform, or create derivative works from any content without prior written consent from
Thrillophilia." robots: 13 Disallow rules plus about 90 named crawlers fully blocked.

**www.lonelyplanet.com.** Terms https://www.lonelyplanet.com/legal/website-terms: "No commercial use
of Sites is permitted." "You may use the Sites for personal, non-commercial purposes only." Users are
"specifically prohibited ... from performing any 'text and data mining' (TDM) activities", while
"we allow public search engine operators to use web crawlers". robots blocks GPTBot among others.

**www.tripadvisor.in.** robots.txt is readable (621 Disallow rules under `*`; 16 crawlers fully
blocked including GPTBot, ClaudeBot, CCBot), but the homepage returns HTTP 403 to our User-Agent.
We do not try to evade it.

**www.thehindu.com.** A newspaper. robots.txt: 96 Disallow rules under `*`; 14 crawlers fully
blocked (GPTBot, ClaudeBot, anthropic-ai, PerplexityBot, CCBot, ...). No terms page found (usual
paths 404; link discovery returned article pages). Closure news is valuable but the publisher
signals it does not want automated collection.

**www.nativeplanet.com.** robots.txt: 11 Disallow rules; 20 crawlers fully blocked (GPTBot,
ClaudeBot, CCBot, Google-Extended, ...). Its terms, privacy and disclaimer paths return HTTP 410.

## Not covered by this allowlist

- **Wikivoyage and Wikipedia** are read through their APIs (CC BY-SA; the User-Agent policy of
  Wikimedia is followed). Their terms were not re-read for this document.
- **Reddit and YouTube** are accessed through their APIs with credentials, under their own API
  terms, which still need a separate review. YouTube transcripts through `youtube-transcript-api`
  are unofficial scraping; that choice should be confirmed.

## What this means for the project

1. **Web coverage will be thin.** Only two domains are included, and only Andhra Pradesh and
   heritage/culture content come from them. The "two source types per core place" target in
   Phase 1 cannot be met from the web collector alone. Realistic paths: (a) ask the eight
   "ask first" owners for permission; (b) get Reddit credentials and a YouTube key; (c) accept
   that the check in `coverage_report.py` will show which places fall short. Wikivoyage plus
   Wikipedia count as one reference-wiki family (D19), so they do not satisfy the target by
   themselves.
2. **Hold-outs** (Chikmagalur, Araku Valley, Thekkady, Thanjavur) are not collected into the
   snapshot at all (D17), so they do not depend on this list.
3. **Time-sensitive content** (closures, timings) would come mostly from the "ask first" official
   sites (Tirumala, HRCE, the state tourism boards). Without permission the freshness slice will rest
   on Reddit and YouTube.

## Decisions needed from the owner

1. Approve the two INCLUDE domains (`sahapedia.org`, `aptourism.gov.in`) as the web allowlist.
2. Which "ask first" owners to write to. I can draft the permission emails; they would be sent
   from your address, not by me. Until a written yes arrives, none of them is collected.
3. Which `source_type` Sahapedia pages get. The contract has `reddit|blog|youtube|wikivoyage|official`;
   Sahapedia is an encyclopedia, so either map it to `blog`, or add a type through a decisions entry.
4. Whether to re-test the four unreachable state sites from another network.

## Proposed shape of `config/sources.json` (not created yet)

```json
{
  "web_allowlist": [
    {"domain": "sahapedia.org", "source_type": "TBD", "license": "CC BY-NC-SA 4.0",
     "min_delay_s": 1.0, "max_pages_per_place": 15},
    {"domain": "aptourism.gov.in", "source_type": "official", "license": "terms silent",
     "min_delay_s": 1.0, "max_pages_per_place": 15}
  ],
  "ask_first": ["www.tamilnadutourism.tn.gov.in", "www.karnatakatourism.org", "www.keralatourism.org",
                "asi.nic.in", "www.incredibleindia.gov.in", "tirumala.org", "hrce.tn.gov.in",
                "www.inditales.com"]
}
```

## Owner decisions, 2026-10-18

- Approved: `sahapedia.org` (source_type `reference`, D22) and `aptourism.gov.in` (`official`). Recorded in
  `config/sources.json` with the robots evidence above.
- Risk acceptance ("robots allows, no terms found, user accepted risk on 2026-10-18") was offered for Kerala
  Tourism, inditales.com, Tirumala and hrce.tn.gov.in, but the per-site answer was left blank. Treated as NOT
  accepted: all four stay ask-first. Add the entry to `config/sources.json` when the owner answers YES.
- Restricted-terms sites (Tamil Nadu Tourism, Karnataka Tourism, ASI, Incredible India) stay ask-first. Draft
  permission emails for TN Tourism, Karnataka Tourism and ASI: `docs/outreach.md`.
- The six unreachable sites are not re-tested. Failures recorded in `config/sources.json`
  (`unreachable_not_retested`) and above.
- YouTube transcripts use `youtube-transcript-api`, which reads YouTube's web interface, not the official API
  (the API key cannot download caption text). This is unofficial scraping that YouTube's terms may not allow and
  that is often blocked from cloud or repeated addresses. Risk accepted by the owner's instruction (optional, small
  caps, cached, backed off, stops on a block). Reddit is built and tested offline only; no credentials yet.

## Round 2: ten more candidate blogs (2026-10-18)

Evidence: `data/manifests/blog_evidence.json`. Regenerate with `python gather_source_evidence.py (deleted, D47) --domains <list> --out data/manifests/blog_evidence.json`
(31 requests). The ten were picked from two web searches and the owner's brief; the searches found no blog that
states a Creative Commons licence, and the homepage check for `creativecommons.org` links found none on any
reachable site. Stopped at ten as instructed. None is added to the allowlist.

| Domain | Verdict | Evidence |
|---|---|---|
| `www.travelindiasmart.com` | ASK FIRST | robots allows our User-Agent (1 Disallow; 32 named crawlers blocked, ours not among them); no terms page found; no licence. |
| `lakshmisharath.com` | ASK FIRST | robots allows, **Crawl-delay 20 s** (would be honoured); no terms page found; no licence. |
| `bhushavali.com` | ASK FIRST | no robots.txt (404); no terms page found; no licence. |
| `wanderingfeet.in` | ASK FIRST | robots allows (10 Disallow rules); no terms page found; no licence. |
| `www.global-gallivanting.com` | EXCLUDE | Terms: "You may not reproduce, distribute, or republish any content ... without our prior written permission"; also a non-Indian author. |
| `www.indiasomeday.com` | EXCLUDE | A trip-planning company; terms: no part "can be reproduced, copied, stored ... without the prior written consent of the Company". |
| `www.theblogchatter.com` | EXCLUDE | A blog aggregator and tour-package page; terms forbid copying and aggregating content. |
| `www.millispotter.com` | EXCLUDE | A tour operator; robots blocks GPTBot and CCBot; terms are booking terms. |
| `ianslife.in` | EXCLUDE (unverifiable) | robots.txt not reachable (connection error). |
| `www.travelogues-of-india.com` | EXCLUDE (unverifiable) | robots.txt not reachable (connection error). |

Reading: no independent South India blog in this sample can be collected without the author's permission. "No
terms found" is not permission (same rule as above). The blog source type therefore stays empty unless an author
says yes. The four ASK FIRST authors could be asked in one short email each; not drafted, not asked.

## Update 2026-10-18 (later): risk accepted for four sites

The owner answered YES for Kerala Tourism (`www.keralatourism.org`, KL places only), inditales
(`www.inditales.com`, blog), Tirumala (`tirumala.org`, Tirupati only) and the Tamil Nadu temples
department (`hrce.tn.gov.in`, TN places only). Recorded in `config/sources.json` as "robots allows, no terms
found, user accepted risk on 2026-10-18", with the robots evidence (above and in the config). `aptourism.gov.in`
is now scoped to AP places. These are not licences: the sites have not given permission, the owner accepted
the risk. Limits: 8 pages per place, 2 s between requests, sitemap discovery only, robots checked per URL.
A 403 or 429 from any site stops the whole run (no retry, no workaround) and the site goes back to ask-first.

## YouTube transcripts: robots.txt check (2026-10-18, one fetch through HttpClient)

`https://www.youtube.com/robots.txt` (HTTP 200, 792 bytes). Our User-Agent falls in the `*` group. Evaluated with the
Python robots parser for the three requests `youtube-transcript-api` 1.2.4 makes per video:

| Request | Path | Result |
|---|---|---|
| GET watch page | `/watch?v=...` | allowed |
| POST player data | `/youtubei/v1/player?...` | **DISALLOWED** (`Disallow: /youtubei/`) |
| GET caption track | `/api/timedtext?...` | **DISALLOWED** (`Disallow: /api/`) |

The video search (`www.googleapis.com/youtube/v3/search`) is the official Data API and is not covered by this file.
The 13 transcripts in the 3-place slice were fetched through the two disallowed paths before this check. Per the
owner's rule the transcript work is stopped and the decision is the owner's; nothing was decided or changed here.
Options for the owner (not chosen): drop transcripts and keep video titles/descriptions from the official API only;
find another permitted transcript source; or accept the risk explicitly as was done for the four sites.

YouTube's own terms on automated access (Terms of Service, API Services terms, transcript access) were NOT reviewed.
That review is left to the project owner by the owner's decision; it is not the implementer's call. Until the owner
decides, no transcript is fetched (no run was started after the robots check) and the 13 transcripts already in the cache
and data/raw are affected: Hampi V8DYqM_F0J8, OKCkLFVl80U, bIgiWIko6wo, 4abaE5HNrfA, xXsSIUGgCXc; Munnar 5G-NaCzvqhI,
ch8SzGmJKPM, 4UQDavvNpSY, VulJxJ0q_x4; Tirupati lKbXR8A9VZc, MATJb38REl4, _-Bo-8Mnsog, 8d0nDbYqGkU.

## Round 4 (2026-10-18): transcripts dropped (D27), batch 1

- YouTube transcripts are dropped (D27) because robots.txt disallows `/youtubei/` and `/api/`; the owner did not accept the
  risk and YouTube's terms were not reviewed. `config/sources.json` has `"youtube_transcripts": false`; the collector
  refuses with zero requests. The 13 transcripts and their cache entries were withdrawn; manifest lines were kept and
  "withdrawn" lines added. The Data API (descriptions, comments) is open item O10, not enabled.
- Batch 1 first contacts: `www.keralatourism.org` robots.txt 200 (clean, from the evidence-round cache), sitemap.xml
  served from cache (238 URLs), one page request (`/districts/wayanad/`) failed with a connection error after the 10 s
  connect timeout; recorded once, not retried or probed. `hrce.tn.gov.in` robots.txt 404 (no rules), `/sitemap.xml` and
  `/wp-sitemap.xml` both 404, so no pages for Chennai. `sahapedia.org` served one Wayanad page (revision date 2019-09-30,
  no publish date). No 403 or 429 anywhere.
- Every request is now written to `data/manifests/request_log.jsonl` (host, path without query, kind, status,
  from_cache, attempt, source); earlier rounds' requests are not in it.

## YouTube API policies read for the transcript exception (2026-10-18, D29)

Fetched once each through HttpClient (logged): developers.google.com robots.txt (200), the YouTube API Services
Developer Policies, the API Services Terms of Service and the quota-cost page. Findings, quotes shortened:
- Storage and retention: "not longer than 30 calendar days"; after that "either delete or refresh the stored data".
- Frozen corpus: NEGATIVE for API Data. "no further permissions or rights to API Data" beyond the stated ones, "including
  temporarily stored" data, and non-authorized API data may be kept 30 days at most. A frozen corpus built from search
  results (title, channel, date) held longer is not covered by that text.
- Scraping: the policies bind API clients: "must not ... scrape YouTube Applications" or "obtain scraped YouTube data".
  Transcripts fetched with youtube-transcript-api are scraped (not API) data. The same page lets search engines scrape
  only "in accordance with YouTube's robots.txt", which disallows the transcript paths.
- Audiovisual content: no copies "without YouTube's prior written approval". UNVERIFIED whether transcripts count.
- Consequences (ToS): YouTube "may suspend or terminate" access, "including any credentials", for any violation; on
  termination "delete all YouTube API Services (including all API Data)". Key and project are what is at risk.
- Quota: search.list has its own bucket (100 calls a day by default); videos.list 1 unit and commentThreads.list 1 unit per
  call. For 26 places: 26 searches, which fits.
Decision pending with the owner; no transcript or Data API collection is enabled.

## Round 5 close-out (2026-10-18, D30)
D27 stands: YouTube transcripts and the Data API stay off for anything stored. The three cached Data API search responses
(fetched 2026-10-18 14:20:47Z to 14:21:58Z) were deleted under the 30-day rule. `data/manifests`, `data/raw`, `docs/`, `LOG.md` and
the request log contain no video titles or channel names; one test fixture in `test_withdraw.py (deleted, D47)` (commit 89b4a9c) contains
a real channel name by mistake, reported to the owner and left untouched pending a decision (history is not rewritten).

## Licence ledger (2026-10-18, owner request; open item O12)

What each collected source's licence or terms status IS, as recorded. Quotes are copied from the page text fetched into
`data/manifests/source_evidence.json` (retrieved 2026-10-18) and are under 15 words; where no page text was read the
cell says UNVERIFIED. Not legal advice. "Redistribution" asks: may the cleaned chunks (a derivative of the page text) be
shared with the partner, pushed to origin or uploaded to Kaggle? The answer is the owner's decision (O12); nothing is
shared until then, and data/processed, data/raw, data/live and .cache stay gitignored and local.

| Source | Licence or terms status as recorded | Evidence (URL, date looked up, quote) | Attribution duty | Redistribution question |
|---|---|---|---|---|
| Wikipedia (raw `wikipedia`, type `reference`) | Record field: "CC BY-SA (Wikipedia; see page history for authors)". Wikimedia terms were NOT re-read this project | UNVERIFIED (no terms text fetched; sources.md "Not covered") | Under CC BY-SA: credit authors (page history link) and state the licence. Not yet implemented in chunks | ShareAlike: derived text shared must carry CC BY-SA. Open: do chunks count as an adaptation, and is a CC BY-SA corpus acceptable beside non-CC pages in one file? |
| Wikivoyage (raw `wikivoyage`) | Record field: "CC BY-SA (Wikivoyage; see page history for authors)". Terms NOT re-read | UNVERIFIED (same) | Same as Wikipedia | Same as Wikipedia |
| Sahapedia (raw `sahapedia`, type `reference`) | Terms page states CC BY-NC-SA 4.0 for content; record field "CC BY-NC-SA 4.0" (1 record, Wayanad). Individual articles may differ | https://sahapedia.org/terms-use, 2026-10-18: "under a CC-BY-NC-SA 4.0 licence" | Attribution required; NonCommercial; ShareAlike | Non-commercial project only. Open: partner/Kaggle sharing is likely fine if non-commercial and attributed, but ShareAlike and the NC condition would bind any published corpus |
| inditales.com (raw `blog`) | No terms page (404). Normal copyright of a personal blog. Robots allow. Owner accepted the RISK for collection on 2026-10-18; this is not a licence | robots/privacy footer, 2026-10-18: "Copyright 2004-2021" then "Anuradha Goyal" (footer, separator omitted) | No licence grants a right to reuse; at minimum cite the URL and author | Not covered by any licence: redistribution of chunks (even to the partner) is NOT permitted by anything on record. Needs the author's permission or stays local |
| aptourism.gov.in (raw `official`, AP only) | Terms are a disclaimer; silent on copying. Robots: 404 (no rules). Approved by the owner 2026-10-18 on that reading | https://aptourism.gov.in/terms-and-conditions/, 2026-10-18: "cannot authorise the use of copyrighted materials contained in linked websites" (the only copyright sentence) | None stated; cite the URL | Silence is not permission. Open: courtesy email recommended in sources.md; sharing not covered |
| www.keralatourism.org (raw `official`, KL only) | No terms page (the usual paths return a "Page not found" shell or HTTP 500); privacy policy has no reproduction clause. Robots allow. Risk accepted 2026-10-18 | https://www.keralatourism.org/, 2026-10-18: no terms text found, no quote | None stated; cite the URL | Not covered; needs permission or stays local. 0 official records collected from it so far |
| hrce.tn.gov.in (raw `official`, TN only) | No robots.txt (404), no terms page found. Risk accepted 2026-10-18 | https://hrce.tn.gov.in/, 2026-10-18: no terms text found, no quote | None stated | Not covered. 0 records collected (no sitemap, batch 1) |
| tirumala.org (raw `official`, Tirupati only) | Robots allow; no terms page found; sitemap answers HTTP 500. Risk accepted 2026-10-18 | https://tirumala.org/, 2026-10-18: no terms text found, no quote | None stated | Not covered. 0 records collected |

Reading: only the two wikis and Sahapedia carry a licence that could allow redistribution, each with attribution, and
Sahapedia only for non-commercial use. inditales, aptourism, Kerala Tourism, hrce and Tirumala have no licence on record, so
the cleaned chunks of those sources should be treated as local-only until the owner decides (O12). The chunk contract has
no field for licence or attribution; if sharing is allowed for some sources, a per-chunk `license` field would need a
decisions entry. Not changed here.

## Round 6 (2026-10-18): batch 2 and the circuit breaker
- Batch 2 ran the 17 remaining places through wikivoyage, wikipedia and web: 60 network requests, 188 cache hits, no 403/429, 0 redirect
  hops in this run (the live redirect path ran once elsewhere: a same-host 301 on developers.google.com, run c2141969, in the policy-page fetch; cross-host, path-prefix and userinfo refusals are fake-transport tests only, never exercised live), no TLS/DNS failure. The circuit breaker (D38) did not trigger.
- First contacts: `aptourism.gov.in` robots.txt and sitemap paths answered from the evidence-round cache (404, no sitemap): 0 network
  requests, 0 pages for Visakhapatnam and Srisailam. `hrce.tn.gov.in`: cached 404s again, 0 pages. `www.keralatourism.org`: robots and
  sitemap from cache; one page request (/districts/alappuzha/, HTTP 200, 410 KB) returned a client-rendered shell with too little text to
  keep, so Kerala Tourism has yielded 0 records in two batches. `tirumala.org`: not asked (Tirupati only). `sahapedia.org`: 2 pages,
  both for Kochi (revision dates only). `inditales.com`: 24 new posts (blog records 18 to 43).
- Official-source result: 0 `official` records after 26 places. The official sites we are allowed to ask do not list place pages in a
  sitemap or render without JavaScript. Permission replies from TN, Karnataka and ASI are the only remaining route (D32).

## Round 7 (2026-10-19): facts layer, Wikidata run (D36, D37)
- Three API hosts are allowlisted in config/sources.json -> api_hosts (www.wikidata.org, overpass-api.de, archive-api.open-meteo.com); terms evidence is in docs/facts-layer.md.
- Only Wikidata was run: 26 Special:EntityData requests plus one robots.txt request, all HTTP 200. wikidata.org/robots.txt (HTTP 200): Python's urllib.robotparser said False for Special:EntityData,
  but that was a parser artifact (it ignores `*`); the RFC 9309 matcher (D41) allows the URL by the Allow on line 436, so the D36 exception was not needed for it. No 403 or 429. CC0 data, no attribution duty; raw envelopes are local files (data/raw is gitignored).
- Overpass and Open-Meteo are built and tested offline but have made 0 requests.

## Round 8 (2026-10-19): facts layer, Open-Meteo and Overpass first calls
- Open-Meteo archive API, one request for Ooty: HTTP 200, no rate-limit headers, no stated call weight. The response is ERA5 data snapped to a grid cell about 2 km from the registry point.
- Overpass public instance, one airport query: timed out at the 30 s read timeout (server overloaded is the policy page's own warning). Not a block (no 403, 406 or 429). Nothing retried.
- Both are covered by D36/D37. Neither source has been run for any other place.

## Round 9 (2026-10-19): facts layer runs
- Open-Meteo archive: 8 of 26 places fetched (HTTP 200 each); the run stopped on a connect-phase timeout at the 8th request (Mysuru). No 429, no rate-limit header.
- Overpass: the single airport query succeeded with a 90 s read timeout (48 elements). Per-place station queries were removed (D40).

## Round 10 (2026-10-18): Open-Meteo for all 26 places
- 26 of 26 ERA5 archive envelopes (2015-2024, 3 variables, 3,653 days, 0 nulls), 18 network requests 360 s apart, HTTP 200 each, no 429, no rate-limit header, call weight still unknown.
- Envelopes stay in data/raw (gitignored); 26 manifest lines (url, sha256, source, place; no content).

## Round 11 (2026-10-19): never-seen-place demo (Yercaud)
- ensure_place resolved Yercaud through Wikidata (8 requests to www.wikidata.org, 4 SPARQL requests to query.wikidata.org, pageviews from cache) and fetched Wikivoyage and Wikipedia (1 request each); the web collector found no page.
  Everything was written to data/live/ only; a repeat call made 0 network requests. The resolver hosts are hard-coded and not in config/sources.json (D37 audit note); this was flagged before the run.
