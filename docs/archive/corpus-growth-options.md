# Corpus growth options (PROPOSAL ONLY, nothing implemented or run)

Status 2026-10-18. The planned collection projects to about 1,600 to 2,400 chunks for 26 places (D26), against the plan's
10,000 to 20,000. These are the in-policy ways to add text. Every gain is an ESTIMATE; the measured inputs and the
assumptions are stated so the owner can check them.

## Measured inputs (3 slice places, strict estimate, scripts/estimate_corpus_size.py)
- Wikipedia place page: about 26.9k characters (about 32 chunks at 250 tokens). Wikivoyage page: about 27.1k (about 32).
- inditales: 3 posts, about 8.5k characters each (about 10 chunks per post); found for 2 of 3 places.
- One place today: about 74 chunks (62.5k characters). One chunk is about 850 new characters (250 tokens, 15% overlap).
- To reach 10,000 chunks for 26 places a place needs about 330k characters, about 5 times today.

## Options
| # | Option | Expected gain (estimate) | Source-type effect | Freshness effect | Policy and cost |
|---|---|---|---|---|---|
| A | Follow links from each place's Wikipedia page to sight, temple, park and district articles, cap 8 per place | Assumption: 3 to 10k chars per article, so 4 to 12 chunks each: +32 to +96 per place, +830 to +2,500 for 26 (about +40% to +130%) | None new: still `reference` | None. Revision date only (O9); a reference page is not "fresh" | Same Wikimedia API (CC BY-SA), about 1 link request + 8 article requests per place, about 230 requests for 26. Raises O8 risk (district articles name neighbours, possibly hold-outs) |
| B | Wikivoyage district / sub-articles for big cities (Chennai, Hyderabad, Bengaluru, Kochi...) | Assumption: 3 to 8 sub-articles of 5 to 15k chars for about 6 hubs: +20 to +120 per hub, +120 to +700 total | None new: `wikivoyage` | None (revision date) | Same API, about 5 requests per hub; only hubs have such pages |
| C | More inditales posts through broader matching (post slugs by alias, state, region; today only the place name in the URL matches) | Assumption: 0.5 to 2 more posts per place: +5 to +20 per place, +130 to +520 | None new: `blog`, but it is the only traveler-voice source we have, so the diversity value is high | Posts carry a real publish date (2004 to 2024 in the slice): usable for freshness, but many are old | Same allowlist (owner accepted the risk). Lower precision: needs a relevance check (Phase 2 / O8) |
| D | Reddit, once credentials are approved (collector is built, offline-tested, skipped now) | Assumption: 30 to 100 records per place of 300 to 1,500 chars: +15 to +60 per place, +400 to +1,500 | New type `reddit`: largest diversity gain | Best: posts and comments have publish dates, recent | Blocked on approval. Reddit API terms need review before the corpus is shared |
| E | Official pages from owner-verified seed URLs (D25), 3 to 5 per place | Assumption: 5 to 20k chars per page: +15 to +100 per place if provided | New type `official` (together with Sahapedia, `reference`) | Mostly undated or revision date only | Manual work: about 80 to 130 URLs verified by hand; never generated from model memory; excluded from the never-seen-place demo |
| F | YouTube Data API descriptions and comments (O10) | Not estimated | would add `youtube` | Dated | NOT enabled; policy and terms not reviewed; listed only |

## Combined picture
A + C + D together add about +1,400 to +4,500 chunks to the 1,600 to 2,400 baseline, so about 3,000 to 6,900 in total. Even with every
in-policy option the 10,000 floor is not reached at this collection scale. The remaining levers are all the owner's decisions:
revise the target (a test collection with 3,000 to 7,000 chunks over 26 places is workable for the IR evaluation but is not the
plan's number), widen the place list, or raise per-place caps. None is done here.

## What does not help freshness
Options A and B add `reference` and `wikivoyage` text only. By O9 those carry a revision date, never a publish date, so
they must not count as fresh in the freshness share. Only C (dated posts) and D (Reddit) add content that can be recent.
Time-sensitive queries (closures, timings) therefore depend on D and E, both blocked on external approval or manual work.
