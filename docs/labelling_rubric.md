# Labelling rubric (Phase 4 relevance judgments)

You judge ONE chunk against ONE query at a time. Judge what the chunk text says, not the place's reputation
and not how you think a search engine would rank it. You are not told which index returned a chunk; the rows
are shuffled. The chunk's `place` and `title` columns tell you where the text comes from.

Excluded-place chunks (the nine D33 places) are judged like any other chunk. If a chunk about such a place answers
the query, it gets the grade it deserves.

| grade | meaning |
|---|---|
| 2 | Directly answers the query: the chunk gives the information the traveller asked for, for the place or kind of place the query asks about. |
| 1 | On topic but partial or tangential: right place or right theme, but it answers only part of the query, is general background, or is about a neighbouring aspect. |
| 0 | Not relevant: wrong place, wrong theme, or only a passing mention. |

Rules of thumb
- A query that names a place (`place_scope` named) needs a chunk about THAT place for a 2. A chunk about another
  place, however good, is at most 1 (only if it still helps the same decision) and usually 0.
- A query that names no place (`free`) can be answered by any in-scope place; judge the match to the theme.
- Time words ("now", "latest", "this year") do not make an old chunk less relevant here: judge whether the text
  addresses the question. Freshness is measured separately.
- If you cannot tell, choose 1, and write why in `note`.
- Do not label from memory of the real world; use only the text.

The examples below are invented toy text about a made-up place, "Sampleville". They are not corpus text.

## Grade 2 (directly answers)

1. Query: "Sampleville temple visiting hours and dress code".
   Chunk: "The Sampleville Shiva temple opens at 5 a.m. and closes at 9 p.m., with a break from 12 to 4 p.m.
   Visitors must cover their shoulders and knees; shorts are not allowed inside the sanctum."
   Grade 2: it gives both hours and dress code for the named temple.
2. Query: "Family-friendly beach for toddlers in South India".
   Chunk: "Sample Bay has calm, shallow water and a lifeguard post. Families bring toddlers because the sand slopes
   gently and there are shaded benches and a small play area."
   Grade 2: it describes a beach and says why it suits toddlers.
3. Query: "Budget two-week itinerary by train and bus".
   Chunk: "A cheap route runs Sampleville, then Sample Hills by bus (6 hours, about 300 rupees), then Sample Port
   by overnight train in sleeper class (about 450 rupees). Budget rooms cost 600 to 900 rupees a night."
   Grade 2: a route with transport modes and costs, which is the content an itinerary needs.

## Grade 1 (on topic, partial or tangential)

1. Query: "Sampleville temple visiting hours and dress code".
   Chunk: "The Sampleville Shiva temple was built in the 12th century and is known for its tall gateway towers
   and carved pillars." Grade 1: right temple, but no hours and no dress code.
2. Query: "Family-friendly beach for toddlers in South India".
   Chunk: "Sample Bay is a popular beach with fishing boats and seafood stalls; the water is rough in the monsoon."
   Grade 1: a beach, but nothing on toddlers and a hint of a safety issue only in one season.
3. Query: "Budget two-week itinerary by train and bus".
   Chunk: "Sample Port is the region's main railway junction, with trains to most large cities."
   Grade 1: useful transport background, but no itinerary and no costs.

## Grade 0 (not relevant)

1. Query: "Sampleville temple visiting hours and dress code".
   Chunk: "Sampleville had a population of 640,000 at the last census, and its main industry is textile
   weaving." Grade 0: the right place, but nothing about the temple or visiting.
2. Query: "Family-friendly beach for toddlers in South India".
   Chunk: "Sample Hills is a cool hill town with tea gardens and a toy train." Grade 0: no beach at all.
3. Query: "Budget two-week itinerary by train and bus".
   Chunk: "A luxury resort near Sample Port offers private pool villas starting at 25,000 rupees a night."
   Grade 0: it is about travel, but the opposite of budget, and gives no route.
