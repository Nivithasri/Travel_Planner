# Labelling decision rules (applied identically to dev and test)

Grade definitions are the rubric's. These are the tie-break rules I fixed for recurring cases.

1. Named place (place_scope named): a 2 needs a chunk about THAT place. Another place's chunk is 0, even if it is the right kind of content (e.g. another temple's timings for a Tirupati query = 0).
2. Free queries with a state or region in the text (Karnataka, Tamil Nadu, South India): a chunk about a place outside that state/region is 0, even if the theme fits.
3. Compound queries (e.g. "where to stay AND how to get around"): a chunk covering both halves = 2; a chunk covering one half with concrete detail = 1.
3b. Region-wide free queries with a hook (e.g. "free or low entry fee heritage sites in South India", "where to travel in South India"): a chunk about one place in the region that states the hook (a fee, a ticket price, "free") = 2. A destination chunk with no hook is 1. If the hook is only a time word, it is ignored, and when the rest of the query is a bare "where to go" (Q25) I gave no 2s: nothing both recommends a place and says why it fits. Accommodation, food and transport lists are 0 for these.
4. Suitability/audience queries (toddlers, family with kids, honeymoon): 2 needs audience-specific content plus enough to judge. A single attraction or generic logistics from the right place = 1. Nothing audience-specific is 1 at most.
5. Descriptive chunk about the right place or temple that lacks the asked-for fact (hours, safety, etc.) = 1 (rubric's Sampleville example). City demographics, politics, etymology, climate tables or history with no tie to the query = 0.
6. Accommodation-only listings: 1 only if the query is about stay/budget and the price tier matches; otherwise 0.
7. Intra-city transport (autos, city buses, suburban trains) for an intercity-itinerary query = 0. Intercity train/bus connectivity = 1 unless it forms a route with costs.
8. Safety queries: explicit statement that swimming/bathing is unsafe, banned or dangerous = 2. Implicit or comparative hints, or incident/lifeguard mentions without an explicit warning = 1.
9. Time words ("now", "latest") ignored. Excluded-place chunks graded like any other.
10. Torn 1 vs 2 -> 1. Torn 0 vs 1 -> 1 only if the chunk adds at least one usable on-topic fact; otherwise 0.
11. Foreign-script or poem/narrative text with no usable fact = 0.

12. Other attractions in the right city: another temple's timings or fees for a named-temple query = 0 (rule 1); another palace/museum in the same city for a palace query = 1 (Q36 rows 581 vs 588).
13. Queries asking for a current status (closures, flooding, ferry running) where no chunk states it: the nearest hazard/closure chunk for the right place gets at most 2 only if it states the hazard directly (Q38 row 604); otherwise 1. Say so in `note`.

Self-consistency check (dev): 8 early rows re-read and re-graded blind. 7 of 8 matched; row 4 moved 2 -> 1 (see note in the CSV).

---
Provenance (added by the repo, not part of the owner's notes): copied unchanged from the owner's labelling_rules_notes.md on 2026-10-20. These rules are the owner's; see D75 in docs/decisions.md for what is and is not known about when they were written and who saw them.
