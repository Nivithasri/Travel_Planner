# Phase 9 case studies (counts and descriptions only: no answer or snippet text, D100, O12)

Source: `.cache/phase9` records (gitignored) and `eval/phase9/results/*.json`; every number below is rebuilt by
`.venv/Scripts/python.exe scripts/phase9.py report`. Scores are the owner's human scores unless stated. Decisions: D123 to D130.

## Failure cases

**F1. Empty days (front-loading).** 12 of 43 days in the full system's corpus outputs and 21 of 43 in the live arm read "No
source-backed details for this day" (P04 has 4 of 5, P05, P09 and P10 have 2). Mechanism, reproduced offline with no LLM call (D129):
a stop gets evidence for all its nights at once, the writer rules say "only what the sources say" and "each fact once", and the
model puts all of a stop's facts on its first day. D129 raised the evidence per stop (3 per night, cap 6) and capped warnings at one,
which fixed the shortage of material but not the front-loading; the live arm is worse, probably because the live items are placed on
the first days too (not tested). Effect on scores: feasibility 2.3 and personalization 2.6 for the full system, below the baselines (not significant).
Fix, not done: assign evidence to days in the plan block and tell the writer to use only each day's tags.

**F2. P04 (pilgrimage, time-sensitive): no cited bullet left.** The full system's P04 output ended with 4 of 5 days empty and
no cited bullet at all; the critic status was repaired (two bullets on one day removed as uncited or carrying an unknown tag, the
no-details line written). It has the lowest total score of the 10 full-system outputs in the owner's sheet. The same request cited 2
live items in the live arm. Cause not fully traced: the writer produced mostly untagged or wrongly tagged bullets for this request.

**F3. A spurious "repaired" status.** The model writes the no-details line with a non-breaking hyphen (U+2011); the critic does not
recognise it, removes it and writes the canonical line, so the status is repaired instead of passed and the notes carry four lines of
noise. It inflates the repaired count in every table that uses the critic status. Fix, not done: normalise U+2011 and similar
characters in the model's output before the critic.

**F4. Live place mislabelling.** The live resolver gives an item the first registry name found in its text (D119 limit 3). In the live
arm 2 of 34 live items carry a label that disagrees with the place named in their URL (both in P03, labelled Chennai, URL about
Mahabalipuram); in the owner's hand-run 2 of 4 were wrong and a Mahabalipuram opening-hours fact landed on the Pondicherry day. Fix, not
done: resolve from the URL and title first, then by the most mentions.

**F5. Weak agreement where the full system loses.** Quadratic weighted kappa with the second scorer is 0.21 on feasibility and 0.33 on
personalization (0.72 groundedness, 0.80 freshness, pooled 0.60, n = 10). The rubric does not say how to score an empty day (it is
not a fact, so it counts under feasibility and personalization), so the loss on those two
dimensions is itself uncertain (the empty-day rule is a likely, untested, contributor). A clearer empty-day rule in eval/phase9/rubric.md is the cheapest improvement.

**F6. The hosted model is not fully deterministic.** With temperature 0 and a fixed seed, the cache-bypassed latency rerun gave
status repaired for P08 where the cached corpus run had passed. The corpus records are the reference; a rerun can differ.

## Cases where live evidence mattered

**L1. The main result: live search made the answers worse (owner scores, 10 requests, paired).** Live minus corpus: feasibility -0.5
(not significant), groundedness -0.9 (p_holm 0.047), freshness -1.9 (p_holm 0.031), personalization -0.2 (not significant); in 9 of 10
requests the live freshness score was not higher. Causes seen in the outputs and the owner's notes (not tested): the cannot-confirm
line is added by the code only when live fails (4 of 10 corpus outputs, 0 of 10 live outputs), so old wiki hours and prices are stated as
current with no caveat when live status is ok (P02, P07, P09); several live pages could not be opened (social-media reels, blogs), one was
a likely May 2025 article and some were booking aggregators (P01, P03, P04, P05, P08); the writer may say "as of <fetch date>" for a page
whose own date is older (P04, P05); mislabelled live items put facts on the wrong day (P03, P10); more empty days (21 of 43 against 12 of 43).
This is a decision-grade finding for the report: the live layer as built does not help, and the next step is the list in D130 amended.

**L2. P03 (honeymoon coast): the most live evidence, and the worst misplacement.** 4 live items cited, critic repaired, both
mislabelled live items belong to this request (F4); the owner's note says Mahabalipuram pages were used on a Chennai day and the live booking
claims could not be verified.

**L3. P04 and P08 (time-sensitive, named places): 2 live items cited each.** These are the requests where a current-status question could
in principle be answered from a dated source. The scores went down: the darshan hours and the closures rested on pages the owner could not
verify, and P08 scored 1 on freshness.

**L4. P10 (no named place): live status empty.** The live search is built from named places and states only (D100, D107), so a vague
request gets none and the live search had nothing to add. One of the 10 live requests was structurally unable to benefit, as the plan
predicted.

**L5. The scope check fired once.** The D120 check dropped 1 live item that named an excluded place (request P07), the case that failed
the owner's first live hand-run with an R5 finding.

## Not shown, by design
The plan's cases "where freshness or corrective retrieval changed the answer" do not exist: the pipeline has neither (D99, D128).
