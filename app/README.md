# South India Travel Planner: local demo (Phase 8)

One Streamlit page over the Phase 7 pipeline (`src/agents/`). It runs on your machine at
`127.0.0.1` only.

## Run

```bash
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

Needs, all local or in `.env` (the page never prints them):

- `index/` (embeddings, BM25, `place_cards.json`: run `scripts/build_place_cards.py` once),
  `data/enriched/`, `data/processed/` and the local bge-m3 model (`EMBED_LOCAL_DIR`). The first
  request loads the model (25 to 40 s); later requests reuse it.
- The hosted language model: `LLM_API_BASE_URL` (or `LLM_API_URL`), `LLM_API_KEY`, the model names
  and limits. The shared rate limiter applies; only one run executes at a time.
- Do NOT install `torchvision`: it is not needed (the app uses only the local text-embedding model)
  and must not be installed. Streamlit's file watcher used to probe every transformers module and
  print "No module named 'torchvision'" lines; `.streamlit/config.toml` now sets
  `server.fileWatcherType = "none"`, so those lines no longer appear (restart the app to pick up
  code changes).
- Optional, for live search: `LIVE_SEARCH_API_KEY`. Live search is a tick box, off by default; it
  spends search credits (assumed 2 per search, not measured).

## What the page does

- Form: duration, budget tier, travel style, start and end city (a destination or airport hub,
  checked against the routing table), dates, interests, constraints, places to include, daily
  travel limit, "Add live web search", "Ignore the cache for this run".
- The itinerary streams while the model writes it, then the checked text replaces the draft.
- Sources panel: corpus pages and live results with links; live rows read "live, fetched <date>";
  blog pages are listed by title and link only, never their text; Wikipedia and Wikivoyage rows
  carry the CC BY-SA attribution. Warnings are highlighted. A collapsible timing table lists every
  node.
- Labels: travel times are provisional; budget figures are a rough estimate, not verified;
  place-card items are auto-generated, check before relying. No corpus item is dated within the
  last 18 months; live items show their fetch date, not a publish date.
- Under the notices, one counts-only line reports the live step when "Add live web search" was
  ticked: `live search: ok, N searches, N results, N in scope, N in plan, N merged, N cited, N not cited`, or `empty` (same
  counts), `error: <short reason>`, `no key`, or `not used (<reason>)`. "results" are what the
  provider returned, "in scope" are those that name a registry place or state and passed the
  safety filter (after the cap of 8), "in plan" are those about a stop of the plan or a place you
  named (the others are left out before the writer, D119), "merged" are the live items put into
  the writer's prompt, "cited" are those the written itinerary cites and "not cited" the rest.
  Cited items and the others (flagged "retrieved, not cited") are rows of the sources table, with
  no snippet text. The line holds no text either and, like every live answer, stays in the session.
- Interests offered: temples, heritage, history, culture, beaches, hill stations, wildlife,
  backwaters, waterfalls, nature, food. They rank a stop's evidence (model-written chunks of the
  matching place type or aspect first, D117).
- Timing panel also says which writer ran: always the language model (a live call, or a replay of
  an earlier reply from the local cache); there is no template writer.
- Sources table: "Chunk id" shows the id exactly as the itinerary text prints it
  (`chunk:<id>` or `live:<id>`), so each citation in the text maps to a row.
- Timing panel: seconds per node and the tokens the hosted model reported per step (0 = served from
  the local LLM cache, nothing sent).
- When live search was not ticked the note reads "live search was not requested for this run".
- The trip start month alone drives the season check (a note appears when the dates span months).

## Cache (D111)

The key is sha256 of (query text, overrides, use_live).

- A corpus-only answer is written as JSON under `.cache/app/` (gitignored) and also kept in the
  session. A repeat is served from the cache and shows "cached answer"; "Ignore the cache" runs
  again; "Clear cache" empties it.
- A live-on answer is held in the browser session's memory only. It is never written to disk and
  never read from disk (nothing from live results is stored, D109), and a live-on request is never
  answered from a corpus-only entry. After a failed run the session's last good answer for the
  same inputs is shown with a notice.

## Live search notice and privacy (D100, D109)

Whenever the live step ran, the page shows: `live search: terms not verified, owner accepted
risk`. Live queries are built only from registry place names, state names, the month, the year and
fixed keywords. Free text (style, interests, constraints, other places) is never sent to the web
search; a place you type is passed on only if it resolves to a registry place. Do not enter
personal information: the free text is sent to the hosted language model that writes the
itinerary.

## When something is unavailable

- No internet or the language model is down: a red message (keys, hosts and tracebacks removed)
  and, if one exists, the last good answer.
- LLM quota used up: a plain message; try again after the daily reset.
- Live search skipped or failed (no key, monthly stop, terms gate, provider error): a visible
  note, and the corpus answer is shown anyway.
- Another run in progress: "another run is in progress"; submit again when it finishes.

## Files

`app/streamlit_app.py` (the page), `app/state.py` (form preparation, run lock, redaction, panel
rows), `app/query_cache.py` (the cache), `.streamlit/config.toml` (no error details, no usage
statistics, address 127.0.0.1). Tests: `tests/test_app_logic.py`, `tests/test_app_page.py`
(headless, with a fake pipeline).

## Hand-run checklist

HAND-RUN STEPS (2026-10-24 morning; nothing is running now; the LLM and Tavily caches/credits are yours to spend):
START: from C:\Users\valin\Travel_Planner run `.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py` (.streamlit/config.toml already sets fileWatcherType none, address 127.0.0.1, headless) and open http://127.0.0.1:8501. Stop it with Ctrl+C afterwards. Do not install torchvision.
FORM (same for every run): duration 4, budget mid, travelling as couple, style relaxed, start city Chennai, dates 2026-12-10 to 2026-12-13, max travel hours 6, interests temples and beaches (both are offered; the list is temples, heritage, history, culture, beaches, hill stations, wildlife, backwaters, waterfalls, nature, food), constraints "no night travel", places Mahabalipuram and Pondicherry. Leave "Add live web search" unticked for runs 1 to 3.
1. CORPUS-ONLY (fresh): click "Plan my trip". The first request takes 30 to 70 s (it loads the local model and calls the hosted model for the parse and the write; the cache was purged and the writer prompt changed, so both are real calls). Expect "fresh run", "corpus only: no live search was used for this answer", the note "live search was not requested for this run" (not "switched off"), NO "live search: ..." status line, no "only places in PY ..." note (a state limit is no longer inferred from place names), Backup options with other places (for example Chennai), correct dashes and phone symbols (no garbled characters), Mahabalipuram bullets that can now mention temples (the temples interest ranks matching chunks first; not guaranteed), a Sources table whose Chunk id column matches the [chunk:...] ids in the text, and in "Timing per node" the line "write step: language model, live call (N tokens)" with parse and write tokens above 0.
2. CACHED REPEAT: click again with nothing changed. Expect "cached answer (<time>)", an instant result, the same text.
3. FORCE REFRESH: tick "Ignore the cache for this run" and click. Expect "fresh run", a short run (a few seconds), and "write step: language model reply replayed from the local cache (0 tokens sent)" because the earlier reply is in the LLM cache. Untick it.
4. LIVE-ON: tick "Add live web search" and click. Expect about 6 to 10 s for the live step, two new /search lines in data/manifests/request_log.jsonl (448 now, 450 after), the notice "live search: terms not verified, owner accepted risk", and a status line "live search: ok, 2 searches, N results, N in scope, N in plan, N merged, N cited, N not cited" where in plan <= in scope, merged <= in plan and <= 4, and cited + not cited = merged. In scope counts results that name a registry place or state; in plan keeps only those about a stop of the plan or a place you named. Sources table: live rows read "live, fetched 2026-10-24" with a Chunk id live:...; a live item the model did not cite is listed with the flag "retrieved, not cited". The notes may say "N live search result(s) were left out because they are not about a place in your plan". The write step is always "language model, live call" here (live prompts are never cached) and costs tokens each time. If the status line says "0 in plan" or "0 merged", the search found nothing about Mahabalipuram or Pondicherry; if merged is above 0 and cited is 0, the model chose not to use them (rule 10 only asks).
