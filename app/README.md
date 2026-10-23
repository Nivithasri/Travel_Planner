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
