"""South India Travel Planner: one local Streamlit page over the Phase 7 pipeline (D110).

    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

Corpus retrieval and the itinerary need no internet beyond the hosted language model; the live
search is an optional tick box (off by default) and the page works from the corpus without it.
Everything shown goes through app.state.redact; keys, API hosts and .env values are never shown.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st  # noqa: E402

from app import state as S  # noqa: E402
from app.query_cache import QueryCache  # noqa: E402
from src.live.config import TERMS_NOTICE  # noqa: E402

st.set_page_config(page_title="South India Travel Planner", layout="wide")
ss = st.session_state


@st.cache_resource
def registry():
    return S.load_registry()


def place_options() -> list[str]:
    return sorted(e.name for e in registry().entries if not e.holdout)


def days_range() -> tuple[int, int]:
    from src.agents.state import load_planner_config

    cfg = load_planner_config()
    return int(cfg["min_days"]), int(cfg["max_days"])


def show(text: str) -> str:
    return S.redact(text)


# ---- header ------------------------------------------------------------------------------------

st.title("South India Travel Planner")
st.caption("Answers come from a frozen local corpus (Wikipedia, Wikivoyage and a travel blog). "
           "An optional live web search adds current snippets. Tamil Nadu, Kerala, Karnataka, "
           "Andhra Pradesh, Telangana and Puducherry only.")

# ---- form --------------------------------------------------------------------------------------

excluded = S.load_excluded()
lo, hi = days_range()
avail = S.live_availability()

with st.form("plan"):
    c1, c2, c3 = st.columns(3)
    days = c1.number_input("Duration (days)", min_value=lo, max_value=hi, value=4, key="f_days")
    budget = c2.selectbox("Budget tier", S.BUDGET_TIERS, index=1, key="f_budget")
    party = c3.selectbox("Travelling as", S.PARTIES, index=3, key="f_party")
    style = st.text_input("Style (for example relaxed, adventurous, family friendly)",
                          key="f_style", max_chars=S.MAX_FREE_CHARS)
    c4, c5 = st.columns(2)
    start_city = c4.text_input("Start city (a destination or airport hub)", key="f_start")
    end_city = c5.text_input("End city (optional)", key="f_end")
    c6, c7, c8 = st.columns(3)
    start_date = c6.date_input("Start date", value=None, key="f_start_date")
    end_date = c7.date_input("End date", value=None, key="f_end_date")
    max_hours = c8.number_input("Max travel hours per day (0 = default)", min_value=0.0,
                                max_value=14.0, value=0.0, step=0.5, key="f_hours")
    interests = st.multiselect("Interests", S.INTEREST_CHOICES, key="f_interests")
    interests_free = st.text_input("Other interests (comma separated)", key="f_interests_free",
                                   max_chars=S.MAX_FREE_CHARS)
    constraints = st.text_input("Constraints (for example avoid hill roads, no night travel)",
                                key="f_constraints", max_chars=S.MAX_FREE_CHARS)
    named = st.multiselect(
        "Places you want to include", place_options(), key="f_places",
        format_func=lambda n: f"{n} (limited evidence)" if n in excluded else n)
    named_free = st.text_input("Other places (registry names only; anything else is ignored)",
                               key="f_places_free", max_chars=S.MAX_FREE_CHARS)
    st.caption(S.PRIVACY_HINT.capitalize() + ". Free text is used to write the itinerary; it "
               "is never sent to the web search.")
    use_live = st.checkbox("Add live web search", value=False, key="f_live",
                           help=S.LIVE_CREDITS_HINT)
    st.caption(S.LIVE_CREDITS_HINT)
    if not avail["key_present"]:
        st.caption("No live-search key is set in .env: a live run would be skipped.")
    force = st.checkbox("Ignore the cache for this run", value=False, key="f_force")
    submitted = st.form_submit_button("Plan my trip")

cache = QueryCache(memory=ss.setdefault("memo", {}))
cols = st.columns([1, 6])
if cols[0].button("Clear cache"):
    n = cache.clear()
    cols[1].caption(f"cache cleared ({n} file(s) removed)")

# ---- submit ------------------------------------------------------------------------------------

if submitted:
    form = S.PlanForm(
        duration_days=int(days), budget_tier=budget, party=party, travel_style=style,
        start_city=start_city, end_city=end_city, start_date=start_date, end_date=end_date,
        interests=list(interests), interests_free=interests_free, constraints=constraints,
        named_places=list(named), named_free=named_free,
        max_daily_travel_hours=float(max_hours) or None, use_live=bool(use_live),
        force_refresh=bool(force))
    prep = S.prepare(form, registry())
    ss["notes"] = prep.notes
    ss.pop("error", None)
    if prep.errors:
        for msg in prep.errors:
            st.error(show(msg))
    else:
        hit = None if (form.use_live or form.force_refresh) else cache.get(
            prep.text, prep.overrides, False)
        if hit is not None:
            ss["view"] = hit
            ss["view_meta"] = {"state": "cached", "at": hit.get("cached_at", "")}
            ss.pop("handle", None)
        else:
            try:
                deps = S.get_deps(form.use_live)
                ss["handle"] = S.start_run(deps, prep.text, prep.overrides)
                ss["ctx"] = (prep.text, prep.overrides, form.use_live)
            except S.RunBusy:
                st.warning(S.BUSY_MESSAGE)
            except Exception as exc:  # noqa: BLE001 - shown redacted, no traceback
                ss["error"] = S.safe_error(exc)

# ---- a run in progress (also picked up again after a page rerun) ---------------------------------

handle = ss.get("handle")
if handle is not None:
    st.info("Planning your trip. The first request loads the local search model, which can "
            "take a minute.")
    live_box = st.empty()
    buf = ""
    while True:
        finished = handle.done.is_set()
        buf += handle.drain()
        if buf:
            live_box.markdown(show(buf))
        if finished:
            break
        handle.done.wait(0.15)
    live_box.empty()
    text, overrides, was_live = ss.pop("ctx", ("", {}, False))
    ss.pop("handle", None)
    if handle.state is not None:
        view = S.view_from_state(handle.state, handle.wall_s)
        cache.put(text, overrides, was_live, view)
        ss["view"] = view
        ss["view_meta"] = {"state": "fresh", "at": ""}
    else:
        ss["error"] = handle.error or "the run failed"
        last = cache.get(text, overrides, was_live)
        if last is not None:
            ss["view"] = last
            ss["view_meta"] = {"state": "last good", "at": last.get("cached_at", "")}

if ss.get("error"):
    st.error(show(f"The run did not finish: {ss['error']}"))

for note in ss.get("notes", []):
    st.info(show(note))

# ---- result ------------------------------------------------------------------------------------

view = ss.get("view")
if view:
    view = S.redact_deep(view)
    final = view.get("final") or {}
    meta = ss.get("view_meta", {"state": "fresh", "at": ""})
    if meta["state"] == "cached":
        st.caption(f"cached answer ({meta['at']})")
    elif meta["state"] == "last good":
        st.warning(f"showing the last good answer for these inputs ({meta['at']})")
    else:
        st.caption("fresh run")

    if S.terms_notice_visible(view):
        st.warning(TERMS_NOTICE)
        if view.get("live_status") != "ok":
            st.warning(view.get("live_note") or "live search returned nothing usable; "
                       "answers use the corpus only")
    elif view.get("live_status") in ("no_key", "quota", "terms_not_verified"):
        st.warning(view.get("live_note") or "live search unavailable; answers use the corpus only")
    else:
        st.caption("corpus only: no live search was used for this answer")
    st.caption(S.RECENCY_NOTE)

    critic = final.get("critic") or {}
    if view.get("repaired"):
        st.info("The rule checker repaired the draft once; the text below is the repaired one.")
    if critic.get("status") == "failed_rules":
        st.error("The rule checker still reports problems:\n\n" + "\n".join(
            f"- {f.get('rule')}: {f.get('message')}" for f in critic.get("findings", [])))

    st.header("Itinerary")
    st.markdown(final.get("text", ""))
    st.caption(f"Travel times are {S.TRAVEL_LABEL}. Budget figures are a {S.BUDGET_LABEL}. "
               f"Place-card items are {S.CARD_LABEL}.")

    plan = final.get("plan") or {}
    cards = S.load_cards()
    if plan.get("stops"):
        st.subheader("Plan")
        st.dataframe([{k: s.get(k) for k in ("place", "state", "nights", "evidence_class",
                                             "season_verdict", "access_risk")}
                      for s in plan["stops"]], hide_index=True)
        legs = S.leg_rows(plan)
        if legs:
            st.markdown(f"**Legs** (hours are {S.TRAVEL_LABEL})")
            st.dataframe(legs, hide_index=True)
        budget_info = plan.get("budget") or {}
        if budget_info.get("total"):
            st.markdown(f"**Budget:** about INR {budget_info['total']:,} in total "
                        f"({budget_info.get('tier')} tier) - {S.BUDGET_LABEL}")
        with st.expander(f"Place cards ({S.CARD_LABEL})"):
            for stop in plan["stops"]:
                card = cards.get(stop["place"]) or {}
                items = [*(card.get("warnings") or [])[:2], *(card.get("tips") or [])[:2]]
                if not items:
                    continue
                st.markdown(f"**{stop['place']}**")
                for it in items:
                    kind = "warning" if it in (card.get("warnings") or []) else "tip"
                    st.markdown(f"- {kind}: {str(it.get('text', ''))[:300]} "
                                f"([source]({it.get('url', '')}))")

    warnings = list(dict.fromkeys([*(final.get("warnings") or []), *(view.get("warnings") or [])]))
    if warnings:
        st.subheader("Warnings")
        for w in warnings:
            st.warning(w)

    rows = S.source_rows(final, S.load_attribution(), S.card_warning_chunk_ids(cards))
    with st.expander(f"Sources ({len(rows)})", expanded=True):
        if rows:
            st.dataframe(rows, hide_index=True, column_config={
                "Link": st.column_config.LinkColumn("Link")})
        else:
            st.caption("no sources were cited")
        st.caption(S.CC_BY_SA + ". Blog pages are listed by title and link only.")

    t_rows, total = S.timing_rows(view.get("timings") or {})
    with st.expander("Timing per node", expanded=False):
        st.dataframe(t_rows, hide_index=True)
        st.caption(f"sum of node times {total} s; wall clock {view.get('wall_s', 0)} s "
                   "(retrieve and live run in parallel)")

st.divider()
st.caption(S.footer_attribution())
