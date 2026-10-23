"""The Streamlit page (app/streamlit_app.py) run headless with streamlit.testing.v1.AppTest and a
fake pipeline: layout, cache behaviour, live notice, degraded modes, redaction, the run lock. No
browser, no network, no LLM, no real key; the pipeline runner and Deps are fakes."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import query_cache as QC  # noqa: E402
from app import state as S  # noqa: E402
from src.agents.state import RunState  # noqa: E402
from src.common.schemas import REPO_ROOT  # noqa: E402
from src.live.config import TERMS_NOTICE  # noqa: E402
from tests.test_graph import make_deps  # noqa: E402

APP = str(REPO_ROOT / "app" / "streamlit_app.py")
WIKI = "https://en.wikivoyage.org/wiki/Hampi"
ATTR = {WIKI: {"title": "Hampi - Wikivoyage", "license": "CC BY-SA", "source": "wikivoyage"},
        "https://inditales.test/hampi": {"title": "Hampi travelogue", "source": "blog",
                                         "license": "no terms"}}
CARDS = {"Hampi": {"warnings": [{"text": "The monolith area closes at dusk.", "chunk_id": "abc",
                                 "url": WIKI}],
                   "tips": [{"text": "Rent a bicycle.", "chunk_id": "zzz", "url": WIKI}]}}


def make_state(*, live=False, text=None, status=None, critic="passed", repaired=False):
    sources = [{"alias": "S1", "chunk_id": "abc", "place": "Hampi", "source_type": "wikivoyage",
                "url": WIKI, "live": False, "fetch_date": None, "date": None,
                "provenance": "llm", "limited": False},
               {"alias": "S2", "chunk_id": "blog1", "place": "Hampi", "source_type": "blog",
                "url": "https://inditales.test/hampi", "live": False, "fetch_date": None,
                "date": "2024-11-21", "provenance": "rule_only", "limited": False}]
    if live:
        sources.append({"alias": "L1", "chunk_id": "live:1", "place": "Hampi",
                        "source_type": "live", "url": "https://a.test/1", "live": True,
                        "fetch_date": "2026-10-23", "date": None, "provenance": "live",
                        "limited": False})
    plan = {"feasible": True, "stops": [{"place": "Hampi", "state": "KA", "nights": 2,
                                         "evidence_class": "full", "season_verdict": "good",
                                         "access_risk": "low"}],
            "legs": [{"origin": "Bengaluru", "destination": "Hampi", "hours": 6.5, "days": 1,
                      "kind": "arrival"}],
            "budget": {"total": 24000, "tier": "mid"}}
    findings = [{"rule": "R4", "message": "a bullet has no citation"}] if critic != "passed" else []
    final = {"text": text or "# South India itinerary\n\nDay 1 Hampi ruins.\n", "sources": sources,
             "plan": plan, "critic": {"status": critic, "findings": findings},
             "live_status": status or ("ok" if live else "disabled"),
             "warnings": ["thin evidence for Hampi"]}
    return RunState(final=final, timings={"parse": 0.5, "write": 2.0},
                    tokens={"parse": 844, "write": 1171, "critic": 0},
                    live_status=status or ("ok" if live else "disabled"),
                    live_note=TERMS_NOTICE if live else None, repaired=repaired)


class Fake:
    def __init__(self):
        self.calls = []
        self.next = lambda use_live: make_state(live=use_live)

    def runner(self, deps, text, overrides):
        self.calls.append({"text": text, "overrides": dict(overrides), "live": deps.live})
        deps.on_piece("Day 1 [S1] ")
        deps.on_piece("Hampi.")
        out = self.next(deps.live)
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = Fake()
    monkeypatch.setattr(QC, "CACHE_DIR", tmp_path / "appcache")
    monkeypatch.setattr(S, "get_deps",
                        lambda use_live, factory=None: SimpleNamespace(live=use_live,
                                                                       on_piece=None))
    monkeypatch.setattr(S, "default_runner", lambda: f.runner)
    monkeypatch.setattr(S, "load_cards", lambda: CARDS)
    monkeypatch.setattr(S, "load_attribution", lambda: ATTR)
    monkeypatch.setattr(S, "live_availability",
                        lambda: {"key_present": True, "enabled": True,
                                 "terms_status": "owner_accepted_risk"})
    f.dir = tmp_path / "appcache"
    yield f
    assert not S.run_lock_held()


def app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def submit(at, **widgets):
    for key, value in widgets.items():
        kind, name = key.split("__")
        getattr(at, kind)(key=name).set_value(value)
    next(b for b in at.button if b.label == "Plan my trip").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at):
    out = []
    for group in (at.markdown, at.caption, at.warning, at.error, at.info, at.header,
                  at.subheader, at.title):
        out += [str(e.value) for e in group]
    return out


def blob(at):
    return "\n".join(texts(at))


def files(fake):
    return sorted(fake.dir.glob("*.json")) if fake.dir.exists() else []


# ---- the form --------------------------------------------------------------------------------

def test_the_form_has_every_field_and_live_search_is_off_by_default(fake):
    at = app()
    assert at.number_input(key="f_days").value == 4
    assert at.checkbox(key="f_live").value is False
    for key in ("f_budget", "f_party"):
        assert at.selectbox(key=key)
    for key in ("f_style", "f_start", "f_end", "f_interests_free", "f_constraints",
                "f_places_free"):
        assert at.text_input(key=key)
    assert at.multiselect(key="f_interests") and at.multiselect(key="f_places")
    assert at.date_input(key="f_start_date") and at.date_input(key="f_end_date")
    page = blob(at)
    assert S.LIVE_CREDITS_HINT in page and S.PRIVACY_HINT in page.casefold()
    assert S.footer_attribution() in page
    assert fake.calls == []


def test_the_credit_hint_and_missing_key_note(fake, monkeypatch):
    monkeypatch.setattr(S, "live_availability",
                        lambda: {"key_present": False, "enabled": True, "terms_status": "x"})
    assert "No live-search key is set" in blob(app())


# ---- a corpus-only run -------------------------------------------------------------------------

def test_a_corpus_only_run_shows_the_itinerary_sources_and_labels_and_caches_to_disk(fake):
    at = submit(app(), **{"text_input__f_start": "Chennai"})
    page = blob(at)
    assert "Day 1 Hampi ruins." in page and "fresh run" in page
    assert "corpus only: no live search was used" in page
    assert TERMS_NOTICE not in page
    assert S.RECENCY_NOTE in page
    assert S.BUDGET_LABEL in page and S.TRAVEL_LABEL in page and S.CARD_LABEL in page
    assert len(fake.calls) == 1 and fake.calls[0]["live"] is False
    assert fake.calls[0]["overrides"]["start_city"] == "Chennai"
    assert len(files(fake)) == 1
    sources = next(d for d in at.dataframe if "Licence" in d.value.columns).value
    assert list(sources["Alias"]) == ["S1", "S2"]
    assert sources.loc[0, "Link"] == WIKI and "CC BY-SA" in sources.loc[0, "Licence"]
    assert "CC BY-SA" not in sources.loc[1, "Licence"] and sources.loc[1, "Title"] == (
        "Hampi travelogue")
    assert "card warning" in sources.loc[0, "Flags"]
    assert tuple(sources.columns) == S.SOURCE_FIELDS
    assert "recent" not in page.casefold().replace("last good", "")
    assert sources.loc[0, "Chunk id"] == "chunk:abc" and sources.loc[1, "Chunk id"] == (
        "chunk:blog1")
    assert "Planning your trip" not in page  # the progress banner is gone when the run ends


def test_the_timing_breakdown_is_collapsed_and_complete(fake):
    at = submit(app())
    exp = next(e for e in at.expander if e.label == "Timing per node")
    assert exp.proto.expanded is False
    assert "sum of node times 2.5 s" in blob(at)
    steps = next(d for d in at.dataframe if "Step" in d.value.columns).value
    assert dict(zip(steps["Step"], steps["Value"])) == {
        "parse (tokens)": 844, "write (tokens)": 1171, "repair (tokens)": 0}
    assert "0 means the step was served from the local LLM cache" in blob(at)
    assert next(e for e in at.expander if e.label.startswith("Sources")).proto.expanded is True


def test_a_repeat_is_served_from_the_cache_and_force_refresh_runs_again(fake):
    at = submit(app())
    at = submit(at)
    assert len(fake.calls) == 1 and "cached answer" in blob(at)
    at = submit(at, **{"checkbox__f_force": True})
    assert len(fake.calls) == 2 and "fresh run" in blob(at)
    assert len(files(fake)) == 1  # the same key was rewritten


def test_clear_cache_empties_the_directory(fake):
    at = submit(app())
    assert len(files(fake)) == 1
    next(b for b in at.button if b.label == "Clear cache").click()
    at.run()
    assert files(fake) == [] and "cache cleared" in blob(at)


# ---- a live-on run -----------------------------------------------------------------------------

def test_a_live_on_run_always_runs_shows_the_notice_and_writes_nothing(fake):
    at = submit(app())  # a corpus-only answer is cached first
    assert len(files(fake)) == 1
    at = submit(at, **{"checkbox__f_live": True})
    assert len(fake.calls) == 2 and fake.calls[1]["live"] is True  # not served from the cache
    page = blob(at)
    assert TERMS_NOTICE in page and "corpus only: no live search" not in page
    assert len(files(fake)) == 1  # nothing new on disk
    sources = next(d for d in at.dataframe if "Licence" in d.value.columns).value
    live_row = sources[sources["Source"].str.startswith("live")].iloc[0]
    assert live_row["Source"] == "live, fetched 2026-10-23"
    assert live_row["Fetch date"] == "2026-10-23" and "recent" not in str(live_row).lower()
    at = submit(at)  # the same inputs again with the tick box still on: still a fresh run
    assert len(fake.calls) == 3
    on_disk = " ".join(p.read_text(encoding="utf-8") for p in files(fake))
    assert "live:1" not in on_disk and "a.test" not in on_disk


def test_a_failed_live_step_still_shows_the_corpus_answer_and_the_notice(fake):
    fake.next = lambda use_live: make_state(live=False, status="error")
    at = submit(app(), **{"checkbox__f_live": True})
    page = blob(at)
    assert "Day 1 Hampi ruins." in page and TERMS_NOTICE in page
    assert "live search" in page and files(fake) == []


@pytest.mark.parametrize("status", ["no_key", "quota", "terms_not_verified"])
def test_a_skipped_live_step_says_so_without_the_terms_notice(fake, status):
    fake.next = lambda use_live: make_state(live=False, status=status)
    page = blob(submit(app(), **{"checkbox__f_live": True}))
    assert "Day 1 Hampi ruins." in page and TERMS_NOTICE not in page
    assert "live search" in page.casefold()


# ---- inputs and notes --------------------------------------------------------------------------

def test_an_unknown_start_city_is_a_clear_error_and_nothing_runs(fake):
    at = submit(app(), **{"text_input__f_start": "Atlantis"})
    assert any("Atlantis" in str(e.value) and "routing table" in str(e.value) for e in at.error)
    assert fake.calls == []


def test_a_trip_across_months_is_noted(fake):
    import datetime

    at = submit(app(), **{"date_input__f_start_date": datetime.date(2026, 12, 28),
                          "date_input__f_end_date": datetime.date(2027, 1, 3)})
    assert S.SEASON_NOTE in blob(at) and fake.calls[0]["overrides"]["month"] == 12


def test_free_text_places_reach_the_pipeline_only_when_they_are_registry_names(fake):
    submit(app(), **{"text_input__f_places_free": "Zzqxsentinelplace, Hampi",
                     "text_input__f_style": "zzqxsentinelstyle"})
    assert fake.calls[0]["overrides"]["named_places"] == ["Hampi"]


# ---- failures and degraded modes ---------------------------------------------------------------

def test_a_pipeline_error_is_shown_redacted_without_a_traceback(fake):
    fake.next = lambda use_live: RuntimeError(
        "boom at https://api.tavily.com/search with Bearer abcdef123456 and gsk_ABCDEF123456")
    at = submit(app())
    page = blob(at)
    assert "The run did not finish" in page
    for bad in ("api.tavily.com", "abcdef123456", "gsk_ABCDEF123456", "Traceback"):
        assert bad not in page


def test_a_quota_stop_gets_a_plain_message(fake):
    from src.common.llm import QuotaExhausted

    fake.next = lambda use_live: QuotaExhausted("daily")
    assert "quota is used up" in blob(submit(app()))


def test_after_a_failure_the_last_good_answer_is_shown(fake):
    at = submit(app())
    fake.next = lambda use_live: RuntimeError("provider down")
    at = submit(at, **{"checkbox__f_force": True})
    page = blob(at)
    assert "The run did not finish" in page and "last good answer" in page
    assert "Day 1 Hampi ruins." in page


def test_a_run_started_elsewhere_blocks_the_form_with_a_message(fake):
    assert S._RUN_LOCK.acquire(blocking=False)
    try:
        at = submit(app())
    finally:
        S._RUN_LOCK.release()
    assert S.BUSY_MESSAGE in blob(at) and fake.calls == []


def test_secrets_and_api_hosts_never_reach_the_page(fake, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key-value-1234567890")
    leaky = ("# Trip\n\nsee https://en.wikipedia.org/wiki/Hampi key unit-test-key-value-1234567890 "
             "host https://api.tavily.com/search gsk_ABCDEF123456\n")
    fake.next = lambda use_live: make_state(text=leaky)
    at = submit(app())
    page = blob(at) + json.dumps([d.value.to_dict() for d in at.dataframe], default=str)
    assert "https://en.wikipedia.org/wiki/Hampi" in page
    for bad in ("unit-test-key-value", "api.tavily.com", "gsk_ABCDEF123456"):
        assert bad not in page


def test_critic_failure_and_repair_are_visible(fake):
    fake.next = lambda use_live: make_state(critic="failed_rules")
    page = blob(submit(app()))
    assert "still reports problems" in page and "R4" in page
    fake.next = lambda use_live: make_state(critic="repaired", repaired=True)
    at = submit(app(), **{"checkbox__f_force": True})
    assert "repaired the draft once" in blob(at)


# ---- the real pipeline runner behind the page -----------------------------------------------

def real_page(tmp_path, monkeypatch, deps):
    """The page on the real pipeline (default runner) with the given fake dependencies."""
    monkeypatch.setattr(QC, "CACHE_DIR", tmp_path / "appcache")
    monkeypatch.setattr(S, "get_deps", lambda use_live, factory=None: deps)
    monkeypatch.setattr(S, "load_cards", lambda: CARDS)
    monkeypatch.setattr(S, "load_attribution", lambda: ATTR)
    monkeypatch.setattr(S, "live_availability",
                        lambda: {"key_present": True, "enabled": True, "terms_status": "x"})


@pytest.mark.parametrize("langgraph_available", [True, False])
def test_the_page_drains_the_streamed_pieces_of_the_real_pipeline(
        tmp_path, monkeypatch, langgraph_available):
    """No fake runner: the default runner (run_graph, or run_local without langgraph) runs the
    real nodes on fake dependencies in the worker thread; the page's loop receives the pieces."""
    real_page(tmp_path, monkeypatch, make_deps())
    if not langgraph_available:
        monkeypatch.setitem(__import__("sys").modules, "langgraph", None)
    seen = []
    original = S.RunHandle.drain

    def spy(self):
        out = original(self)
        seen.append(out)
        return out

    monkeypatch.setattr(S.RunHandle, "drain", spy)
    at = submit(app())
    streamed = "".join(seen)
    assert "Visitors enjoy" in streamed and "## Day 1" in streamed
    page = blob(at)
    assert "# South India itinerary" in page and "## Sources" in page
    assert "Visitors enjoy" in page and "fresh run" in page
    assert not S.run_lock_held()
    # live was not requested: the note says so instead of "switched off (config/live.json)"
    assert S.NOT_REQUESTED_NOTE in page and "switched off" not in page
    assert "live search:" not in page.replace("live search: terms", "")  # no status line


LIVE_OUTCOMES = [
    ("ok", {}, "live search: ok, 2 searches, 8 results, 2 in scope, 2 merged, 1 cited"),
    ("empty", {"client": "empty"},
     "live search: empty, 2 searches, 0 results, 0 in scope, 0 merged, 0 cited"),
    ("error", {"client": "boom"},
     "live search: error: live search failed; answers use the corpus only"),
    ("no_key", {"key": None}, "live search: no key"),
]


@pytest.mark.parametrize("name,kw,line", LIVE_OUTCOMES, ids=[o[0] for o in LIVE_OUTCOMES])
def test_the_page_shows_a_counts_only_live_status_line(tmp_path, monkeypatch, name, kw, line):
    from tests.test_live import FakeClient as SearchClient
    from tests.test_live import retriever as live_retriever

    client = {"empty": SearchClient(results=[]), "boom": SearchClient(boom=True)}.get(
        kw.get("client"), SearchClient())
    live = live_retriever(tmp_path, client=client, key=kw.get("key", "k"))
    real_page(tmp_path, monkeypatch, make_deps(live=live))
    at = app()
    at.text_input(key="f_places_free").set_value("Hampi")
    at = submit(at, **{"checkbox__f_live": True})
    page = blob(at)
    assert line in page
    caption = next(str(c.value) for c in at.caption if str(c.value).startswith("live search:")
                   and "terms" not in str(c.value))
    assert caption == line  # the line itself carries counts and labels only
    assert not any(w in caption for w in ("ruins", "Badami", "Paris", "http"))
    assert files_in(tmp_path) == []  # a live-on answer writes nothing
    sources = next(d for d in at.dataframe if "Chunk id" in d.value.columns).value
    live_rows = sources[sources["Source"].str.startswith("live")]
    assert len(live_rows) == (1 if name == "ok" else 0)  # live rows only when a live item was cited
    if name == "ok":
        assert live_rows.iloc[0]["Chunk id"].startswith("live:")
        assert live_rows.iloc[0]["Fetch date"] != ""
    assert (TERMS_NOTICE in page) is (name != "no_key")


def files_in(tmp_path):
    d = tmp_path / "appcache"
    return sorted(d.glob("*.json")) if d.exists() else []


def test_missing_cards_or_attribution_are_said_on_the_page(fake, monkeypatch):
    monkeypatch.setattr(S, "load_cards", lambda: {})
    monkeypatch.setattr(S, "load_attribution", lambda: {})
    page = blob(submit(app()))
    assert "place cards are not available" in page and "source titles are not available" in page
