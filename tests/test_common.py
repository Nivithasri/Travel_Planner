"""Tests for src/common: cache, settings, schemas, HTTP client, LLM client (D47)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel, ValidationError

from src.common import cache as cache_mod
from src.common.cache import (
    DiskCache,
    append_jsonl,
    atomic_write_text,
    cache_key,
    read_jsonl,
)
from src.common.http import (
    DeadlineExceeded,
    FetchError,
    HttpClient,
    RequestBudgetExhausted,
    RequestLog,
    RobotsDisallowed,
    TransportError,
    TransportResponse,
    redact,
    requests_transport,
)
from src.common.llm import (
    BadResponseError,
    JSONValidationError,
    LLMError,
    ModelNotFoundError,
    OllamaClient,
)
from src.common.schemas import (
    BaseChunk,
    ConfigError,
    EnrichedChunk,
    PlaceRegistryEntry,
    load_settings,
    make_chunk_id,
)

# ==== from test_cache.py


ABSENT = object()  # get(..., default=ABSENT) is ABSENT means "not cached"


def test_key_is_stable_and_order_independent():
    a = cache_key("llm", "m", {"x": 1, "y": [1, 2], "z": "é"})
    b = cache_key("llm", "m", {"z": "é", "y": [1, 2], "x": 1})
    assert a == b and len(a) == 64


def test_key_changes_with_kind_model_inputs():
    base = cache_key("llm", "m", {"x": 1})
    assert cache_key("emb", "m", {"x": 1}) != base
    assert cache_key("llm", "m2", {"x": 1}) != base
    assert cache_key("llm", "m", {"x": 2}) != base


def test_get_or_compute_hit_skips_fn(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    def fn():
        calls.append(1)
        return {"v": 42}

    assert cache.get_or_compute("k", "m", {"q": 1}, fn) == {"v": 42}
    assert cache.get_or_compute("k", "m", {"q": 1}, fn) == {"v": 42}
    assert len(calls) == 1
    # a fresh instance on the same dir also hits (persisted on disk)
    assert DiskCache(tmp_path).get_or_compute("k", "m", {"q": 1}, fn) == {"v": 42}
    assert len(calls) == 1


def test_cached_none_is_a_hit(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []
    for _ in range(2):
        cache.get_or_compute("k", "m", {}, lambda: calls.append(1))
    assert len(calls) == 1
    assert cache.get("k", "m", {}, default=ABSENT) is not ABSENT


def test_failed_fn_stores_nothing(tmp_path):
    cache = DiskCache(tmp_path)

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        cache.get_or_compute("k", "m", {}, boom)
    assert cache.get("k", "m", {}, default=ABSENT) is ABSENT
    assert not list(tmp_path.rglob("*.tmp"))


def test_corrupt_entry_is_a_miss_and_overwritten(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("k", "m", {"a": 1}, "good")
    cache.path_for("k", "m", {"a": 1}).write_text("{not json", encoding="utf-8")
    assert cache.get("k", "m", {"a": 1}) is None
    assert cache.get_or_compute("k", "m", {"a": 1}, lambda: "fresh") == "fresh"


def test_atomic_write_leaves_no_temp_files_and_cleans_up_on_error(tmp_path, monkeypatch):
    target = tmp_path / "d" / "f.json"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]

    import os

    def bad_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", bad_replace)
    with pytest.raises(OSError):
        atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "hello"  # original untouched
    assert [p.name for p in target.parent.iterdir()] == ["f.json"]  # temp removed


def test_jsonl_roundtrip_and_append(tmp_path):
    p = tmp_path / "a.jsonl"
    assert list(read_jsonl(p)) == []  # missing file
    append_jsonl(p, [{"a": 1}, {"b": "é"}])
    append_jsonl(p, [{"c": 3}])
    assert list(read_jsonl(p)) == [{"a": 1}, {"b": "é"}, {"c": 3}]


def test_jsonl_truncated_tail_is_skipped_but_mid_corruption_raises(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n{"c": ', encoding="utf-8")
    assert list(read_jsonl(p)) == [{"a": 1}, {"b": 2}]
    # resume: appending after a truncated tail would corrupt the middle; callers re-write
    p.write_text('{"a": 1}\n{oops\n{"c": 3}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(p))


def test_append_after_crash_repairs_partial_tail_and_keeps_new_records(tmp_path):
    p = tmp_path / "a.jsonl"
    append_jsonl(p, [{"a": 1}])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"b":')  # half a line, no newline: simulated crash mid-write
    append_jsonl(p, [{"c": 3}, {"d": 4}])
    assert list(read_jsonl(p)) == [{"a": 1}, {"c": 3}, {"d": 4}]


def test_append_after_crash_when_the_only_line_is_partial(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"half"', encoding="utf-8")
    append_jsonl(p, [{"x": 1}, {"y": 2}])
    assert list(read_jsonl(p)) == [{"x": 1}, {"y": 2}]


def test_append_keeps_a_complete_record_that_only_lacks_its_newline(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}', encoding="utf-8")  # last record complete, newline lost
    append_jsonl(p, [{"c": 3}])
    assert list(read_jsonl(p)) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_append_to_empty_or_new_file(tmp_path):
    p = tmp_path / "sub" / "a.jsonl"
    append_jsonl(p, [{"a": 1}])
    q = tmp_path / "empty.jsonl"
    q.write_text("", encoding="utf-8")
    append_jsonl(q, [{"a": 1}])
    assert list(read_jsonl(p)) == list(read_jsonl(q)) == [{"a": 1}]


def test_read_never_drops_a_complete_corrupt_line_silently(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{oops\n', encoding="utf-8")  # newline-terminated: not a crash tail
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(p))


def test_read_yields_complete_unterminated_last_record(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}', encoding="utf-8")
    assert list(read_jsonl(p)) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",  # corrupt
        b"\xff\xfe\x00bad utf8",  # not UTF-8
        b"[1, 2, 3]",  # valid JSON, wrong shape (not a dict)
        b'{"kind": "k"}',  # dict without "value"
        b"",  # empty file
    ],
)
def test_unreadable_cache_entries_are_a_miss_and_get_overwritten(tmp_path, content):
    cache = DiskCache(tmp_path)
    path = cache.path_for("k", "m", {"a": 1})
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    assert cache.get("k", "m", {"a": 1}, default="dflt") == "dflt"
    assert cache.get("k", "m", {"a": 1}, default=ABSENT) is ABSENT
    assert cache.get_or_compute("k", "m", {"a": 1}, lambda: "fresh") == "fresh"
    assert cache.get("k", "m", {"a": 1}) == "fresh"


def test_oserror_on_read_is_a_miss(tmp_path):
    cache = DiskCache(tmp_path)
    cache.path_for("k", "m", {}).mkdir(parents=True)  # a directory where the file should be
    assert cache.get("k", "m", {}, default="dflt") == "dflt"


def test_atomic_write_retries_on_windows_permission_error(tmp_path, monkeypatch):
    real = cache_mod.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("file in use")
        return real(src, dst)

    monkeypatch.setattr(cache_mod.os, "replace", flaky)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda s: None)
    target = tmp_path / "f.json"
    atomic_write_text(target, "x")
    assert target.read_text() == "x" and calls["n"] == 3


def test_atomic_write_gives_up_after_attempts(tmp_path, monkeypatch):
    def always(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(cache_mod.os, "replace", always)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        atomic_write_text(tmp_path / "f.json", "x")
    assert list(tmp_path.glob("*.tmp")) == []


# ==== from test_config.py


KEYS = [
    "OLLAMA_BASE_URL",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "EMBED_MODEL",
    "EVAL_REFERENCE_DATE",
    "CACHE_DIR",
]


@pytest.fixture
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def write_env(tmp_path, **over):
    vals = {
        "OLLAMA_BASE_URL": "http://host:11434/",
        "GENERATOR_MODEL": "gen",
        "JUDGE_MODEL": "judge",
        "EMBED_MODEL": "emb",
        "EVAL_REFERENCE_DATE": "2026-10-18",
    }
    vals.update(over)
    p = tmp_path / ".env"
    p.write_text("\n".join(f"{k}={v}" for k, v in vals.items() if v is not None), encoding="utf-8")
    return p


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_loads_and_normalises(tmp_path, monkeypatch):
    s = load_settings(write_env(tmp_path))
    assert s.ollama_base_url == "http://host:11434"
    assert s.eval_reference_date == date(2026, 10, 18)
    assert s.generator_model == "gen" and s.judge_model == "judge"
    monkeypatch.delenv("OLLAMA_BASE_URL")  # load_dotenv put these in os.environ


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_missing_var_raises(tmp_path):
    with pytest.raises(ConfigError, match="JUDGE_MODEL"):
        load_settings(write_env(tmp_path, JUDGE_MODEL=None))


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_generator_equals_judge_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must differ"):
        load_settings(write_env(tmp_path, JUDGE_MODEL="gen"))


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_bad_date_rejected(tmp_path):
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        load_settings(write_env(tmp_path, EVAL_REFERENCE_DATE="18/10/2026"))


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GENERATOR_MODEL", "from-env")
    assert load_settings(write_env(tmp_path)).generator_model == "from-env"


@pytest.fixture
def clean_env_optional(monkeypatch):
    for k in [*KEYS, "LLM_SEED", "LLM_RETRY_BUDGET_S"]:
        monkeypatch.delenv(k, raising=False)


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
def test_seed_and_retry_budget_defaults_and_overrides(tmp_path, monkeypatch):
    s = load_settings(write_env(tmp_path))
    assert s.llm_seed == 42 and s.llm_retry_budget_s == 300
    monkeypatch.setenv("LLM_SEED", "7")
    monkeypatch.setenv("LLM_RETRY_BUDGET_S", "90.5")
    s = load_settings(write_env(tmp_path))
    assert s.llm_seed == 7 and s.llm_retry_budget_s == 90.5


@pytest.mark.usefixtures("clean_env", "clean_env_optional")
@pytest.mark.parametrize(
    "name,value", [("LLM_SEED", "abc"), ("LLM_RETRY_BUDGET_S", "x"), ("LLM_RETRY_BUDGET_S", "0")]
)
def test_bad_optional_numbers_rejected(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_settings(write_env(tmp_path))


# ==== from test_schemas.py


BASE = {
    "chunk_id": "abc",
    "text": "The fort is open 9-5.",
    "source_type": "blog",
    "url": "https://example.com/a",
    "date": "2025-03-01",
    "date_kind": "published",
    "revised_date": None,
    "state": "KA",
}


ENRICHED = {
    **BASE,
    "context_note": "Hampi fort timings from a blog.",
    "place": "Hampi",
    "place_type": "heritage",
    "season_tags": ["winter"],
    "aspect": "timing",
    "is_warning": False,
}


PLACE = {
    "name": "Munnar",
    "aliases": ["Munar"],
    "state": "KL",
    "region_cluster": "western_ghats",
    "popularity": "popular",
    "lat": 10.0889,
    "lon": 77.0595,
}


def test_base_chunk_has_no_fake_phase3_values():
    c = BaseChunk(**BASE)
    assert c.aspect is None and c.place is None and c.is_warning is None
    assert c.season_tags is None and c.place_type is None
    assert c.context_note == ""  # contract: "" before Phase 3


def test_enriched_roundtrip_matches_contract_fields():
    c = EnrichedChunk(**ENRICHED)
    assert c.model_dump(mode="json") == {**ENRICHED, "mentioned_places": []}  # D50 default


def test_enriched_requires_everything():
    for missing in ("context_note", "place", "place_type", "season_tags", "aspect", "is_warning"):
        data = {k: v for k, v in ENRICHED.items() if k != missing}
        with pytest.raises(ValidationError):
            EnrichedChunk(**data)
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, "context_note": ""})


def test_date_null_ok_but_key_required_and_format_checked():
    assert BaseChunk(**{**BASE, "date": None, "date_kind": "none"}).date is None
    no_date = {k: v for k, v in BASE.items() if k != "date"}
    with pytest.raises(ValidationError):
        BaseChunk(**no_date)
    for bad in ("2025-3-1", "01/03/2025", "2025-13-40", ""):
        with pytest.raises(ValidationError):
            BaseChunk(**{**BASE, "date": bad})


def test_date_kind_invariants_d48():
    ok = [
        {"date": "2025-03-01", "date_kind": "published", "revised_date": None},
        {"date": None, "date_kind": "revised", "revised_date": "2026-09-01"},
        {"date": None, "date_kind": "sitemap_lastmod", "revised_date": None},
        {"date": None, "date_kind": "none", "revised_date": None},
    ]
    for d in ok:
        assert BaseChunk(**{**BASE, **d}).date_kind.value == d["date_kind"]
    bad = [
        {"date": None, "date_kind": "published", "revised_date": None},  # published needs a date
        {"date": "2025-03-01", "date_kind": "revised", "revised_date": "2026-09-01"},  # no date
        {"date": "2025-03-01", "date_kind": "none", "revised_date": None},
        {"date": "2025-03-01", "date_kind": "sitemap_lastmod", "revised_date": None},
        {"date": None, "date_kind": "revised", "revised_date": None},  # revised needs revised_date
        {"date": None, "date_kind": "none", "revised_date": "2026-09-01"},
        {"date": "2025-03-01", "date_kind": "published", "revised_date": "2026-09-01"},
        {"date": None, "date_kind": "revised", "revised_date": "2026-9-1"},  # format
        {"date": None, "date_kind": "lastmod", "revised_date": None},  # enum
    ]
    for d in bad:
        with pytest.raises(ValidationError):
            BaseChunk(**{**BASE, **d})
    with pytest.raises(ValidationError):  # date_kind is required
        BaseChunk(**{k: v for k, v in BASE.items() if k != "date_kind"})


def test_live_is_not_a_corpus_source_type():
    with pytest.raises(ValidationError):
        BaseChunk(**{**BASE, "source_type": "live"})


@pytest.mark.parametrize(
    "field,value",
    [("source_type", "twitter"), ("state", "MH"), ("aspect", "weather")],
)
def test_enums_rejected(field, value):
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, field: value})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        BaseChunk(**{**BASE, "score": 0.5})


def test_chunk_id_deterministic():
    assert make_chunk_id("u", 0) == make_chunk_id("u", 0)
    assert make_chunk_id("u", 0) != make_chunk_id("u", 1)
    assert make_chunk_id("u", 0) != make_chunk_id("v", 0)


def test_place_registry():
    p = PlaceRegistryEntry(**PLACE)
    assert p.model_dump(mode="json", exclude_none=True) == PLACE
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "popularity": "famous"})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "lat": 123})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{**PLACE, "state": "GJ"})
    with pytest.raises(ValidationError):
        PlaceRegistryEntry(**{k: v for k, v in PLACE.items() if k != "aliases"})


def test_reference_is_a_valid_source_type_and_enum_matches_contract():
    from src.common.schemas import SourceType
    assert {s.value for s in SourceType} == {
        "reddit", "blog", "youtube", "wikivoyage", "official", "reference"}
    contract = (Path(__file__).resolve().parents[1] / "docs" / "contracts.md").read_text("utf-8")
    assert all(s.value in contract.split("| source_type |")[1].split("\n")[0] for s in SourceType)


def test_chunk_accepts_reference_source_type():
    assert BaseChunk(**{**BASE, "source_type": "reference"}).source_type.value == "reference"


@pytest.mark.parametrize("bad", ["yes", "true", 1, 0, "False", None])
def test_is_warning_is_strict_bool_on_enriched(bad):
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, "is_warning": bad})


@pytest.mark.parametrize("bad", ["yes", "true", 1, 0])
def test_is_warning_is_strict_bool_on_base(bad):
    with pytest.raises(ValidationError):
        BaseChunk(**{**BASE, "is_warning": bad})


def test_is_warning_accepts_real_bools_and_json_booleans():
    assert EnrichedChunk(**{**ENRICHED, "is_warning": True}).is_warning is True
    assert BaseChunk(**BASE).is_warning is None
    raw = (
        '{"chunk_id":"c","text":"t","source_type":"blog","url":"u","date":null,'
        '"date_kind":"none","state":"TN","is_warning":true}'
    )
    assert BaseChunk.model_validate_json(raw).is_warning is True


# ==== from test_http.py


UA = "TravelPlannerTest/0.1 (test@example.com)"


class http_FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeServer:
    """Maps url -> response (or list of responses consumed in order)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        r = self.routes.get(url)
        if r is None:
            return TransportResponse(404, {}, "")
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r


def http_make(tmp_path, routes, **kw):
    clock = http_FakeClock()
    server = FakeServer(routes)
    client = HttpClient(
        DiskCache(tmp_path / "c"), UA, transport=server, clock=clock, sleep=clock.sleep, **kw
    )
    return client, server, clock


def http_ok(text="hi", **headers):
    return TransportResponse(200, headers, text)


ROBOTS_OPEN = TransportResponse(200, {}, "User-agent: *\nAllow: /\n")


def test_cache_hit_makes_no_request(tmp_path):
    client, server, _ = http_make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                        "https://a.org/p": http_ok("body")})
    first = client.fetch("https://a.org/p")
    n = client.requests_made
    second = client.fetch("https://a.org/p")
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.text == "body"
    assert client.requests_made == n == 2  # robots.txt + page


def test_stale_entry_is_revalidated_with_etag_and_304_refreshes(tmp_path):
    routes = {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/p": [http_ok("v1", etag='"e1"'), TransportResponse(304, {}, "")],
    }
    client, server, clock = http_make(tmp_path, routes)
    client.fetch("https://a.org/p", max_age=60)
    clock.t += 120
    res = client.fetch("https://a.org/p", max_age=60)
    assert res.text == "v1" and not res.from_cache
    assert server.calls[-1][1]["If-None-Match"] == '"e1"'
    assert res.retrieved_at == clock.t
    n = client.requests_made
    client.fetch("https://a.org/p", max_age=60)  # fresh again
    assert client.requests_made == n


def test_robots_disallow_blocks_without_fetching_page(tmp_path):
    robots = TransportResponse(200, {}, "User-agent: *\nDisallow: /private\n")
    client, server, _ = http_make(tmp_path, {"https://a.org/robots.txt": robots,
                                        "https://a.org/private/x": http_ok()})
    with pytest.raises(RobotsDisallowed):
        client.fetch("https://a.org/private/x")
    assert all("private" not in u for u, _ in server.calls)
    assert client.fetch("https://a.org/robots.txt", check_robots=False).ok


def test_missing_robots_allows_but_unreachable_robots_disallows(tmp_path):
    client, _, _ = http_make(tmp_path, {"https://a.org/p": http_ok()})  # robots.txt -> 404
    assert client.fetch("https://a.org/p").ok
    bad_robots = TransportResponse(503, {}, "")
    client2, _, _ = http_make(
        tmp_path / "x", {"https://b.org/robots.txt": bad_robots, "https://b.org/p": http_ok()},
        max_retries=1,
    )
    with pytest.raises(RobotsDisallowed):
        client2.fetch("https://b.org/p")


def test_min_delay_between_requests_to_one_host(tmp_path):
    client, _, clock = http_make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                       "https://a.org/1": http_ok(), "https://a.org/2": http_ok()},
                            min_delay_s=2.0)
    client.fetch("https://a.org/1")
    t = clock.t
    client.fetch("https://a.org/2")
    assert clock.t - t >= 2.0


def test_retries_5xx_and_transport_errors_then_succeeds(tmp_path):
    routes = {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/p": [TransportResponse(503, {}, ""), TransportError("boom"),
                            http_ok("fine")],
    }
    client, _, _ = http_make(tmp_path, routes)
    assert client.fetch("https://a.org/p").text == "fine"


def test_gives_up_and_does_not_cache_errors(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/p": [TransportResponse(500, {}, "")]}
    client, _, _ = http_make(tmp_path, routes, max_retries=2)
    with pytest.raises(FetchError) as ei:
        client.fetch("https://a.org/p")
    assert ei.value.status == 500
    absent = object()
    assert client.cache.get("http", "a.org", {"url": "https://a.org/p", "headers": {}},
                            default=absent) is absent


def test_404_is_cached_as_negative_result(tmp_path):
    client, _, _ = http_make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN})
    assert client.fetch("https://a.org/missing").status == 404
    n = client.requests_made
    assert client.fetch("https://a.org/missing").status == 404
    assert client.requests_made == n


def test_deadline_stops_fetching(tmp_path):
    client, _, clock = http_make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                       "https://a.org/p": http_ok()})
    with pytest.raises(DeadlineExceeded):
        client.fetch("https://a.org/p", deadline=clock.t - 1)


def test_params_are_part_of_the_cache_key(tmp_path):
    client, _, _ = http_make(tmp_path, {
        "https://a.org/robots.txt": ROBOTS_OPEN,
        "https://a.org/api?q=1": http_ok("one"), "https://a.org/api?q=2": http_ok("two")})
    assert client.fetch("https://a.org/api", {"q": 1}).text == "one"
    assert client.fetch("https://a.org/api", {"q": 2}).text == "two"


def test_user_agent_required(tmp_path):
    with pytest.raises(ValueError):
        HttpClient(DiskCache(tmp_path), " ")


def test_failed_robots_lookup_is_not_memoized(tmp_path):
    routes = {
        "https://a.org/robots.txt": [TransportResponse(503, {}, ""), ROBOTS_OPEN],
        "https://a.org/p": http_ok("late"),
    }
    client, _, _ = http_make(tmp_path, routes, max_retries=0)
    with pytest.raises(RobotsDisallowed):
        client.fetch("https://a.org/p")
    assert client.fetch("https://a.org/p").text == "late"  # robots recovered, same client


def _no_error(text: str) -> bool:
    return '"error"' not in text


def test_validate_rejects_error_bodies_and_never_caches_them(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/api": [http_ok('{"error": {"code": "x"}}'), http_ok('{"fine": 1}')]}
    client, _, _ = http_make(tmp_path, routes)
    with pytest.raises(FetchError):
        client.fetch("https://a.org/api", validate=_no_error)
    assert client.fetch("https://a.org/api", validate=_no_error).text == '{"fine": 1}'


def test_cached_error_body_is_treated_as_a_miss(tmp_path):
    routes = {"https://a.org/robots.txt": ROBOTS_OPEN,
              "https://a.org/api": [http_ok('{"error": 1}'), http_ok('{"good": 1}')]}
    client, _, _ = http_make(tmp_path, routes)
    assert client.fetch("https://a.org/api").text == '{"error": 1}'  # cached without validation
    assert client.fetch("https://a.org/api", validate=_no_error).text == '{"good": 1}'


def test_secret_headers_are_sent_but_not_part_of_the_cache_key_or_the_cache_file(tmp_path):
    client, server, _ = http_make(tmp_path, {"https://a.org/robots.txt": ROBOTS_OPEN,
                                        "https://a.org/p": http_ok("body")})
    client.fetch("https://a.org/p", secret_headers={"Authorization": "Bearer TOKEN-1"})
    assert server.calls[-1][1]["Authorization"] == "Bearer TOKEN-1"
    n = client.requests_made
    hit = client.fetch("https://a.org/p", secret_headers={"Authorization": "Bearer TOKEN-2"})
    assert hit.from_cache and client.requests_made == n  # a rotated token still hits the cache
    stored = "".join(p.read_text("utf-8") for p in (tmp_path / "c").rglob("*.json"))
    assert "TOKEN" not in stored


POLICY_UA = "Test/0.1 (x@example.org)"


OPEN = TransportResponse(404, {}, "")  # no robots.txt: everything allowed


def redirect(to, status=301):
    return TransportResponse(status, {"location": to}, "")


def pok(text="body"):
    return TransportResponse(200, {}, text)


class Server:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, method, url, headers, timeout):
        self.calls.append(url)
        r = self.routes.get(url, OPEN)
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r


def pmake(tmp_path, routes, log=False, **kw):
    server = Server(routes)
    rlog = RequestLog(tmp_path / "log.jsonl", run_id="t1") if log else None
    http = HttpClient(DiskCache(tmp_path / "c"), POLICY_UA, transport=server, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0, request_log=rlog, **kw)
    return http, server, rlog


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "log.jsonl").read_text("utf-8").splitlines()]


def test_transport_does_not_follow_redirects_itself(monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen.update(kw)
        raise requests.ConnectionError("stop")

    monkeypatch.setattr(requests, "request", fake_request)
    with pytest.raises(TransportError):
        requests_transport("GET", "https://a.org/", {}, (1, 1))
    assert seen["allow_redirects"] is False


def test_same_site_redirect_is_followed_and_every_hop_is_counted_and_logged(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://www.a.org/y"),
                                      "https://www.a.org/y": pok("final")}, log=True)
    r = http.fetch("https://a.org/x", check_robots=False)
    assert r.text == "final" and http.requests_made == 2
    assert [(x["host"], x["status"], x["note"]) for x in lines(tmp_path)] == [
        ("a.org", 301, "redirect"), ("www.a.org", 200, None)]


def test_redirect_to_another_site_is_refused_and_logged(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://evil.example/y")},
                           log=True)
    with pytest.raises(FetchError, match="not followed"):
        http.fetch("https://a.org/x", check_robots=False)
    assert server.calls == ["https://a.org/x"]  # the other host was never contacted
    refused = [x for x in lines(tmp_path) if (x["note"] or "").startswith("redirect-refused")]
    assert len(refused) == 1 and refused[0]["host"] == "evil.example"
    assert "outside the allowlist" in refused[0]["note"]


def test_redirect_host_on_the_allowlist_is_followed(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://b.org/y"),
                                 "https://b.org/y": pok("b")}, redirect_hosts={"b.org"})
    assert http.fetch("https://a.org/x", check_robots=False).text == "b"


def test_https_to_http_downgrade_is_refused(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/x": redirect("http://a.org/y")})
    with pytest.raises(FetchError, match="downgrade"):
        http.fetch("https://a.org/x", check_robots=False)
    assert server.calls == ["https://a.org/x"]


def test_redirect_target_disallowed_by_robots_is_not_followed(tmp_path):
    http, server, _ = pmake(tmp_path, {
        "https://a.org/robots.txt": pok("User-agent: *\nDisallow: /private\n"),
        "https://a.org/x": redirect("https://a.org/private/y")})
    with pytest.raises(FetchError, match="robots.txt disallows the target"):
        http.fetch("https://a.org/x")
    assert "https://a.org/private/y" not in server.calls


def test_redirect_loop_is_cut_off(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/x": redirect("https://a.org/x")})
    with pytest.raises(FetchError, match="too many redirects"):
        http.fetch("https://a.org/x", check_robots=False)
    assert http.requests_made == 6  # the first request plus 5 followed hops


FAKE_KEY = "ABCDEFGH12345678"


def test_redact_removes_query_strings_and_key_values():
    text = (f"403 Client Error: Forbidden for url: https://www.youtube.com/youtubei/v1/player"
            f"?key={FAKE_KEY}&prettyPrint=false and token={FAKE_KEY} and api_key: {FAKE_KEY}")
    out = redact(text)
    assert FAKE_KEY not in out and "prettyPrint" not in out
    assert "https://www.youtube.com/youtubei/v1/player" in out


def test_http_error_with_a_key_in_its_url_is_redacted_in_fetch_errors(tmp_path):
    http_error = requests.HTTPError(f"403 Client Error: Forbidden for url: "
                                    f"https://a.org/p?key={FAKE_KEY}")
    boom = TransportError(str(http_error))
    http, _, _ = pmake(tmp_path, {"https://a.org/p?key=" + FAKE_KEY: boom})
    with pytest.raises(FetchError) as ei:
        http.fetch("https://a.org/p", {"key": FAKE_KEY}, check_robots=False)
    assert FAKE_KEY not in str(ei.value)
    assert FAKE_KEY not in redact(http_error)


def test_request_log_has_no_query_string_content_or_key(tmp_path):
    http, _, _ = pmake(tmp_path, {"https://a.org/p?key=" + FAKE_KEY: pok("SECRET-PAGE-TEXT")},
                      log=True)
    http.fetch("https://a.org/p", {"key": FAKE_KEY}, check_robots=False)
    raw = (tmp_path / "log.jsonl").read_text("utf-8")
    assert FAKE_KEY not in raw and "SECRET-PAGE-TEXT" not in raw and "?" not in raw


def test_request_log_records_fields_cache_hits_and_retry_attempts(tmp_path):
    routes = {"https://a.org/robots.txt": OPEN,
              "https://a.org/sitemap.xml": [TransportResponse(503, {}, ""), pok("<urlset/>")],
              "https://a.org/w/api.php?x=1": pok("{}")}
    http, _, rlog = pmake(tmp_path, routes, log=True)
    http.max_retries = 1
    rlog.source = "web"
    http.fetch("https://a.org/sitemap.xml")
    http.fetch("https://a.org/sitemap.xml")  # cache hit
    http.fetch("https://a.org/w/api.php", {"x": 1}, check_robots=False)
    got = [(x["kind"], x["status"], x["from_cache"], x["attempt"]) for x in lines(tmp_path)]
    assert got == [("robots", 404, False, 1), ("sitemap", 503, False, 1),
                   ("sitemap", 200, False, 2), ("sitemap", 200, True, 0),
                   ("api", 200, False, 1)]
    first = lines(tmp_path)[0]
    assert set(first) == {"ts", "run_id", "host", "path", "kind", "status", "from_cache",
                          "attempt", "source", "note"}
    assert first["source"] == "web" and first["run_id"] == "t1"


def test_request_cap_raises_at_the_next_network_request_and_cache_hits_are_free(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://a.org/1": pok(), "https://a.org/2": pok(),
                                      "https://a.org/3": pok()}, max_requests=2)
    http.fetch("https://a.org/1", check_robots=False)
    http.fetch("https://a.org/1", check_robots=False)  # cache hit: free
    http.fetch("https://a.org/2", check_robots=False)
    with pytest.raises(RequestBudgetExhausted):
        http.fetch("https://a.org/3", check_robots=False)
    assert http.requests_made == 2 and "https://a.org/3" not in server.calls


def test_budget_exhaustion_is_not_swallowed_by_a_collector(tmp_path):
    from src.collect.run import Budget
    from src.collect.sources import WikiCollector
    from src.common.schemas import PlaceRegistryEntry

    http, _, _ = pmake(tmp_path, {}, max_requests=0)
    e = PlaceRegistryEntry(name="X", aliases=[], state="KL", region_cluster="x",
                           popularity="popular", lat=1.0, lon=1.0, wikipedia_title="X")
    with pytest.raises(RequestBudgetExhausted):
        WikiCollector("wikipedia", http).collect(e, Budget(), lambda: 0, tmp_path / "raw")


@pytest.mark.parametrize("kind", ["tls", "dns"])
def test_tls_and_dns_failures_are_recorded_once_never_retried_and_not_counted(tmp_path, kind):
    http, server, _ = pmake(tmp_path, {"https://bad.org/a": TransportError("handshake", kind)},
                           log=True, max_requests=1)
    http.max_retries = 3
    for _ in range(3):
        with pytest.raises(FetchError, match="unreachable") as ei:
            http.fetch("https://bad.org/a", check_robots=False)
        assert ei.value.status is None  # not a 403/429, so never a block
    assert server.calls == ["https://bad.org/a"]  # one attempt in total, no retry
    assert http.requests_made == 0 and http.unreachable_attempts == 1
    assert list(http.unreachable_hosts) == ["bad.org"]
    notes = [x["note"] for x in lines(tmp_path)]
    assert notes == [f"unreachable:{kind}"]  # logged once
    http._transport = Server({"https://good.org/": pok()})  # another host works under the same cap
    assert http.fetch("https://good.org/", check_robots=False).ok


def test_preloaded_unreachable_hosts_are_never_contacted(tmp_path):
    http, server, _ = pmake(tmp_path, {}, unreachable_hosts={"tourism.telangana.gov.in": "tls"})
    with pytest.raises(FetchError, match="unreachable"):
        http.fetch("https://tourism.telangana.gov.in/", check_robots=False)
    assert server.calls == [] and http.requests_made == 0


def test_timeout_is_not_treated_as_unreachable(tmp_path):
    http, server, _ = pmake(tmp_path, {"https://slow.org/": TransportError("slow", "timeout")})
    with pytest.raises(FetchError):
        http.fetch("https://slow.org/", check_robots=False)
    assert http.unreachable_hosts == {} and http.requests_made == 1


# ==== from test_llm.py


class FakeResp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def llm_ok(content):
    return FakeResp(200, {"message": {"role": "assistant", "content": content}})


class Script:
    """Callable standing in for requests.post; plays back a list of responses/exceptions."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def llm_make(tmp_path, *items, **kw):
    post = Script(*items)
    sleeps = []
    client = OllamaClient(
        "http://host:11434/", DiskCache(tmp_path), post=post, sleep=sleeps.append, **kw
    )
    return client, post, sleeps


MSGS = [{"role": "user", "content": "hi"}]


def test_chat_basic_and_request_shape(tmp_path):
    client, post, _ = llm_make(tmp_path, llm_ok("hello"))
    assert client.chat(MSGS, "m1") == "hello"
    url, kw = post.calls[0]
    assert url == "http://host:11434/api/chat"
    assert kw["json"]["stream"] is False and kw["json"]["options"]["temperature"] == 0
    assert kw["timeout"][1] >= 300  # long read timeout
    assert "ngrok-skip-browser-warning" not in kw["headers"]


def test_cache_hit_makes_zero_calls_even_after_host_change(tmp_path):
    client, post, _ = llm_make(tmp_path, llm_ok("hello"))
    client.chat(MSGS, "m1")
    other = OllamaClient("https://abc.ngrok-free.app", DiskCache(tmp_path), post=Script())
    assert other.chat(MSGS, "m1") == "hello"  # Script() has no items: a call would raise
    assert len(post.calls) == 1
    assert OllamaClient("http://x", DiskCache(tmp_path), post=Script(llm_ok("z"))).chat(
        MSGS, "other-model"
    ) == "z"  # different model is a different key


def test_ngrok_header_added_only_for_ngrok(tmp_path):
    post = Script(llm_ok("a"))
    OllamaClient("https://abc.ngrok-free.app", DiskCache(tmp_path), post=post).chat(MSGS, "m")
    assert post.calls[0][1]["headers"]["ngrok-skip-browser-warning"] == "1"


def test_retry_then_success_with_backoff(tmp_path):
    client, post, sleeps = llm_make(
        tmp_path, requests.ConnectionError("x"), FakeResp(503, text="busy"), FakeResp(429),
        llm_ok("ok")
    )
    assert client.chat(MSGS, "m") == "ok"
    assert len(post.calls) == 4 and len(sleeps) == 3
    assert sleeps[0] < sleeps[2]  # exponential growth


def test_timeout_exhausted_raises_and_caches_nothing(tmp_path):
    client, post, _ = llm_make(tmp_path, *[requests.Timeout("t")] * 3, max_retries=2)
    with pytest.raises(LLMError, match="Giving up"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 3
    assert not list(tmp_path.rglob("*.json"))


def test_4xx_not_retried(tmp_path):
    client, post, sleeps = llm_make(tmp_path, FakeResp(400, text="bad request"))
    with pytest.raises(LLMError, match="HTTP 400"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 1 and sleeps == []


def test_404_clear_model_error(tmp_path):
    client, post, _ = llm_make(tmp_path, FakeResp(404, text="model not found"))
    with pytest.raises(ModelNotFoundError) as exc:
        client.chat(MSGS, "qwen")
    assert "model or endpoint not found" in str(exc.value)
    assert "check that the model is pulled and the URL is right" in str(exc.value)
    assert len(post.calls) == 1


def test_interstitial_200_is_retried_and_never_cached(tmp_path):
    html = FakeResp(200, text="<html>You are about to visit ngrok</html>")
    client, post, _ = llm_make(tmp_path, html, html, llm_ok("real"))
    assert client.chat(MSGS, "m") == "real"
    assert len(post.calls) == 3
    cached = list((tmp_path / "llm_chat").rglob("*.json"))
    assert len(cached) == 1 and "real" in cached[0].read_text(encoding="utf-8")


def test_interstitial_200_forever_raises_bad_response(tmp_path):
    html = FakeResp(200, text="<html>interstitial</html>")
    client, _, _ = llm_make(tmp_path, html, html, max_retries=1)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")
    assert not list(tmp_path.rglob("*.json"))


def test_200_json_without_message_is_bad_response(tmp_path):
    client, _, _ = llm_make(tmp_path, FakeResp(200, {"hello": 1}), max_retries=0)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")


class Plan(BaseModel):
    days: int
    city: str


def test_json_valid_is_cached_and_second_call_skips_network(tmp_path):
    client, post, _ = llm_make(tmp_path, llm_ok('{"days": 3, "city": "Hampi"}'))
    first = client.chat_json(MSGS, "m", Plan)
    again = client.chat_json(MSGS, "m", Plan)
    assert first == again == Plan(days=3, city="Hampi")
    assert len(post.calls) == 1
    assert post.calls[0][1]["json"]["format"] == "json"


def test_json_invalid_then_repaired(tmp_path):
    client, post, _ = llm_make(
        tmp_path, llm_ok("not json"), llm_ok('{"days": "x"}'), llm_ok('{"days": 2, "city": "A"}')
    )
    assert client.chat_json(MSGS, "m", Plan) == Plan(days=2, city="A")
    assert len(post.calls) == 3
    repair_msgs = post.calls[1][1]["json"]["messages"]
    assert repair_msgs[-2] == {"role": "assistant", "content": "not json"}
    assert "corrected JSON" in repair_msgs[-1]["content"]
    # the cache holds the validated result only
    files = list((tmp_path / "llm_json").rglob("*.json"))
    assert len(files) == 1 and not (tmp_path / "llm_chat").exists()


def test_json_permanently_invalid_raises_and_caches_nothing(tmp_path):
    client, post, _ = llm_make(tmp_path, llm_ok("x"), llm_ok("y"), llm_ok("z"), max_json_attempts=3)
    with pytest.raises(JSONValidationError):
        client.chat_json(MSGS, "m", Plan)
    assert len(post.calls) == 3
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    "body",
    [
        [1, 2, 3],  # JSON but not an object
        {"error": "overloaded"},  # no message / response key
        {"message": {"role": "assistant"}},  # message without content
        {"message": {"content": 5}},  # content not a string
    ],
)
def test_bad_body_structures_are_rejected_not_cached(tmp_path, body):
    client, _, _ = llm_make(tmp_path, FakeResp(200, body), max_retries=0)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")
    assert not list(tmp_path.rglob("*.json"))


def test_generate_style_response_key_is_accepted(tmp_path):
    client, _, _ = llm_make(tmp_path, FakeResp(200, {"response": "hi"}))
    assert client.chat(MSGS, "m") == "hi"


@pytest.mark.parametrize("content", ["", "   \n"])
def test_empty_reply_is_retried_and_never_cached(tmp_path, content):
    client, post, _ = llm_make(tmp_path, llm_ok(content), llm_ok("real"))
    assert client.chat(MSGS, "m") == "real"
    assert len(post.calls) == 2
    stored = [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")]
    assert len(stored) == 1 and "real" in stored[0]


def test_persistently_empty_reply_raises_and_caches_nothing(tmp_path):
    client, _, _ = llm_make(tmp_path, llm_ok(""), llm_ok(""), max_retries=1)
    with pytest.raises(BadResponseError, match="empty reply"):
        client.chat(MSGS, "m")
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ChunkedEncodingError("cut"),
        requests.exceptions.ContentDecodingError("bad gzip"),
        requests.exceptions.SSLError("tls"),
        requests.exceptions.ReadTimeout("slow"),
    ],
)
def test_transport_errors_are_retried(tmp_path, exc):
    client, post, _ = llm_make(tmp_path, exc, llm_ok("fine"))
    assert client.chat(MSGS, "m") == "fine"
    assert len(post.calls) == 2


def test_invalid_url_is_not_retried(tmp_path):
    client, post, _ = llm_make(tmp_path, requests.exceptions.MissingSchema("no scheme"))
    with pytest.raises(LLMError, match="not a valid URL"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 1


class llm_FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_retry_time_is_bounded_by_budget_not_attempts(tmp_path):
    clock = llm_FakeClock()
    calls = []

    def hung_post(url, **kwargs):
        calls.append(kwargs)
        clock.advance(100)  # each attempt burns 100 s before timing out
        raise requests.Timeout("hung")

    client = OllamaClient(
        "http://h",
        DiskCache(tmp_path),
        post=hung_post,
        sleep=clock.advance,
        clock=clock,
        retry_budget_s=250,
        max_retries=50,
    )
    with pytest.raises(LLMError, match="Giving up"):
        client.chat(MSGS, "m")
    assert len(calls) <= 3  # nowhere near max_retries=50
    assert clock.now <= 250 + 100  # never starts an attempt it has no budget for
    assert calls[0]["timeout"][1] <= 250  # read timeout is capped by the budget
    assert calls[-1]["timeout"][1] < calls[0]["timeout"][1]  # and shrinks as budget is used


def test_default_budget_is_a_few_minutes(tmp_path):
    client = OllamaClient("http://h", DiskCache(tmp_path))
    assert 60 <= client.retry_budget_s <= 600


def test_seed_is_sent_and_part_of_the_cache_key(tmp_path):
    post = Script(llm_ok("a"), llm_ok("b"))
    c1 = OllamaClient("http://h", DiskCache(tmp_path), post=post, seed=7)
    assert c1.chat(MSGS, "m") == "a"
    assert post.calls[0][1]["json"]["options"] == {"temperature": 0, "seed": 7}
    c2 = OllamaClient("http://h", DiskCache(tmp_path), post=post, seed=8)
    assert c2.chat(MSGS, "m") == "b"  # different seed, different cache entry
    assert len(post.calls) == 2


def test_default_seed_is_fixed(tmp_path):
    post = Script(llm_ok("a"))
    OllamaClient("http://h", DiskCache(tmp_path), post=post).chat(MSGS, "m")
    assert post.calls[0][1]["json"]["options"]["seed"] == 42


def test_chat_cache_key_changes_with_format_and_options(tmp_path):
    post = Script(llm_ok("plain"), llm_ok("json"), llm_ok("hot"), llm_ok("hot2"))
    client = OllamaClient("http://h", DiskCache(tmp_path), post=post)
    assert client.chat(MSGS, "m") == "plain"
    assert client.chat(MSGS, "m", format="json") == "json"
    assert client.chat(MSGS, "m", options={"temperature": 0.9}) == "hot"
    assert client.chat(MSGS, "m", options={"num_predict": 5}) == "hot2"
    assert len(post.calls) == 4
    # and each of those is now a hit
    assert client.chat(MSGS, "m") == "plain"
    assert client.chat(MSGS, "m", format="json") == "json"
    assert client.chat(MSGS, "m", options={"temperature": 0.9}) == "hot"
    assert len(post.calls) == 4


@pytest.mark.parametrize("stale", ["not json at all", '{"days": "x"}', 123, None])
def test_chat_json_stale_cache_entry_is_a_miss(tmp_path, stale):
    client, post, _ = llm_make(tmp_path, llm_ok('{"days": 1, "city": "Goa"}'))
    inputs = {
        "messages": MSGS,
        "options": {"temperature": 0, "seed": 42},
        "schema": Plan.model_json_schema(),
        "path": "/api/chat",
    }
    client.cache.set("llm_json", "m", inputs, stale)
    assert client.chat_json(MSGS, "m", Plan) == Plan(days=1, city="Goa")
    assert len(post.calls) == 1
    # overwritten with the validated result
    assert client.cache.get("llm_json", "m", inputs) == Plan(days=1, city="Goa").model_dump_json()


def test_list_models(tmp_path):
    get = Script(FakeResp(200, {"models": [{"name": "qwen2.5:7b"}, {"name": "bge-m3:latest"}]}))
    client = OllamaClient("https://x.ngrok-free.app", DiskCache(tmp_path), get=get)
    assert client.list_models() == ["qwen2.5:7b", "bge-m3:latest"]
    url, kw = get.calls[0]
    assert url == "https://x.ngrok-free.app/api/tags"
    assert kw["headers"]["ngrok-skip-browser-warning"] == "1"


def test_list_models_bad_body(tmp_path):
    client = OllamaClient(
        "http://h", DiskCache(tmp_path), get=Script(FakeResp(200, {"x": 1})), max_retries=0
    )
    with pytest.raises(BadResponseError):
        client.list_models()


def test_from_settings(tmp_path, monkeypatch):
    env = {
        "OLLAMA_BASE_URL": "http://box:11434/",
        "GENERATOR_MODEL": "g",
        "JUDGE_MODEL": "j",
        "EMBED_MODEL": "e",
        "EVAL_REFERENCE_DATE": "2026-10-18",
        "CACHE_DIR": str(tmp_path / "c"),
        "LLM_SEED": "99",
        "LLM_RETRY_BUDGET_S": "120",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("src.common.schemas.load_dotenv", lambda *a, **k: None)
    client = OllamaClient.from_settings(post=Script(llm_ok("hi")))
    assert client.base_url == "http://box:11434"
    assert client.cache.root == tmp_path / "c"
    assert client.seed == 99 and client.retry_budget_s == 120
    assert client.chat(MSGS, "g") == "hi"
    assert any((tmp_path / "c").rglob("*.json"))


def test_enriched_closed_enums_for_place_type_and_season_tags():
    from src.common.schemas import PlaceType, SeasonTag

    assert {p.value for p in PlaceType} == {
        "hill_station", "beach", "temple", "heritage", "wildlife", "backwater", "waterfall",
        "city", "region", "other"}
    assert {t.value for t in SeasonTag} == {"winter", "summer", "monsoon", "post_monsoon",
                                            "all_year"}
    EnrichedChunk(**{**ENRICHED, "place_type": "region", "season_tags": ["winter", "monsoon"]})
    for bad in ({"place_type": "mountain"}, {"place_type": ""}, {"season_tags": ["spring"]},
                {"season_tags": []}, {"season_tags": ["winter", "winter"]},
                {"season_tags": ["all_year", "winter"]}):
        with pytest.raises(ValidationError):
            EnrichedChunk(**{**ENRICHED, **bad})
    assert EnrichedChunk(**{**ENRICHED, "season_tags": ["all_year"]}).season_tags == ["all_year"]


def test_enriched_mentioned_places_is_optional_and_a_list_of_strings():
    assert EnrichedChunk(**ENRICHED).mentioned_places == []
    c = EnrichedChunk(**{**ENRICHED, "mentioned_places": ["Munnar", "Ooty"]})
    assert c.mentioned_places == ["Munnar", "Ooty"]
    with pytest.raises(ValidationError):
        EnrichedChunk(**{**ENRICHED, "mentioned_places": "Munnar"})
    with pytest.raises(ValidationError):  # BaseChunk (Phase 2) does not know the field
        BaseChunk(**{**BASE, "mentioned_places": []})


def test_load_embed_local_dir_reads_only_the_env(tmp_path, monkeypatch):
    from src.common.schemas import load_embed_local_dir

    monkeypatch.delenv("EMBED_LOCAL_DIR", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    assert load_embed_local_dir(env) is None
    env.write_text("EMBED_LOCAL_DIR=models/bge-m3-full\n", encoding="utf-8")
    assert load_embed_local_dir(env) == Path("models/bge-m3-full")
