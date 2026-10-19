"""Hosted OpenAI-compatible API client, rate limiter and settings (D59).

Offline: a fake transport and a fake clock; nothing here touches the network or the real .env."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel

from src.common.cache import DiskCache
from src.common.llm import (
    AuthError,
    BadResponseError,
    JSONValidationError,
    LLMError,
    ModelNotFoundError,
    OpenAICompatClient,
    QuotaExhausted,
    RateInfo,
    RateLimiter,
    parse_duration,
    parse_rate_headers,
    retry_after_seconds,
    scrub_text,
)
from src.common.schemas import (
    ConfigError,
    env_duplicate_keys,
    load_llm_settings,
)

BASE = "https://api.fake-provider.test/openai/v1"
HOST = "api.fake-provider.test"
KEY = "gsk_TESTKEY0123456789abcdef"
MODEL = "openai/gpt-oss-20b"
OPTIONS = {"max_tokens": 300, "reasoning_effort": "low"}
MESSAGES = [{"role": "system", "content": "Label the text."}, {"role": "user", "content": "hello"}]


class Pong(BaseModel):
    ok: bool
    word: str


class HTTP:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def completion(content, *, prompt=100, comp=50, reasoning=10, message_extra=None,
               finish="stop", usage=True):
    msg = {"role": "assistant", "content": content, **(message_extra or {})}
    body = {"choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": prompt, "completion_tokens": comp,
                         "total_tokens": prompt + comp,
                         "completion_tokens_details": {"reasoning_tokens": reasoning}}
    return HTTP(200, body)


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class Post:
    """Fake requests.post: answers from a list (the last answer repeats) or a callable."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[dict] = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        a = self.answers
        r = a(len(self.calls), json) if callable(a) else (a.pop(0) if len(a) > 1 else a[0])
        if isinstance(r, Exception):
            raise r
        return r


def make(tmp_path, answers, *, limiter=None, clk=None, **kw):
    clk = clk or Clock()
    post = Post(answers)
    client = OpenAICompatClient(BASE, KEY, DiskCache(tmp_path / "cache"), limiter=limiter,
                                post=post, get=kw.pop("get", None), sleep=clk.sleep,
                                clock=clk.clock, max_retries=3, max_json_attempts=3, **kw)
    return client, post, clk


GOOD_JSON = '{"ok": true, "word": "pong"}'


# ---- request shape ----------------------------------------------------------------------

def test_request_shape_for_a_json_call_on_a_reasoning_model(tmp_path):
    client, post, _ = make(tmp_path, [completion(GOOD_JSON)])
    out = client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert out.ok is True
    call = post.calls[0]
    assert call["url"] == BASE + "/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    p = call["json"]
    assert p["model"] == MODEL and p["messages"] == MESSAGES and p["stream"] is False
    assert p["max_tokens"] == 300 and p["temperature"] == 0 and "seed" in p
    assert p["reasoning_effort"] == "low"
    assert p["response_format"] == {"type": "json_object"}
    assert not ({"options", "format", "num_ctx", "num_predict"} & set(p))  # no Ollama shape


def test_reasoning_effort_is_sent_only_to_reasoning_models_and_only_valid_values(tmp_path):
    client, post, _ = make(tmp_path, [completion("hi")])
    client.chat(MESSAGES, "llama-3.1-8b-instant", options=OPTIONS)
    assert "reasoning_effort" not in post.calls[0]["json"]
    assert "response_format" not in post.calls[0]["json"]  # plain chat: no JSON mode
    with pytest.raises(ValueError, match="reasoning_effort"):
        client.chat(MESSAGES, MODEL, options={"max_tokens": 300, "reasoning_effort": "extreme"})


@pytest.mark.parametrize("opts", [{}, {"max_tokens": 401}, {"max_tokens": 0},
                                  {"max_tokens": "300"}, {"max_tokens": True}])
def test_max_tokens_is_required_and_capped_at_400(tmp_path, opts):
    client, post, _ = make(tmp_path, [completion("hi")])
    with pytest.raises(ValueError, match="max_tokens"):
        client.chat(MESSAGES, MODEL, options=opts)
    assert post.calls == []  # nothing was sent


def test_json_mode_can_be_switched_off(tmp_path):
    client, post, _ = make(tmp_path, [completion(GOOD_JSON)], json_mode="none")
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert "response_format" not in post.calls[0]["json"]


def test_embeddings_are_not_available_through_the_hosted_client(tmp_path):
    client, _, _ = make(tmp_path, [completion("x")])
    with pytest.raises(NotImplementedError):
        client.embed(["a"], "m")


def test_model_listing_reads_the_data_ids(tmp_path):
    seen = {}

    def get(url, headers=None, timeout=None):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return HTTP(200, {"data": [{"id": "openai/gpt-oss-20b"}, {"id": "openai/gpt-oss-120b"}]})

    client, _, _ = make(tmp_path, [completion("x")], get=get)
    assert client.list_models() == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert seen["url"] == BASE + "/models" and seen["auth"] == f"Bearer {KEY}"


# ---- usage and reasoning ----------------------------------------------------------------

def test_usage_tokens_are_recorded_and_reasoning_text_is_never_stored(tmp_path):
    secret = "PRIVATE-CHAIN-OF-THOUGHT-do-not-keep"
    reply = completion(GOOD_JSON, prompt=120, comp=80, reasoning=55,
                       message_extra={"reasoning": secret, "reasoning_content": secret})
    client, _, _ = make(tmp_path, [reply])
    out = client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert out == Pong(ok=True, word="pong")
    assert client.last_usage == {"prompt": 120, "completion": 80, "reasoning": 55, "total": 200}
    assert client.usage["total"] == 200 and client.usage["reasoning"] == 55
    assert client.usage["calls"] == 1
    stored = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "cache").rglob("*")
                     if p.is_file())
    assert "pong" in stored  # the final content is cached
    assert secret not in stored  # the reasoning text never reaches the cache


def test_a_cache_hit_costs_no_call_and_no_tokens(tmp_path):
    client, post, _ = make(tmp_path, [completion(GOOD_JSON)])
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert len(post.calls) == 1 and client.usage["calls"] == 1


def test_missing_usage_falls_back_to_the_estimate_and_is_counted(tmp_path):
    client, _, _ = make(tmp_path, [completion(GOOD_JSON, usage=False)])
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert client.usage["usage_missing"] == 1 and client.last_usage["total"] > 300


def test_cache_key_changes_with_model_prompt_and_options(tmp_path):
    client, post, _ = make(tmp_path, [completion(GOOD_JSON)])
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    client.chat_json(MESSAGES, "openai/gpt-oss-120b", Pong, options=OPTIONS)
    client.chat_json([{"role": "system", "content": "Another prompt."}, MESSAGES[1]], MODEL, Pong,
                     options=OPTIONS)
    client.chat_json(MESSAGES, MODEL, Pong, options={"max_tokens": 250, "reasoning_effort": "low"})
    client.chat_json(MESSAGES, MODEL, Pong, options={"max_tokens": 300, "reasoning_effort": "high"})
    assert len(post.calls) == 5
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)  # the first one again: a hit
    assert len(post.calls) == 5


def test_json_repair_loop_discards_invalid_replies_and_caches_only_the_valid_one(tmp_path):
    client, post, _ = make(tmp_path, [completion("not json"), completion('{"ok": "maybe"}'),
                                      completion(GOOD_JSON)])
    out = client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert out.word == "pong" and len(post.calls) == 3
    second = post.calls[1]["json"]["messages"]
    assert second[-2] == {"role": "assistant", "content": "not json"}
    assert "not valid for the required JSON schema" in second[-1]["content"]
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert len(post.calls) == 3  # served from the cache now


def test_json_that_never_validates_raises_and_is_not_cached(tmp_path):
    client, post, _ = make(tmp_path, [completion("{}")])
    with pytest.raises(JSONValidationError):
        client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    n = len(post.calls)
    with pytest.raises(JSONValidationError):
        client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert len(post.calls) == 2 * n


def test_an_empty_reply_cut_off_by_the_token_cap_is_a_bad_body_not_a_cached_answer(tmp_path):
    client, post, _ = make(tmp_path, [completion("", finish="length")])
    with pytest.raises(BadResponseError, match="finish_reason"):
        client.chat(MESSAGES, MODEL, options=OPTIONS)
    assert len(post.calls) > 1  # retried, then given up
    assert not list((tmp_path / "cache").rglob("*.json"))


# ---- headers, durations -------------------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [
    ("30", 30.0), ("2m59.56s", 179.56), ("6ms", 0.006), ("1h2m3s", 3723.0), ("1d", 86400.0),
    ("7.66s", 7.66), ("0s", 0.0)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", [None, "", "soon", "5 minutes", "m5"])
def test_parse_duration_rejects_what_it_cannot_read(bad):
    assert parse_duration(bad) is None


def test_rate_headers_requests_are_the_daily_limit_and_tokens_the_per_minute_limit():
    info = parse_rate_headers({
        "X-RateLimit-Limit-Requests": "1000", "x-ratelimit-remaining-requests": "999",
        "x-ratelimit-reset-requests": "2m59.56s", "X-Ratelimit-Limit-Tokens": "8000",
        "x-ratelimit-remaining-tokens": "7500", "x-ratelimit-reset-tokens": "6ms",
        "retry-after": "17"})
    assert info == RateInfo(rpd_limit=1000, rpd_remaining=999, rpd_reset_s=pytest.approx(179.56),
                            tpm_limit=8000, tpm_remaining=7500, tpm_reset_s=pytest.approx(0.006))
    assert parse_rate_headers({}) == RateInfo() and parse_rate_headers(None) == RateInfo()
    assert retry_after_seconds({"Retry-After": "17"}) == 17.0
    assert retry_after_seconds({}) is None


def test_retry_after_is_ignored_on_a_successful_response(tmp_path):
    reply = completion(GOOD_JSON)
    reply.headers = {"retry-after": "500"}
    lim_clock = Clock()
    lim = RateLimiter(60, 100000, 1000, 1000000, clock=lim_clock.clock, sleep=lim_clock.sleep,
                      utcnow=lambda: datetime(2026, 10, 20, tzinfo=timezone.utc))
    client, post, _ = make(tmp_path, [reply], limiter=lim, clk=lim_clock)
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert lim_clock.slept == [] and len(post.calls) == 1


# ---- the limiter --------------------------------------------------------------------------

def limiter(clk, state=None, now=None, **limits):
    base = {"rpm": 100, "tpm": 100000, "rpd": 10000, "tpd": 1000000}
    base.update(limits)
    now = now or [datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)]
    return RateLimiter(base["rpm"], base["tpm"], base["rpd"], base["tpd"], clock=clk.clock,
                       sleep=clk.sleep, utcnow=lambda: now[0], state_path=state)


def test_effective_limits_keep_a_20_percent_margin_and_85_percent_of_the_daily_tokens():
    lim = limiter(Clock(), rpm=30, tpm=8000, rpd=1000, tpd=200000)
    assert (lim.rpm_eff, lim.tpm_eff, lim.rpd_eff, lim.tpd_eff) == (24, 6400, 800, 170000)


def test_rpm_pacing_waits_for_the_oldest_request_to_leave_the_window():
    clk = Clock()
    lim = limiter(clk, rpm=10)  # 8 allowed per minute
    start = clk.t
    for _ in range(8):
        lim.after_call(lim.before_call(10), 10, None)
    assert clk.slept == []
    lim.before_call(10)  # the 9th must wait for the first to age out
    assert len(clk.slept) == 1 and 59.9 < clk.t - start < 60.2


def test_tpm_pacing_waits_until_enough_tokens_leave_the_window():
    clk = Clock()
    lim = limiter(clk, tpm=1000)  # 800 tokens per minute
    start = clk.t
    lim.after_call(lim.before_call(300), 300, None)
    clk.t += 10
    lim.after_call(lim.before_call(300), 300, None)
    assert clk.slept == []
    lim.before_call(300)  # 900 > 800: wait for the first (at t0) to age out, i.e. about 50 s more
    assert 59.9 < clk.t - start < 60.2


def test_pacing_uses_actual_usage_not_the_estimate():
    clk = Clock()
    lim = limiter(clk, tpm=1000)
    for _ in range(3):
        lim.after_call(lim.before_call(300), 100, None)  # estimated 300, really 100
    assert clk.slept == []  # 300 actual tokens in the window, not 900


def test_one_request_larger_than_the_minute_budget_is_refused_not_slept():
    with pytest.raises(LLMError, match="larger than the per-minute"):
        limiter(Clock(), tpm=1000).before_call(900)


def test_daily_token_budget_stops_the_run_with_a_quota_error():
    clk = Clock()
    lim = limiter(clk, tpm=100000, tpd=1000)  # 850 tokens a day
    lim.after_call(lim.before_call(100), 800, None)
    with pytest.raises(QuotaExhausted) as exc:
        lim.before_call(100)
    assert exc.value.tokens_today == 800 and "daily token budget" in exc.value.reason
    assert 0 < exc.value.reset_in_s <= 86400
    assert clk.slept == []  # it stops; it does not sleep until tomorrow


def test_daily_request_budget_stops_the_run():
    lim = limiter(Clock(), rpd=10)  # 8 requests a day
    for _ in range(8):
        lim.after_call(lim.before_call(10), 10, None)
    with pytest.raises(QuotaExhausted, match="daily request budget"):
        lim.before_call(10)


def test_the_day_counters_are_saved_and_reset_on_a_new_utc_date(tmp_path):
    clk, state = Clock(), tmp_path / "q.json"
    now = [datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)]
    lim = limiter(clk, state=state, now=now, tpd=1000)
    lim.after_call(lim.before_call(100), 800, None)
    again = limiter(clk, state=state, now=now, tpd=1000)  # a restart on the same day
    assert again.day_tokens == 800 and again.day_requests == 1
    with pytest.raises(QuotaExhausted):
        again.before_call(100)
    now[0] += timedelta(hours=3)  # past UTC midnight
    again.before_call(100)  # a fresh budget
    assert again.day_tokens == 0 and again.snapshot()["date"] == "2026-10-21"
    assert limiter(clk, state=state, now=now, tpd=1000).day_tokens == 0  # state file followed


def test_response_headers_can_close_the_day_and_slow_the_minute():
    clk = Clock()
    lim = limiter(clk)
    lim.after_call(lim.before_call(100), 100, RateInfo(tpm_remaining=50, tpm_reset_s=7.0))
    lim.after_call(lim.before_call(100), 100, None)
    assert 6.9 < sum(clk.slept) < 7.2  # waited for the token window the API reported
    lim.after_call(lim.before_call(100), 100, RateInfo(rpd_remaining=0, rpd_reset_s=300.0))
    with pytest.raises(QuotaExhausted) as exc:
        lim.before_call(100)
    assert exc.value.reset_in_s == 300.0 and "no daily requests left" in exc.value.reason


def test_a_failed_call_counts_a_request_but_no_tokens():
    lim = limiter(Clock())
    lim.failed(lim.before_call(300))
    assert lim.day_requests == 1 and lim.day_tokens == 0


def test_limiter_rejects_nonsense_limits():
    with pytest.raises(ValueError):
        RateLimiter(0, 1, 1, 1)


def test_the_client_paces_every_call_and_reports_the_day(tmp_path):
    clk = Clock()
    lim = limiter(clk, rpm=10)  # 8 a minute
    client, post, _ = make(tmp_path, [completion(GOOD_JSON, prompt=10, comp=10)], limiter=lim,
                           clk=clk)
    for i in range(10):
        client.chat([{"role": "user", "content": f"m{i}"}], MODEL, options=OPTIONS)
    assert len(post.calls) == 10 and sum(clk.slept) >= 59.9  # calls 9 and 10 waited
    snap = lim.snapshot()
    assert snap["requests_today"] == 10 and snap["tokens_today"] == 200 and snap["http_429"] == 0


# ---- 429, errors, scrubbing --------------------------------------------------------------

def test_429_with_retry_after_is_honoured_not_replaced_by_backoff(tmp_path):
    clk = Clock()
    lim = limiter(clk)
    answers = [HTTP(429, {"error": "slow down"}, headers={"Retry-After": "7"}),
               completion(GOOD_JSON)]
    client, post, _ = make(tmp_path, answers, limiter=lim, clk=clk)
    out = client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert out.ok and len(post.calls) == 2
    assert clk.slept == [7.0]  # exactly the server's number
    assert lim.n429 == 1
    assert lim.day_requests == 2  # the rejected call still counts as a request


def test_429_with_a_long_retry_after_is_a_quota_stop_not_a_wait(tmp_path):
    clk = Clock()
    lim = limiter(clk)
    client, post, _ = make(tmp_path, [HTTP(429, {}, headers={"retry-after": "3600"})],
                           limiter=lim, clk=clk)
    with pytest.raises(QuotaExhausted) as exc:
        client.chat(MESSAGES, MODEL, options=OPTIONS)
    assert exc.value.reset_in_s == 3600.0 and len(post.calls) == 1 and clk.slept == []


def test_429_without_retry_after_backs_off_and_retries(tmp_path):
    client, post, clk = make(tmp_path, [HTTP(429, {}), completion("fine")])
    assert client.chat(MESSAGES, MODEL, options=OPTIONS) == "fine"
    assert len(post.calls) == 2 and len(clk.slept) == 1


def test_5xx_is_retried_and_401_404_are_not(tmp_path):
    client, post, _ = make(tmp_path, [HTTP(503, {}), completion("ok")])
    assert client.chat(MESSAGES, MODEL, options=OPTIONS) == "ok" and len(post.calls) == 2
    client, post, _ = make(tmp_path / "a", [HTTP(401, {"error": f"bad key {KEY}"})])
    with pytest.raises(AuthError) as exc:
        client.chat(MESSAGES, MODEL, options=OPTIONS)
    assert len(post.calls) == 1 and KEY not in str(exc.value)
    client, post, _ = make(tmp_path / "b", [HTTP(404, {})])
    with pytest.raises(ModelNotFoundError):
        client.chat(MESSAGES, MODEL, options=OPTIONS)
    assert len(post.calls) == 1


def test_scrub_text_removes_keys_headers_and_urls():
    text = (f"failed: Authorization: Bearer {KEY} at https://{HOST}/openai/v1/chat/completions "
            f"key=sk-abcdefghijklmnop host {HOST} Authorization: abc123 gsk_ZZZZZZZZZZZZ")
    out = scrub_text(text, [BASE, HOST, KEY])
    for secret in (KEY, HOST, "https://", "sk-abcdefghijklmnop", "gsk_ZZZZZZZZZZZZ", "abc123",
                   "Bearer "):
        assert secret not in out, secret
    assert "<redacted>" in out and scrub_text("plain words") == "plain words"


@pytest.mark.parametrize("make_answer", [
    lambda: HTTP(400, {"error": f"bad request to https://{HOST}/x with Bearer {KEY}"}),
    lambda: HTTP(404, {"error": f"https://{HOST} unknown"}),
    lambda: HTTP(401, {"error": f"Authorization: Bearer {KEY}"}),
    lambda: HTTP(200, None, text=f"<html>sign in at https://{HOST}/login {KEY}</html>"),
    lambda: requests.ConnectionError(f"HTTPSConnectionPool(host='{HOST}', port=443): {KEY}"),
    lambda: requests.exceptions.InvalidURL(f"Invalid URL 'https://{HOST}': {KEY}"),
])
def test_no_error_text_contains_the_key_the_host_or_the_url(tmp_path, make_answer):
    client, _, _ = make(tmp_path, [make_answer()])
    with pytest.raises(LLMError) as exc:
        client.chat(MESSAGES, MODEL, options=OPTIONS)
    chain, e = [], exc.value
    while e is not None:
        chain.append(f"{e!s} {e!r}")
        e = e.__cause__
    text = " ".join(chain)
    for secret in (KEY, HOST, BASE, "Bearer "):
        assert secret not in text, secret


def test_the_settings_object_never_shows_the_key_or_the_url(tmp_path, monkeypatch):
    env(monkeypatch)
    s = load_llm_settings(tmp_path / "none.env")
    assert KEY not in repr(s) and HOST not in repr(s) and BASE not in repr(s)


def test_debug_records_are_scrubbed_of_keys_and_urls(tmp_path):
    from src.process.enrich import enrich_chunk
    from tests.test_enrich import CHUNK, HOLDOUT, REGISTRY, TITLE

    leaky = {"context_note": "x", "aspect": f"see https://{HOST}/a?key={KEY}"}  # invalid -> repair
    good = {"context_note": "A blog post about Munnar describing tea-garden viewpoints.",
            "place_type": "hill_station", "season_tags": ["all_year"], "aspect": "scenery",
            "is_warning": False, "mentioned_places": [f"Kochi https://{HOST}/z {KEY}"]}
    client, _, _ = make(tmp_path, [completion(json.dumps(leaky)), completion(json.dumps(good))])
    records = []
    enrich_chunk(client, MODEL, CHUNK, TITLE, REGISTRY, HOLDOUT, debug=records.append)
    text = json.dumps(records)
    assert records and KEY not in text and HOST not in text and "https://" not in text
    assert any(r["kind"] == "attempt" and r["validation_error"] for r in records)


# ---- run_batch with quota / auth, the exit code ---------------------------------------------

def test_quota_and_auth_errors_stop_the_batch_without_counting_as_failures(tmp_path):
    from src.process import enrich

    chunks = {f"c{i}": {"chunk_id": f"c{i}"} for i in range(5)}
    ck = tmp_path / "ck.jsonl"

    def ok(chunk):
        return ({"aspect": "other"}, {"source": "llm", "reason": None, "dropped": {}})

    n = {"i": 0}

    def process(chunk):
        n["i"] += 1
        if n["i"] == 3:
            raise QuotaExhausted("daily token budget used", tokens_today=850)
        return ok(chunk)

    log = []
    with pytest.raises(QuotaExhausted):
        enrich.run_batch(list(chunks), chunks, process, ck, MODEL, log=log.append)
    assert [r["chunk_id"] for r in enrich.read_jsonl(ck)] == ["c0", "c1"] and log == []
    with pytest.raises(AuthError):
        enrich.run_batch(list(chunks), chunks, lambda c: (_ for _ in ()).throw(AuthError("no")),
                         tmp_path / "ck2.jsonl", MODEL, log=log.append)
    assert log == []  # neither was logged as a retryable failure


def test_run_batch_records_the_tokens_each_chunk_cost(tmp_path):
    from src.process import enrich

    chunks = {"c0": {"chunk_id": "c0"}, "c1": {"chunk_id": "c1"}}
    meter = Counter()

    def process(chunk):
        meter.update(prompt=100, completion=40, reasoning=10, total=140)
        return ({"aspect": "other"}, {"source": "llm", "reason": None, "dropped": {}})

    enrich.run_batch(list(chunks), chunks, process, tmp_path / "ck.jsonl", MODEL,
                     usage=lambda: dict(meter))
    recs = list(enrich.read_jsonl(tmp_path / "ck.jsonl"))
    assert all(r["prov"]["usage"] == {"prompt": 100, "completion": 40, "reasoning": 10,
                                      "total": 140} for r in recs)


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"script_{name}", Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_enrich_script_exit_codes_are_distinct_and_quota_prints_the_resume_command(
        tmp_path, monkeypatch, capsys):
    script = load_script("enrich")
    assert len({script.EXIT_STOP, script.EXIT_CONFIG, script.EXIT_QUOTA, 0, 1, 130}) == 6
    assert script.EXIT_QUOTA == 5
    clk = Clock()
    lim = limiter(clk, tpd=1000)
    lim.after_call(lim.before_call(100), 800, None)

    class StubClient:
        limiter = lim
        usage: Counter = Counter()

        @classmethod
        def from_settings(cls, **kw):
            return cls()

    env(monkeypatch)
    settings = load_llm_settings(tmp_path / "none.env")
    monkeypatch.setattr(script, "load_llm_settings", lambda: settings)
    monkeypatch.setattr(script, "OpenAICompatClient", StubClient)
    monkeypatch.setattr(script, "env_duplicate_keys", lambda: [])
    monkeypatch.setattr(script, "scope", lambda *a: (["x"], {"skipped": {}}))
    monkeypatch.setattr(script, "ENRICHED", tmp_path)

    def boom(*a, **k):
        raise QuotaExhausted("daily token budget used (850 of 850 allowed)", tokens_today=850,
                             requests_today=12, reset_in_s=7200)

    monkeypatch.setattr(script, "run_batch", boom)
    code = script.main(["--full"])
    out = capsys.readouterr().out
    assert code == 5 and "DAILY QUOTA USED UP" in out and "Tokens used today 800" in out
    assert "in about 2.0 h" in out and "re-run the same command to resume" in out
    assert KEY not in out and HOST not in out and "https://" not in out
    monkeypatch.setattr(script, "run_batch", lambda *a, **k: (_ for _ in ()).throw(AuthError("x")))
    assert script.main(["--full"]) == 4
    monkeypatch.setattr(script, "run_batch",
                        lambda *a, **k: (_ for _ in ()).throw(script.StopRun("5 failures")))
    assert script.main(["--full"]) == 3


def test_enrich_script_refuses_to_run_when_a_limit_is_unset(tmp_path, monkeypatch, capsys):
    script = load_script("enrich")
    env(monkeypatch)
    monkeypatch.delenv("LLM_TPD")
    monkeypatch.setattr(script, "load_llm_settings", lambda: load_llm_settings(tmp_path / "x.env"))
    monkeypatch.setattr(script, "scope", lambda *a: (["x"], {"skipped": {}}))
    assert script.main(["--full"]) == 4
    out = capsys.readouterr().out
    assert "REFUSING TO RUN" in out and "LLM_TPD" in out and KEY not in out


def test_checkpoint_files_are_per_prompt_version_and_model():
    script = load_script("enrich")
    a = script.checkpoint_path("openai/gpt-oss-20b")
    b = script.checkpoint_path("openai/gpt-oss-120b")
    assert a != b and "enrich-v4" in a.name and "gpt-oss-20b" in a.name
    assert "/" not in a.name


# ---- settings ---------------------------------------------------------------------------------

ALL = {"LLM_API_BASE_URL": BASE, "LLM_API_KEY": KEY, "GENERATOR_MODEL": MODEL,
       "JUDGE_MODEL": "openai/gpt-oss-120b", "LLM_RPM": "30", "LLM_TPM": "8000",
       "LLM_RPD": "1000", "LLM_TPD": "200000"}


def env(monkeypatch, **override):
    for k in (*ALL, "LLM_API_URL", "LLM_JSON_MODE", "LLM_RETRY_BUDGET_S"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**ALL, **override}.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


def test_settings_load_and_the_old_url_name_is_accepted_as_an_alias(tmp_path, monkeypatch):
    env(monkeypatch)
    s = load_llm_settings(tmp_path / "none.env")
    assert (s.rpm, s.tpm, s.rpd, s.tpd) == (30, 8000, 1000, 200000)
    assert s.generator_model == MODEL and s.json_mode == "json_object"
    env(monkeypatch, LLM_API_BASE_URL=None, LLM_API_URL=BASE + "/")
    assert load_llm_settings(tmp_path / "none.env").base_url == BASE  # alias, trailing slash cut


@pytest.mark.parametrize("name", ["LLM_API_KEY", "LLM_RPM", "LLM_TPM", "LLM_RPD", "LLM_TPD",
                                  "GENERATOR_MODEL", "JUDGE_MODEL"])
def test_any_unset_key_means_refuse_to_run_and_the_error_lists_names_only(
        tmp_path, monkeypatch, name):
    env(monkeypatch, **{name: None})
    with pytest.raises(ConfigError) as exc:
        load_llm_settings(tmp_path / "none.env")
    assert name in str(exc.value) and KEY not in str(exc.value) and BASE not in str(exc.value)


def test_a_missing_url_is_reported_by_its_main_name(tmp_path, monkeypatch):
    env(monkeypatch, LLM_API_BASE_URL=None)
    with pytest.raises(ConfigError, match="LLM_API_BASE_URL"):
        load_llm_settings(tmp_path / "none.env")


@pytest.mark.parametrize("bad", [{"LLM_RPM": "0"}, {"LLM_TPD": "-5"}, {"LLM_TPM": "lots"},
                                 {"LLM_API_BASE_URL": "not-a-url"},
                                 {"JUDGE_MODEL": MODEL}, {"LLM_JSON_MODE": "yaml"}])
def test_nonsense_settings_are_refused(tmp_path, monkeypatch, bad):
    env(monkeypatch, **bad)
    with pytest.raises(ConfigError):
        load_llm_settings(tmp_path / "none.env")


def test_duplicate_env_keys_are_reported_by_name_only(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=1\nGENERATOR_MODEL=old\n# JUDGE_MODEL=x\nJUDGE_MODEL=a\nGENERATOR_MODEL=new\n"
                 "JUDGE_MODEL=b\n", encoding="utf-8")
    assert env_duplicate_keys(f) == ["GENERATOR_MODEL", "JUDGE_MODEL"]
    assert env_duplicate_keys(tmp_path / "missing.env") == []


# ---- the shrunken prompt and the scope rules --------------------------------------------------

def test_prompt_plus_longest_chunk_plus_max_tokens_stays_under_1500_tokens():
    from src.process import enrich

    longest = {**enrich_chunk_stub(), "text": "w" * 1600}  # longer than any frozen chunk (1,507)
    msgs = enrich.build_messages(longest, "T" * 120)
    assert enrich.estimate_tokens(msgs) + enrich.ENRICH_OPTIONS["max_tokens"] < 1500
    assert enrich.ENRICH_OPTIONS["max_tokens"] <= 400


def enrich_chunk_stub():
    return {"chunk_id": "c", "place": "Munnar", "state": "KL", "source_type": "wikivoyage"}


@pytest.mark.skipif(not Path("data/processed/chunks.jsonl").exists(),
                    reason="frozen corpus not present in this checkout")
def test_every_real_chunk_fits_prompt_plus_reply_under_1500_tokens():
    from src.process import enrich

    with open("data/processed/chunks.jsonl", encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]
    worst = max(enrich.estimate_tokens(enrich.build_messages(c, "T" * 120)) for c in chunks)
    assert worst + enrich.ENRICH_OPTIONS["max_tokens"] < 1500, worst


def toy_corpus():
    out = []
    for place in ("Munnar", "Hampi", "Ooty"):
        for st in ("blog", "reference", "wikivoyage"):
            for k in range(3):
                out.append({"chunk_id": f"{place[:2]}{st[:2]}{k}", "place": place,
                            "source_type": st, "url": f"https://x/{place}/{st}", "state": "KL",
                            "date": None, "date_kind": "none", "revised_date": None,
                            "text": f"{place} {st} {k}"})
    return out


def test_scope_flags_skip_and_count_instead_of_dropping_silently():
    from src.process import enrich

    chunks = toy_corpus()
    pool, skipped = enrich.candidate_pool(chunks, excluded_places=["Munnar"], eligible_only=False,
                                          include_blog=False)
    assert len(pool) == 18 and skipped == {"blog_not_sent": 9}  # blog off by default
    pool, skipped = enrich.candidate_pool(chunks, excluded_places=["Munnar"], eligible_only=True,
                                          include_blog=False)
    assert skipped == {"excluded_place": 9, "blog_not_sent": 6} and len(pool) == 12
    assert not any(c["place"] == "Munnar" or c["source_type"] == "blog" for c in pool)
    pool, skipped = enrich.candidate_pool(chunks, excluded_places=["Munnar"], eligible_only=True,
                                          include_blog=True)
    assert skipped == {"excluded_place": 9} and sum(c["source_type"] == "blog" for c in pool) == 6
    assert sum(skipped.values()) + len(pool) == len(chunks)


def test_assemble_labelled_gives_unprocessed_out_of_scope_chunks_a_labelled_rule_fill():
    from src.process import enrich
    from tests.test_enrich import GOOD, REGISTRY

    chunks = toy_corpus()
    titles = {c["url"]: "A title" for c in chunks}
    sent = [c for c in chunks if c["place"] != "Munnar" and c["source_type"] != "blog"]
    results = {c["chunk_id"]: {"fields": dict(GOOD),
                               "prov": {"source": "llm", "reason": None}} for c in sent}
    kw = {"excluded_places": ["Munnar"], "eligible_only": True, "include_blog": False}
    rows, prov = enrich.assemble_labelled(chunks, results, titles, REGISTRY, **kw)
    assert [r["chunk_id"] for r in rows] == [c["chunk_id"] for c in chunks]  # frozen order
    why = Counter((p["source"], p["reason"]) for p in prov)
    assert why == Counter({("llm", None): 12, ("rule_only", "excluded_place"): 9,
                           ("rule_only", "blog_not_sent"): 6})
    munnar = next(r for r in rows if r["place"] == "Munnar")
    assert munnar["context_note"].startswith(("Travel blog post", "Encyclopedia article",
                                              "Wikivoyage guide"))  # the template note
    with pytest.raises(KeyError, match="in scope but was not processed"):
        enrich.assemble_labelled(chunks, {k: v for k, v in list(results.items())[1:]}, titles,
                                 REGISTRY, **kw)


def test_projection_is_limited_by_the_tighter_daily_budget():
    from src.process import enrich

    pr = enrich.project_days(mean_tokens=1000, mean_calls=1.2, n_chunks=1696, tpd=200000, rpd=1000)
    assert pr["tokens"] == 1_696_000 and pr["requests"] == 2035
    assert pr["days_by_tokens"] == pytest.approx(1_696_000 / 170000)
    assert pr["days_by_requests"] == pytest.approx(2035.2 / 800)
    assert pr["days"] == pr["days_by_tokens"] and pr["limited_by"] == "tokens (TPD)"
    assert "PROJECTION" in enrich.format_projection("all", pr)


# ---- the API rejecting a JSON-mode reply (HTTP 400 "Failed to generate JSON") -------------

def json_mode_400(text='{"ok": tr'):
    return HTTP(400, {"error": {"message": "Failed to generate JSON. Please adjust your prompt.",
                                "type": "invalid_request_error", "code": "json_validate_failed",
                                "failed_generation": text}})


def test_a_json_mode_rejection_is_a_repair_turn_not_an_outage(tmp_path):
    clk = Clock()
    lim = limiter(clk)
    client, post, _ = make(tmp_path, [json_mode_400('{"ok": tr'), completion(GOOD_JSON)],
                           limiter=lim, clk=clk)
    seen = []
    out = client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS,
                           on_attempt=lambda n, raw, err: seen.append((n, raw, err)))
    assert out.ok is True and len(post.calls) == 2
    repair = post.calls[1]["json"]["messages"]
    assert repair[-2] == {"role": "assistant", "content": '{"ok": tr'}  # what the model produced
    assert "not valid JSON" in repair[-1]["content"]
    assert seen[0][0] == 1 and seen[0][1] == '{"ok": tr' and "invalid JSON" in seen[0][2]
    assert seen[1] == (2, GOOD_JSON, None)
    # the rejected call was billed as far as we can tell: its estimate counts against the day
    assert lim.day_requests == 2 and lim.day_tokens > 0


def test_a_json_mode_rejection_that_repeats_gives_up_as_invalid_json_not_a_transport_failure(
        tmp_path):
    client, post, _ = make(tmp_path, [json_mode_400()])
    with pytest.raises(JSONValidationError):
        client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert len(post.calls) == 3  # max_json_attempts
    assert not list((tmp_path / "cache").rglob("*.json"))  # nothing cached


def test_the_rejected_text_is_scrubbed_and_other_400s_stay_errors(tmp_path):
    seen = []
    client, _, _ = make(tmp_path, [json_mode_400(f"see https://{HOST}/x {KEY}"),
                                   completion(GOOD_JSON)])
    client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS,
                     on_attempt=lambda n, raw, err: seen.append(raw))
    assert KEY not in seen[0] and HOST not in seen[0] and "https://" not in seen[0]
    client, post, _ = make(tmp_path / "b", [HTTP(400, {"error": {"message": "bad model"}})])
    with pytest.raises(LLMError, match="HTTP 400") as exc:
        client.chat_json(MESSAGES, MODEL, Pong, options=OPTIONS)
    assert not isinstance(exc.value, JSONValidationError) and len(post.calls) == 1


def test_enrich_chunk_falls_back_when_the_api_keeps_rejecting_the_json(tmp_path):
    from src.process.enrich import enrich_chunk, rule_based
    from tests.test_enrich import CHUNK, HOLDOUT, REGISTRY, TITLE

    client, _, _ = make(tmp_path, [json_mode_400('{"context_note": "cut off')])
    records = []
    fields, prov = enrich_chunk(client, MODEL, CHUNK, TITLE, REGISTRY, HOLDOUT,
                                debug=records.append)
    assert prov["source"] == "fallback" and prov["reason"] == "json_invalid"
    assert fields == rule_based(CHUNK, TITLE)
    assert [r["kind"] for r in records] == ["attempt", "attempt", "attempt", "final"]
    assert records[0]["raw"] == '{"context_note": "cut off'  # kept for diagnosis (untrusted)


def test_the_daily_token_share_is_configurable_and_defaults_to_85_percent():
    assert limiter(Clock(), tpd=200000).tpd_eff == 170000
    lim = RateLimiter(30, 8000, 1000, 200000, tpd_share=0.95)
    assert lim.tpd_eff == pytest.approx(190000)
