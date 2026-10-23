"""OpenAICompatClient.chat_stream (D103): offline, with a fake streaming transport."""
import json

import pytest
import requests

from src.common.cache import DiskCache
from src.common.llm import AuthError, BadResponseError, LLMError, OpenAICompatClient, RateLimiter

BASE = "https://api.fake-provider.test/openai/v1"
KEY = "gsk_TESTKEY0123456789abcdef"
MSGS = [{"role": "user", "content": "plan a trip"}]


def sse(pieces, usage=None, reasoning=False):
    lines = []
    for p in pieces:
        delta = {"content": p}
        if reasoning:
            delta["reasoning"] = "secret chain of thought"
        lines.append("data: " + json.dumps({"choices": [{"delta": delta}]}))
    if usage:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return lines


class Resp:
    def __init__(self, lines=(), status=200, boom=False):
        self.status_code, self.headers, self.text = status, {}, ""
        self._lines, self._boom = list(lines), boom

    def iter_lines(self, decode_unicode=True):
        for i, line in enumerate(self._lines):
            yield line
            if self._boom and i == 0:
                raise requests.exceptions.ConnectionError("reset")

    def json(self):
        return {}


class Post:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def __call__(self, url, json=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "json": json, "stream": stream})
        return self.resp


def make(tmp_path, resp, cap=2000):
    post = Post(resp)
    limiter = RateLimiter(30, 8000, 1000, 200000)
    client = OpenAICompatClient(BASE, KEY, DiskCache(tmp_path / "c"), limiter=limiter,
                                post=post, sleep=lambda s: None, max_tokens_cap=cap, seed=42)
    return client, post, limiter


OPTS = {"max_tokens": 1500, "reasoning_effort": "low"}


def test_stream_yields_pieces_and_caches_the_text_for_chat_too(tmp_path):
    usage = {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420}
    client, post, limiter = make(tmp_path, Resp(sse(["Day ", "1: ", "Hampi"], usage)))
    pieces = list(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))
    assert pieces == ["Day ", "1: ", "Hampi"]
    wire = post.calls[0]["json"]
    assert post.calls[0]["stream"] is True and wire["stream"] is True
    assert wire["stream_options"] == {"include_usage": True} and wire["max_tokens"] == 1500
    assert client.last_usage["total"] == 420 and limiter.day_tokens == 420
    # the same entry serves a later non-streamed call and a replay, with no new request
    assert client.chat(MSGS, "openai/gpt-oss-20b", options=OPTS) == "Day 1: Hampi"
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS,
                                      replay_chars=4)) == "Day 1: Hampi"
    assert len(post.calls) == 1


def test_missing_usage_keeps_the_estimate_charged_and_is_counted(tmp_path):
    client, post, limiter = make(tmp_path, Resp(sse(["ok"])))
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS)) == "ok"
    assert client.usage["usage_missing"] == 1 and limiter.day_tokens > 1500


def test_reasoning_text_is_never_yielded(tmp_path):
    client, _, _ = make(tmp_path, Resp(sse(["answer"], reasoning=True)))
    out = list(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))
    assert out == ["answer"] and "secret" not in "".join(out)


def test_a_broken_stream_raises_and_caches_nothing(tmp_path):
    client, post, limiter = make(tmp_path, Resp(sse(["a", "b"]), boom=True))
    with pytest.raises(LLMError, match="stream broke"):
        list(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))
    ok = Post(Resp(sse(["fresh"])))
    client._post = ok
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS)) == "fresh"
    assert len(ok.calls) == 1  # nothing had been cached by the failed attempt


def test_an_empty_stream_is_an_error_and_not_cached(tmp_path):
    client, post, _ = make(tmp_path, Resp(sse([])))
    with pytest.raises(BadResponseError):
        list(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))
    assert client.cache.get("llm_chat", "openai/gpt-oss-20b", {"x": 1}) is None


def test_auth_failure_raises_before_any_text(tmp_path):
    client, _, _ = make(tmp_path, Resp(status=401))
    with pytest.raises(AuthError):
        next(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))


def test_the_writer_cap_is_a_per_client_setting_and_the_default_stays_400(tmp_path):
    big = {"max_tokens": 2000, "reasoning_effort": "low"}
    client, _, _ = make(tmp_path, Resp(sse(["x"])), cap=2000)
    assert list(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=big)) == ["x"]
    small, _, _ = make(tmp_path / "s", Resp(sse(["x"])), cap=400)
    with pytest.raises(ValueError, match="max_tokens"):
        list(small.chat_stream(MSGS, "openai/gpt-oss-20b", options=big))


# ---- D115: UTF-8 stream decoding and cache=False ---------------------------------------------

def real_sse_response(lines, charset=None):
    """A real requests.Response the way the adapter builds it: a text/event-stream body with no
    charset in the content type gets ISO-8859-1 (requests.utils.get_encoding_from_headers)."""
    import io

    body = "".join(line + chr(10) for line in lines).encode("utf-8")
    r = requests.Response()
    r.status_code = 200
    r.headers["content-type"] = "text/event-stream" + (f"; charset={charset}" if charset else "")
    r.raw = io.BytesIO(body)
    r.encoding = requests.utils.get_encoding_from_headers(r.headers)
    return r


def raw_utf8_lines(text):
    event = {"choices": [{"delta": {"content": text}}]}
    return ["data: " + json.dumps(event, ensure_ascii=False), "data: [DONE]"]


def test_a_stream_without_a_charset_is_decoded_as_utf_8_not_latin_1(tmp_path):
    text = "6AM" + chr(0x2013) + "12:30PM " + chr(0x260F) + " +91 44"
    resp = real_sse_response(raw_utf8_lines(text))
    assert resp.encoding == "ISO-8859-1"  # what requests assumes for this content type
    client, _, _ = make(tmp_path, resp)
    out = "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS))
    assert out == text
    assert not any(0x80 <= ord(ch) <= 0x9F for ch in out)  # no latin-1 control characters
    # the cached copy is the clean text too, and a replay returns it
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS)) == text


def test_the_old_latin_1_decoding_is_what_produced_the_garbled_characters():
    text = "6AM" + chr(0x2013) + "12"
    garbled = text.encode("utf-8").decode("latin-1")
    assert garbled == "6AM" + chr(0xE2) + chr(0x80) + chr(0x93) + "12"
    assert garbled.encode("latin-1").decode("utf-8") == text


def test_cache_false_neither_reads_nor_writes_the_cache_in_a_stream(tmp_path):
    client, post, _ = make(tmp_path, Resp(sse(["one"])))
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS,
                                      cache=False)) == "one"
    assert list((tmp_path / "c").rglob("*.json")) == []  # nothing was stored
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS,
                                      cache=False)) == "one"
    assert len(post.calls) == 2  # and nothing was read
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS)) == "one"
    assert len(list((tmp_path / "c").rglob("*.json"))) == 1  # a normal call does store
    assert "".join(client.chat_stream(MSGS, "openai/gpt-oss-20b", options=OPTS,
                                      cache=False)) == "one"
    assert len(post.calls) == 4  # an existing entry is ignored by cache=False


class JsonResp(Resp):
    def json(self):
        return {"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 9}}


def test_cache_false_neither_reads_nor_writes_the_cache_in_chat(tmp_path):
    client, post, _ = make(tmp_path, JsonResp())
    assert client.chat(MSGS, "openai/gpt-oss-20b", options=OPTS, cache=False) == "hello"
    assert list((tmp_path / "c").rglob("*.json")) == []
    assert client.chat(MSGS, "openai/gpt-oss-20b", options=OPTS) == "hello"
    assert len(list((tmp_path / "c").rglob("*.json"))) == 1
    assert client.chat(MSGS, "openai/gpt-oss-20b", options=OPTS, cache=False) == "hello"
    assert len(post.calls) == 3  # the stored entry was not used
