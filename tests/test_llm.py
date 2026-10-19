import importlib.util
import json
from pathlib import Path

import pytest
import requests
from pydantic import BaseModel

from src.common.cache import DiskCache
from src.common.llm import (
    BadResponseError,
    JSONValidationError,
    LLMError,
    ModelNotFoundError,
    OllamaClient,
)


class FakeResp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def ok(content):
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


def make(tmp_path, *items, **kw):
    post = Script(*items)
    sleeps = []
    client = OllamaClient(
        "http://host:11434/", DiskCache(tmp_path), post=post, sleep=sleeps.append, **kw
    )
    return client, post, sleeps


MSGS = [{"role": "user", "content": "hi"}]


def test_chat_basic_and_request_shape(tmp_path):
    client, post, _ = make(tmp_path, ok("hello"))
    assert client.chat(MSGS, "m1") == "hello"
    url, kw = post.calls[0]
    assert url == "http://host:11434/api/chat"
    assert kw["json"]["stream"] is False and kw["json"]["options"]["temperature"] == 0
    assert kw["timeout"][1] >= 300  # long read timeout
    assert "ngrok-skip-browser-warning" not in kw["headers"]


def test_cache_hit_makes_zero_calls_even_after_host_change(tmp_path):
    client, post, _ = make(tmp_path, ok("hello"))
    client.chat(MSGS, "m1")
    other = OllamaClient("https://abc.ngrok-free.app", DiskCache(tmp_path), post=Script())
    assert other.chat(MSGS, "m1") == "hello"  # Script() has no items: a call would raise
    assert len(post.calls) == 1
    assert OllamaClient("http://x", DiskCache(tmp_path), post=Script(ok("z"))).chat(
        MSGS, "other-model"
    ) == "z"  # different model is a different key


def test_ngrok_header_added_only_for_ngrok(tmp_path):
    post = Script(ok("a"))
    OllamaClient("https://abc.ngrok-free.app", DiskCache(tmp_path), post=post).chat(MSGS, "m")
    assert post.calls[0][1]["headers"]["ngrok-skip-browser-warning"] == "1"


def test_retry_then_success_with_backoff(tmp_path):
    client, post, sleeps = make(
        tmp_path, requests.ConnectionError("x"), FakeResp(503, text="busy"), FakeResp(429), ok("ok")
    )
    assert client.chat(MSGS, "m") == "ok"
    assert len(post.calls) == 4 and len(sleeps) == 3
    assert sleeps[0] < sleeps[2]  # exponential growth


def test_timeout_exhausted_raises_and_caches_nothing(tmp_path):
    client, post, _ = make(tmp_path, *[requests.Timeout("t")] * 3, max_retries=2)
    with pytest.raises(LLMError, match="Giving up"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 3
    assert not list(tmp_path.rglob("*.json"))


def test_4xx_not_retried(tmp_path):
    client, post, sleeps = make(tmp_path, FakeResp(400, text="bad request"))
    with pytest.raises(LLMError, match="HTTP 400"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 1 and sleeps == []


def test_404_clear_model_error(tmp_path):
    client, post, _ = make(tmp_path, FakeResp(404, text="model not found"))
    with pytest.raises(ModelNotFoundError) as exc:
        client.chat(MSGS, "qwen")
    assert "model or endpoint not found" in str(exc.value)
    assert "check that the model is pulled and the URL is right" in str(exc.value)
    assert len(post.calls) == 1


def test_interstitial_200_is_retried_and_never_cached(tmp_path):
    html = FakeResp(200, text="<html>You are about to visit ngrok</html>")
    client, post, _ = make(tmp_path, html, html, ok("real"))
    assert client.chat(MSGS, "m") == "real"
    assert len(post.calls) == 3
    cached = list((tmp_path / "llm_chat").rglob("*.json"))
    assert len(cached) == 1 and "real" in cached[0].read_text(encoding="utf-8")


def test_interstitial_200_forever_raises_bad_response(tmp_path):
    html = FakeResp(200, text="<html>interstitial</html>")
    client, _, _ = make(tmp_path, html, html, max_retries=1)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")
    assert not list(tmp_path.rglob("*.json"))


def test_200_json_without_message_is_bad_response(tmp_path):
    client, _, _ = make(tmp_path, FakeResp(200, {"hello": 1}), max_retries=0)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")


class Plan(BaseModel):
    days: int
    city: str


def test_json_valid_is_cached_and_second_call_skips_network(tmp_path):
    client, post, _ = make(tmp_path, ok('{"days": 3, "city": "Hampi"}'))
    first = client.chat_json(MSGS, "m", Plan)
    again = client.chat_json(MSGS, "m", Plan)
    assert first == again == Plan(days=3, city="Hampi")
    assert len(post.calls) == 1
    assert post.calls[0][1]["json"]["format"] == "json"


def test_json_invalid_then_repaired(tmp_path):
    client, post, _ = make(
        tmp_path, ok("not json"), ok('{"days": "x"}'), ok('{"days": 2, "city": "A"}')
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
    client, post, _ = make(tmp_path, ok("x"), ok("y"), ok("z"), max_json_attempts=3)
    with pytest.raises(JSONValidationError):
        client.chat_json(MSGS, "m", Plan)
    assert len(post.calls) == 3
    assert not list(tmp_path.rglob("*.json"))


# ---- merged from tests/test_llm_resilience.py: hardening tests for llm.py (bad bodies, transport
# errors, time budget, seed, keys).

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
    client, _, _ = make(tmp_path, FakeResp(200, body), max_retries=0)
    with pytest.raises(BadResponseError):
        client.chat(MSGS, "m")
    assert not list(tmp_path.rglob("*.json"))


def test_generate_style_response_key_is_accepted(tmp_path):
    client, _, _ = make(tmp_path, FakeResp(200, {"response": "hi"}))
    assert client.chat(MSGS, "m") == "hi"


@pytest.mark.parametrize("content", ["", "   \n"])
def test_empty_reply_is_retried_and_never_cached(tmp_path, content):
    client, post, _ = make(tmp_path, ok(content), ok("real"))
    assert client.chat(MSGS, "m") == "real"
    assert len(post.calls) == 2
    stored = [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")]
    assert len(stored) == 1 and "real" in stored[0]


def test_persistently_empty_reply_raises_and_caches_nothing(tmp_path):
    client, _, _ = make(tmp_path, ok(""), ok(""), max_retries=1)
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
    client, post, _ = make(tmp_path, exc, ok("fine"))
    assert client.chat(MSGS, "m") == "fine"
    assert len(post.calls) == 2


def test_invalid_url_is_not_retried(tmp_path):
    client, post, _ = make(tmp_path, requests.exceptions.MissingSchema("no scheme"))
    with pytest.raises(LLMError, match="not a valid URL"):
        client.chat(MSGS, "m")
    assert len(post.calls) == 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_retry_time_is_bounded_by_budget_not_attempts(tmp_path):
    clock = FakeClock()
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
    post = Script(ok("a"), ok("b"))
    c1 = OllamaClient("http://h", DiskCache(tmp_path), post=post, seed=7)
    assert c1.chat(MSGS, "m") == "a"
    assert post.calls[0][1]["json"]["options"] == {"temperature": 0, "seed": 7}
    c2 = OllamaClient("http://h", DiskCache(tmp_path), post=post, seed=8)
    assert c2.chat(MSGS, "m") == "b"  # different seed, different cache entry
    assert len(post.calls) == 2


def test_default_seed_is_fixed(tmp_path):
    post = Script(ok("a"))
    OllamaClient("http://h", DiskCache(tmp_path), post=post).chat(MSGS, "m")
    assert post.calls[0][1]["json"]["options"]["seed"] == 42


def test_chat_cache_key_changes_with_format_and_options(tmp_path):
    post = Script(ok("plain"), ok("json"), ok("hot"), ok("hot2"))
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
    client, post, _ = make(tmp_path, ok('{"days": 1, "city": "Goa"}'))
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
    monkeypatch.setattr("src.common.config.load_dotenv", lambda *a, **k: None)
    client = OllamaClient.from_settings(post=Script(ok("hi")))
    assert client.base_url == "http://box:11434"
    assert client.cache.root == tmp_path / "c"
    assert client.seed == 99 and client.retry_budget_s == 120
    assert client.chat(MSGS, "g") == "hi"
    assert any((tmp_path / "c").rglob("*.json"))


# ---- merged from tests/test_smoke_helpers.py: only the pure helper of scripts/smoke_ollama.py is
# tested here; the script itself needs a live server.

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_ollama.py"


def load():
    spec = importlib.util.spec_from_file_location("smoke_ollama", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines functions only; main() is guarded by __name__
    return mod


def test_model_present_handles_latest_suffix():
    smoke = load()
    assert smoke.model_present(["qwen2.5:7b", "bge-m3:latest"], "qwen2.5:7b")
    assert smoke.model_present(["bge-m3:latest"], "bge-m3")
    assert not smoke.model_present(["qwen2.5:7b"], "llama3.1:8b")
    assert not smoke.model_present(["bge-m3:567m"], "bge-m3")
    assert not smoke.model_present([], "x")
