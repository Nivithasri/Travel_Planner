import json

import pytest

from src.common import cache as cache_mod
from src.common.cache import (
    DiskCache,
    append_jsonl,
    atomic_write_text,
    cache_key,
    read_jsonl,
)


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
    assert cache.has("k", "m", {})


def test_failed_fn_stores_nothing(tmp_path):
    cache = DiskCache(tmp_path)

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        cache.get_or_compute("k", "m", {}, boom)
    assert not cache.has("k", "m", {})
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


# ---- merged from tests/test_cache_recovery.py: crash recovery for JSONL files and unreadable
# cache entries.

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
    assert not cache.has("k", "m", {"a": 1})
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
