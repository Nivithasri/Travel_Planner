from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from src.collect.raw import RawStore, make_record
from src.collect.withdraw import (
    count_dir_files,
    phrases_from,
    pick_phrase,
    plan_withdrawal,
    scan,
    withdraw,
)

TEXT = ("welcome to munnar the tea gardens are lovely in the morning and the mist is thick "
        "so bring a jacket and walk slowly through the estate before the buses arrive ") * 3


def seed(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    for place in ("Munnar", "Hampi"):
        RawStore(raw).write(make_record("youtube", place, f"https://y/{place}", TEXT + place, 0.0))
    keep = make_record("wikipedia", "Munnar", "https://w/Munnar", "keep me " * 20, 0.0)
    RawStore(raw).write(keep)
    (cache / "youtube_transcript" / "ab").mkdir(parents=True)
    (cache / "youtube_transcript" / "ab" / "x.json").write_text("{}")
    (cache / "http" / "cd").mkdir(parents=True)
    (cache / "http" / "cd" / "search.json").write_text("{}")  # the API search cache is kept
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"url": "https://y/Munnar", "source": "youtube"}) + "\n")
    status = tmp_path / "status" / "places"
    status.mkdir(parents=True)
    (status / "munnar.json").write_text(json.dumps(
        {"place": "Munnar", "sources": {"youtube": {"n_raw": 4, "skipped": None}}}))
    return raw, cache, manifest, tmp_path / "status"


def test_pick_phrase_is_forty_plain_characters_found_verbatim_in_the_json_file():
    p = pick_phrase(TEXT)
    assert p is not None and len(p) == 40 and p in TEXT
    assert p in json.dumps(TEXT, ensure_ascii=False)
    assert pick_phrase("short") is None


def test_withdraw_deletes_records_and_transcript_cache_but_keeps_the_rest(tmp_path):
    raw, cache, manifest, status = seed(tmp_path)
    plan = plan_withdrawal(raw, cache, "youtube", "youtube_transcript")
    assert len(plan.records) == 2 and len(plan.raw_files) == 2 and len(plan.cache_files) == 1
    before = count_dir_files(cache)
    res = withdraw(plan, raw, status, manifest, "youtube", "D27")
    assert res == {"records": 2, "manifest_lines": 2}
    assert not (raw / "youtube").exists()  # raw records and the empty folders are gone
    assert len(RawStore(raw).read("wikipedia", "Munnar")) == 1  # other sources untouched
    after = count_dir_files(cache)
    assert before["youtube_transcript"] == 1 and after["youtube_transcript"] == 0
    assert after["http"] == before["http"] == 1  # the search cache stays
    lines = [json.loads(x) for x in manifest.read_text("utf-8").splitlines()]
    assert lines[0] == {"url": "https://y/Munnar", "source": "youtube"}  # old line untouched
    withdrawn = [x for x in lines if x.get("event") == "withdrawn"]
    assert len(withdrawn) == 2 and {x["reason"] for x in withdrawn} == {"D27"}
    doc = json.loads((status / "places" / "munnar.json").read_text("utf-8"))
    assert doc["sources"]["youtube"]["skipped"] == "withdrawn (D27)"


def git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, check=True)


def test_scan_finds_nothing_in_a_clean_repo_and_finds_a_planted_phrase(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "notes.md").write_text("nothing here\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    raw, cache, manifest, status = seed(tmp_path)
    phrases = phrases_from(plan_withdrawal(raw, cache, "youtube", "youtube_transcript"))
    assert len(phrases) >= 2
    clean = scan(repo, phrases)
    assert (clean["working_tree"], clean["tracked_files"], clean["git_grep_revisions"],
            clean["git_log_p"]) == (0, 0, 0, 0)
    # negative control: plant the phrase in an untracked file, then in a committed file
    (repo / "scratch.txt").write_text(phrases[0] + "\n")
    untracked = scan(repo, phrases)
    assert untracked["working_tree"] >= 1 and untracked["tracked_files"] == 0
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "planted")
    tracked = scan(repo, phrases)
    assert tracked["tracked_files"] >= 1 and tracked["git_grep_revisions"] >= 1
    assert tracked["git_log_p"] >= 1
    assert Path(repo / ".git").exists()


# ---- host-filtered cache entries (D30) ----------------------------------------------------
def seed_http_cache(cache: Path):
    from src.collect.withdraw import phrase_variants  # noqa: F401

    search = json.dumps({"items": [{"snippet": {"title": 'Munnar "Tea" Trip: a 3-day guide',
                                                "channelTitle": "Example Channel One"}}]})
    entries = {
        "g1": {"kind": "http", "model": "www.googleapis.com",
               "value": {"retrieved_at": 1_000_000.0, "text": search, "status": 200}},
        "g2": {"kind": "http", "model": "www.googleapis.com",
               "value": {"retrieved_at": 2_000_000.0, "text": "{}", "status": 200}},
        "w1": {"kind": "http", "model": "en.wikipedia.org",
               "value": {"retrieved_at": 1_500_000.0, "text": "x", "status": 200}},
    }
    (cache / "http" / "ab").mkdir(parents=True)
    for name, doc in entries.items():
        (cache / "http" / "ab" / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")


def test_plan_cache_host_selects_only_that_host_and_reports_the_time_range(tmp_path):
    from src.collect.withdraw import plan_cache_host, retrieved_range

    seed_http_cache(tmp_path)
    files = plan_cache_host(tmp_path, "www.googleapis.com")
    assert [f.stem for f in files] == ["g1", "g2"]
    assert retrieved_range(files) == ("1970-01-12T13:46:40Z", "1970-01-24T03:33:20Z")
    assert plan_cache_host(tmp_path, "nope.example") == []


def test_metadata_phrases_include_json_spellings_and_delete_touches_only_those_files(tmp_path):
    from src.collect.withdraw import delete_files, plan_cache_host, search_metadata_phrases

    seed_http_cache(tmp_path)
    files = plan_cache_host(tmp_path, "www.googleapis.com")
    phrases = search_metadata_phrases(files)
    assert 'Munnar "Tea" Trip: a 3-day guide' in phrases
    assert 'Munnar \\"Tea\\" Trip: a 3-day guide' in phrases  # as it appears inside a JSON file
    assert "Example Channel One" in phrases
    assert delete_files(files) == 2
    left = sorted(p.stem for p in (tmp_path / "http").rglob("*.json"))
    assert left == ["w1"]  # the other host's entry is untouched


def test_scanner_sees_a_title_before_deletion_and_not_after(tmp_path):
    from src.collect.withdraw import delete_files, plan_cache_host, search_metadata_phrases

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "notes.md").write_text("clean\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    seed_http_cache(repo / ".cache")
    files = plan_cache_host(repo / ".cache", "www.googleapis.com")
    phrases = search_metadata_phrases(files)
    before = scan(repo, phrases)
    assert before["working_tree"] >= 1 and before["tracked_files"] == 0  # control: found, untracked
    delete_files(files)
    after = scan(repo, phrases)
    assert after["working_tree"] == 0 and after["git_grep_revisions"] == 0
