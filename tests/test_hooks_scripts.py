"""Tests for the hooks and scripts: attribution and frozen hooks, scripts/collect.py
(facts runner, demo), scripts/smoke_ollama.py (D47)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.collect.run import PlaceStore
from src.common.schemas import CollectSettings

# ==== from test_hooks.py: Tests for scripts/hooks/*.py and .githooks/commit-msg.


HOOKS_ROOT = Path(__file__).resolve().parents[1]


ATTRIB = HOOKS_ROOT / "scripts" / "hooks" / "guard_attribution.py"


FROZEN = HOOKS_ROOT / "scripts" / "hooks" / "guard_frozen.py"


COMMIT_MSG = HOOKS_ROOT / ".githooks" / "commit-msg"


# Built from pieces so this file itself never contains a literal trailer line.
TRAILER = "Co-Authored-By: " + "Cla" + "ude <noreply@" + "anthropic.com>"


SESSION = "Cla" + "ude-Session: abc123"


def run_hook(script, payload, project_dir=None):
    env = {**os.environ}
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    return subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )


def bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


@pytest.mark.parametrize(
    "cmd",
    [
        f'git commit -m "Add x" -m "{TRAILER}"',
        f'git commit -m "Add x\n\n{SESSION}"',
        'git tag -a v1 -m "Generated with a tool"',
        'gh pr create --title t --body "see https://claude.ai/code/abc"',
        'git commit -m "x" -m "noreply@' + 'anthropic.com"',
    ],
)
def test_attribution_hook_blocks(cmd):
    r = run_hook(ATTRIB, bash(cmd))
    assert r.returncode == 2
    assert "BLOCKED" in r.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        'git commit -m "Add RRF fusion and tests"',
        "git status",
        "pytest -q",
        'echo "Co-Authored-By is forbidden"',  # not a git/gh command
        "git log --format=%an",
    ],
)
def test_attribution_hook_allows(cmd):
    r = run_hook(ATTRIB, bash(cmd))
    assert r.returncode == 0 and r.stderr == ""


@pytest.mark.parametrize(
    "cmd",
    [
        "git commit -F msg.txt",
        "git commit --file=msg.txt",
        "git commit --file msg.txt",
        "git commit -C HEAD",
        "git commit --reuse-message=HEAD~1",
        "git commit -aF msg.txt",
        "git commit --amend",
        "git commit --amend --no-edit",
        "git add -A && git commit -F msg.txt",
        'git -c core.hooksPath=x commit -F msg.txt',
        'git commit -m "ok" -C HEAD',
        "git commit -F 'unbalanced",  # unparseable quotes still caught by the fallback
    ],
)
def test_attribution_hook_blocks_hidden_message_commits(cmd):
    r = run_hook(ATTRIB, bash(cmd))
    assert r.returncode == 2
    assert "git commit -m" in r.stderr


@pytest.mark.parametrize(
    "cmd",
    [
        'git commit -m "Add RRF fusion and tests"',
        'git commit -am "Fix cache"',
        'git commit -mFix',  # attached value that merely starts with F
        'git commit -m "mention -F and -C in text"',
        'git commit --amend -m "Reword"',
        'git commit --message="Add x"',
        'git add -A && git commit -m "Add x" && git tag phase-1-done',
        'git commit -m "Add x" -m "second paragraph"',
        "git commit --dry-run -m x",
        "git status && echo done",
    ],
)
def test_attribution_hook_still_allows_clean_commit_m(cmd):
    r = run_hook(ATTRIB, bash(cmd))
    assert r.returncode == 0 and r.stderr == ""


def test_attribution_hook_tolerates_garbage_stdin():
    r = subprocess.run(
        [sys.executable, str(ATTRIB)], input="not json", capture_output=True, text=True
    )
    assert r.returncode == 0


def write_payload(path):
    return {"tool_name": "Write", "tool_input": {"file_path": str(path)}}


def test_frozen_hook_blocks_only_when_marker_exists(tmp_path):
    chunks = tmp_path / "data" / "processed" / "chunks.jsonl"
    chunks.parent.mkdir(parents=True)
    assert run_hook(FROZEN, write_payload(chunks), tmp_path).returncode == 0

    (tmp_path / "data" / "processed" / "FROZEN").write_text("", encoding="utf-8")
    r = run_hook(FROZEN, write_payload(chunks), tmp_path)
    assert r.returncode == 2 and "frozen" in r.stderr
    variant = chunks.with_name("chunks_enriched.jsonl")
    assert run_hook(FROZEN, write_payload(variant), tmp_path).returncode == 2
    for name in ("drops.jsonl", "attribution.json", "stats.md", "FROZEN"):  # all of data/processed
        r = run_hook(FROZEN, write_payload(chunks.with_name(name)), tmp_path)
        assert r.returncode == 2, name
    edit = {"tool_name": "Edit", "tool_input": {"file_path": str(chunks)}}
    assert run_hook(FROZEN, edit, tmp_path).returncode == 2
    outside = tmp_path / "data" / "manifests" / "corpus_stats.md"
    assert run_hook(FROZEN, write_payload(outside), tmp_path).returncode == 0


def test_frozen_hook_qrels_marker(tmp_path):
    qrels = tmp_path / "eval" / "qrels.json"
    qrels.parent.mkdir(parents=True)
    assert run_hook(FROZEN, write_payload(qrels), tmp_path).returncode == 0
    (tmp_path / "eval" / "QRELS_FROZEN").write_text("", encoding="utf-8")
    assert run_hook(FROZEN, write_payload(qrels), tmp_path).returncode == 2


def test_frozen_hook_ignores_payload_without_path(tmp_path):
    assert run_hook(FROZEN, {"tool_input": {}}, tmp_path).returncode == 0


def git(cwd, *args, check=True):
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_commit_msg_hook_strips_attribution_in_a_real_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    hooks = repo / "hooks"
    hooks.mkdir()
    hook = hooks / "commit-msg"
    hook.write_bytes(COMMIT_MSG.read_bytes().replace(b"\r\n", b"\n"))
    hook.chmod(0o755)
    git(repo, "config", "core.hooksPath", str(hooks))
    (repo / "f.txt").write_text("x", encoding="utf-8")
    git(repo, "add", "f.txt")
    msg = f"Add feature\n\nDetails here.\n\n{TRAILER}\n{SESSION}\n"
    r = git(repo, "commit", "-q", "-m", msg, check=False)
    assert r.returncode == 0, r.stderr
    body = git(repo, "log", "-1", "--format=%B").stdout
    assert "Add feature" in body and "Details here." in body
    low = body.lower()
    assert "co-authored-by" not in low and "claude-session" not in low
    assert "anthropic" not in low


def test_repo_commit_msg_hook_has_lf_endings():
    assert b"\r\n" not in COMMIT_MSG.read_bytes()


# ==== from test_facts.py


UA = "Test/0.1 (x@example.org)"


RUNNER_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = RUNNER_ROOT / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_refuses_a_holdout_with_exit_code_2(monkeypatch, capsys):
    mod = load_script()
    cfg = CollectSettings("x@example.org", UA, Path("."))
    monkeypatch.setattr(mod, "load_collect_settings", lambda: cfg)
    assert mod.main(["facts", "--source", "wikidata", "--places", "Thekkady", "--execute"]) == 2
    assert "hold-out" in capsys.readouterr().out


# ==== from test_ensure_place.py


DEMO_ROOT = Path(__file__).resolve().parents[1]


def demo_load():
    spec = importlib.util.spec_from_file_location(
        "collect", DEMO_ROOT / "scripts" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_store_must_not_overlap_the_snapshot_raw_root(tmp_path):
    mod = demo_load()
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    mod.assert_live_only(PlaceStore(tmp_path / "data" / "live"), raw)  # fine
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(raw), raw)  # the snapshot itself
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(raw / "live"), raw)  # inside the snapshot
    with pytest.raises(ValueError):
        mod.assert_live_only(PlaceStore(tmp_path / "data"), raw)  # a parent of the snapshot


def test_default_request_cap_is_25_and_stays_a_hard_cap(monkeypatch):
    mod = demo_load()
    assert mod.DEFAULT_CAP == 25
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "--place", "Yercaud"])
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=mod.DEFAULT_CAP)
    assert ap.parse_args([]).cap == 25
    src = (DEMO_ROOT / "scripts" / "collect.py").read_text("utf-8")
    assert "default=DEFAULT_CAP" in src and "max_retries = 0" in src  # wired in, no retry


def test_shipped_live_root_is_outside_data_raw():
    mod = demo_load()
    mod.assert_live_only(PlaceStore(mod.LIVE_ROOT))
    assert mod.LIVE_ROOT.name == "live"


# ==== from test_llm.py


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_ollama.py"


def smoke_load():
    spec = importlib.util.spec_from_file_location("smoke_ollama", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # defines functions only; main() is guarded by __name__
    return mod


def test_model_present_handles_latest_suffix():
    smoke = smoke_load()
    assert smoke.model_present(["qwen2.5:7b", "bge-m3:latest"], "qwen2.5:7b")
    assert smoke.model_present(["bge-m3:latest"], "bge-m3")
    assert not smoke.model_present(["qwen2.5:7b"], "llama3.1:8b")
    assert not smoke.model_present(["bge-m3:567m"], "bge-m3")
    assert not smoke.model_present([], "x")


# ==== freeze tooling (D48): temp directories only, the real data/processed/FROZEN is never created

REAL_FROZEN = HOOKS_ROOT / "data" / "processed" / "FROZEN"


def test_process_write_refuses_when_frozen_exists(tmp_path, monkeypatch):
    import scripts.process as sp

    (tmp_path / "FROZEN").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sp, "PROCESSED", tmp_path)
    with pytest.raises(SystemExit, match="frozen"):
        sp.run_batch(True)
    assert not REAL_FROZEN.exists()


def _toy_tokenizer(d):
    from tokenizers import Tokenizer, models, pre_tokenizers

    d.mkdir()
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "fort": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    return d


def _freeze_root(tmp_path, monkeypatch, counter_kind):
    import hashlib

    from src.common.schemas import PlaceRegistryEntry
    from src.facts.facts import HoldoutFilter
    from src.process.process import process_place, proxy_count

    root = tmp_path / "root"
    proc, man, cfg = root / "data" / "processed", root / "data" / "manifests", root / "config"
    for d in (proc, man, cfg):
        d.mkdir(parents=True)
    tdir = _toy_tokenizer(tmp_path / "bge-m3")
    monkeypatch.setenv("TOKENIZER_DIR", str(tdir))
    monkeypatch.setenv("EVAL_REFERENCE_DATE", "2026-10-18")
    env = tmp_path / "empty.env"
    env.write_text("", encoding="utf-8")
    e = PlaceRegistryEntry(name="Hampi", aliases=[], state="KA", region_cluster="r",
                           popularity="popular", lat=15.3, lon=76.4)
    body = " ".join(f"Sentence number {i} about the fort." for i in range(80))
    rec = {"source": "blog", "url": "https://inditales.com/h/", "place": "Hampi", "body": body,
           "title": "T", "publish_date": "2011-01-01", "lastmod": None, "license": "L", "extra": {}}
    out, _ = process_place([rec], e, HoldoutFilter(), None, proxy_count)
    files = {"chunks.jsonl": "".join(json.dumps(c, sort_keys=True) + "\n" for _, c in out),
             "drops.jsonl": "", "attribution.json": "{}\n"}
    for name, text in files.items():
        (proc / name).write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256((tdir / "tokenizer.json").read_bytes()).hexdigest()[:12]
    counter = "regex-proxy" if counter_kind == "proxy" else f"tokenizer.json sha256:{digest}"
    (man / "processed_manifest.json").write_text(json.dumps({"counter": counter, "files": {
        f"data/processed/{n}": {"sha256": hashlib.sha256(t.encode()).hexdigest()}
        for n, t in files.items()}}), encoding="utf-8")
    (man / "corpus_stats.json").write_text(json.dumps({"freshness": {
        "reference_date": "2026-10-18",
        "reference_date_source": "EVAL_REFERENCE_DATE in .env"}}), encoding="utf-8")
    (cfg / "freeze.json").write_text(json.dumps({"reddit_resolved": True, "d33_mode": "sidecar"}),
                                     encoding="utf-8")
    (proc / "d33_excluded.json").write_text("{}", encoding="utf-8")
    return root, env


def test_freeze_refuses_the_proxy_counter(tmp_path, monkeypatch, capsys):
    import scripts.freeze as fz

    root, env = _freeze_root(tmp_path, monkeypatch, "proxy")
    assert fz.run(root, env, execute=True, apply_d33=False) == 2
    out = capsys.readouterr().out
    assert "[FAIL] counter is the real bge-m3 tokenizer" in out and "proxy is refused" in out
    assert out.count("[FAIL]") == 1  # everything else in the fixture passes
    assert not (root / "data" / "processed" / "FROZEN").exists() and not REAL_FROZEN.exists()


def test_freeze_dry_run_writes_nothing_and_execute_writes_the_marker(tmp_path, monkeypatch,
                                                                      capsys):
    import scripts.freeze as fz

    root, env = _freeze_root(tmp_path, monkeypatch, "real")
    frozen = root / "data" / "processed" / "FROZEN"
    assert fz.run(root, env, execute=False, apply_d33=False) == 0 and not frozen.exists()
    assert "[FAIL]" not in capsys.readouterr().out
    assert fz.run(root, env, execute=True, apply_d33=False) == 0
    doc = json.loads(frozen.read_text(encoding="utf-8"))
    assert set(doc) == {"chunks_jsonl_sha256", "attribution_json_sha256", "chunk_count", "counter",
                        "tokenizer_dir", "reference_date", "reference_date_source",
                        "frozen_at_utc"}
    assert doc["tokenizer_dir"] == "bge-m3" and doc["reference_date_source"].endswith(".env")
    assert fz.run(root, env, execute=True, apply_d33=False) == 2  # already frozen
    assert not REAL_FROZEN.exists()


def test_freeze_refuses_flag_date_unresolved_reddit_and_d33_without_decision(tmp_path,
                                                                            monkeypatch, capsys):
    import scripts.freeze as fz

    root, env = _freeze_root(tmp_path, monkeypatch, "real")
    stats = root / "data" / "manifests" / "corpus_stats.json"
    stats.write_text(json.dumps({"freshness": {"reference_date": "2026-10-18",
                                               "reference_date_source": "--reference-date"}}),
                     encoding="utf-8")
    (root / "config" / "freeze.json").write_text(json.dumps({"reddit_resolved": False,
                                                             "d33_mode": None}), encoding="utf-8")
    assert fz.run(root, env, execute=True, apply_d33=False) == 2
    out = capsys.readouterr().out
    for name in ("reference date comes from .env", "Reddit recorded as resolved", "D33 applied"):
        assert f"[FAIL] {name}" in out
    assert fz.run(root, env, execute=True, apply_d33=True) == 2  # d33_mode null -> refused
    assert "REFUSED" in capsys.readouterr().out
    assert not (root / "data" / "processed" / "FROZEN").exists()
