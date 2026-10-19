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
    other = tmp_path / "data" / "processed" / "stats.md"
    assert run_hook(FROZEN, write_payload(other), tmp_path).returncode == 0


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
