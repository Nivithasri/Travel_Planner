"""Tests for scripts/hooks/*.py and .githooks/commit-msg."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTRIB = ROOT / "scripts" / "hooks" / "guard_attribution.py"
FROZEN = ROOT / "scripts" / "hooks" / "guard_frozen.py"
COMMIT_MSG = ROOT / ".githooks" / "commit-msg"

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
