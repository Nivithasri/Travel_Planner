#!/usr/bin/env python3
"""PreToolUse(Bash) hook.

1. Refuse git commit / tag / notes, gh pr and git config user.* commands that carry AI attribution.
2. Refuse `git commit` forms whose message this hook cannot read (-F/--file, -C/--reuse-message,
   --amend without -m). Use `git commit -m "<message>"` so the message can be checked.
"""
import json
import re
import shlex
import sys

BAD = re.compile(
    r"co-authored-by|claude-session|generated with|claude\.ai/code|noreply@anthropic|🤖",
    re.IGNORECASE,
)
GIT = re.compile(
    r"\bgit\s+(commit|tag|notes)\b|\bgh\s+pr\b|\bgit\s+config\b.*user\.", re.IGNORECASE
)
# Used only when the command cannot be tokenised (e.g. unbalanced quotes from a heredoc).
UNREADABLE_FALLBACK = re.compile(
    r"\bgit\b.*\bcommit\b.*(\s-[A-Za-z]*[FC]\b|\s--file\b|\s--reuse-message\b)", re.IGNORECASE
)

ARG_TAKING_GIT_OPTS = {"-c", "-C", "--git-dir", "--work-tree", "--namespace"}
ARG_TAKING_COMMIT_OPTS = {
    "-m", "--message", "-F", "--file", "-C", "--reuse-message", "-c", "--reedit-message",
    "--author", "--date", "-t", "--template", "--cleanup",
}  # fmt: skip
HELP = (
    "BLOCKED: this git commit form hides its message from the attribution check "
    "(-F/--file, -C/--reuse-message, or --amend without -m). "
    'Use `git commit -m "<message>"` instead.\n'
)


def _segments(cmd):
    """Split a shell command into argument lists at ; && || | and newlines."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    segments, cur = [], []
    for tok in lex:
        if tok and set(tok) <= set(";&|()"):
            segments.append(cur)
            cur = []
        else:
            cur.append(tok)
    segments.append(cur)
    return segments


def _commit_args(seg):
    """Return the arguments after `git [global opts] commit`, or None if seg is not a commit."""
    if "git" not in seg:
        return None
    i = seg.index("git") + 1
    while i < len(seg) and seg[i].startswith("-"):
        i += 2 if seg[i] in ARG_TAKING_GIT_OPTS else 1
    if i < len(seg) and seg[i] == "commit":
        return seg[i + 1 :]
    return None


def hidden_message_commit(cmd):
    try:
        segments = _segments(cmd)
    except ValueError:
        return bool(UNREADABLE_FALLBACK.search(cmd))
    for seg in segments:
        args = _commit_args(seg)
        if args is None:
            continue
        has_message = has_hidden = amend = False
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-m", "--message") or a.startswith("--message="):
                has_message = True
            elif a in ("-F", "--file", "-C", "--reuse-message") or a.startswith(
                ("--file=", "--reuse-message=")
            ):
                has_hidden = True
            elif a == "--amend":
                amend = True
            elif re.fullmatch(r"-[A-Za-z]+", a):  # short cluster such as -am or -sF
                for pos, ch in enumerate(a[1:], start=1):
                    if ch in "mFCc":  # takes a value: the rest of the cluster (or next arg)
                        has_message = has_message or ch == "m"
                        has_hidden = has_hidden or ch in "FC"
                        if pos == len(a) - 1:
                            i += 1  # value is the next argument; skip it
                        break
            if a in ARG_TAKING_COMMIT_OPTS:
                i += 1  # skip the option's value so it is not parsed as an option
            i += 1
        if has_hidden or (amend and not has_message):
            return True
    return False


def main() -> int:
    try:
        cmd = (json.load(sys.stdin).get("tool_input") or {}).get("command") or ""
    except Exception:
        return 0
    if GIT.search(cmd) and BAD.search(cmd):
        sys.stderr.write(
            "BLOCKED: remove all Claude/Anthropic attribution (Co-Authored-By, Claude-Session, "
            "'Generated with', claude.ai links) from this command. "
            "Project rule: no AI attribution in git history.\n"
        )
        return 2
    if hidden_message_commit(cmd):
        sys.stderr.write(HELP)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
