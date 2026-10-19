#!/usr/bin/env python3
"""PreToolUse hook: block edits to frozen artifacts. Exit 2 = block, message on stderr."""
import fnmatch
import json
import os
import sys

RULES = [  # (path glob relative to repo root, marker file that activates the freeze)
    ("data/processed/*", "data/processed/FROZEN"),  # every file in data/processed (D48)
    ("eval/qrels*.json", "eval/QRELS_FROZEN"),
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not path:
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    rel = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
    for pattern, marker in RULES:
        if fnmatch.fnmatch(rel, pattern) and os.path.exists(os.path.join(root, marker)):
            sys.stderr.write(
                f"BLOCKED: {rel} is frozen ({marker} exists). "
                "Do not modify it. "
                "Ask the user to remove the marker deliberately if truly needed.\n"
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
