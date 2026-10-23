"""Find (and with --execute delete) cache entries that must not be kept (D115).

    .venv/Scripts/python.exe scripts/purge_cache.py --garbled               # dry run
    .venv/Scripts/python.exe scripts/purge_cache.py --garbled --execute
    .venv/Scripts/python.exe scripts/purge_cache.py --live-prompts          # dry run

--garbled: LLM replies (.cache/llm_chat) and app answers (.cache/app) whose text holds a C1
control character (U+0080 to U+009F). Those never occur in real text; they are UTF-8 bytes that a
stream decoded as ISO-8859-1 turned into 'a-circumflex' plus control characters (the bug fixed in
D115). A hit on such an entry would replay the garbled text.

--live-prompts: LLM entries (.cache/llm_chat, .cache/llm_json) whose prompt holds a live search
item (an evidence block <<L1 | ...>>). They were written before D115 stopped caching such
prompts; they hold live snippet text on disk, which D109 says must not be stored.

A dry run changes nothing. Output is counts and the first 12 characters of each file name, never
cache text.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.writer import LIVE_BLOCK  # noqa: E402

C1 = re.compile("[\u0080-\u009f]")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def has_c1(obj) -> bool:
    if isinstance(obj, str):
        return C1.search(obj) is not None
    if isinstance(obj, dict):
        return any(has_c1(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_c1(v) for v in obj)
    return False


def prompt_has_live(entry) -> bool:
    inputs = entry.get("inputs") if isinstance(entry, dict) else None
    msgs = inputs.get("messages") if isinstance(inputs, dict) else None
    if not isinstance(msgs, list):
        return False
    return any(isinstance(m, dict) and LIVE_BLOCK.search(str(m.get("content", "")))
               for m in msgs)


def scan(cache_dir: Path, *, garbled: bool, live_prompts: bool) -> dict[str, list[Path]]:
    """{'garbled': [...], 'live_prompts': [...]} - entry files that match."""
    found: dict[str, list[Path]] = {"garbled": [], "live_prompts": []}
    if garbled:
        for p in sorted((cache_dir / "llm_chat").glob("**/*.json")):
            entry = _load(p)
            if isinstance(entry, dict) and has_c1(entry.get("value")):
                found["garbled"].append(p)
        for p in sorted((cache_dir / "app").glob("*.json")):
            if has_c1(_load(p)):
                found["garbled"].append(p)
    if live_prompts:
        for kind in ("llm_chat", "llm_json"):
            for p in sorted((cache_dir / kind).glob("**/*.json")):
                if prompt_has_live(_load(p)):
                    found["live_prompts"].append(p)
    return found


def safe_cache_dir(path: Path) -> Path:
    """The directory to scan, or SystemExit: it must be a '.cache' directory or inside one, and
    must not be (or lie inside) data/, eval/, index/ or config/ of this repository."""
    p = Path(path).resolve()
    if ".cache" not in p.parts:
        raise SystemExit(f"refused: {p} is not a .cache directory")
    for protected in ("data", "eval", "index", "config", "src", "docs"):
        root = (ROOT / protected).resolve()
        if p == root or root in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}/")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--garbled", action="store_true")
    ap.add_argument("--live-prompts", action="store_true")
    ap.add_argument("--execute", action="store_true", help="delete the matching files")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = ap.parse_args(argv)
    if not (args.garbled or args.live_prompts):
        ap.error("choose --garbled and/or --live-prompts")
    cache_dir = safe_cache_dir(args.cache_dir)
    found = scan(cache_dir, garbled=args.garbled, live_prompts=args.live_prompts)
    deleted = 0
    for label, paths in found.items():
        if label == "garbled" and not args.garbled or label == "live_prompts" and (
                not args.live_prompts):
            continue
        print(f"{label}: {len(paths)} entr{'y' if len(paths) == 1 else 'ies'}")
        for p in paths:
            print(f"  {p.parent.name}/{p.name[:12]}")
            if args.execute:
                p.unlink(missing_ok=True)
                deleted += 1
    print(f"deleted {deleted}" if args.execute else "dry run: nothing deleted (use --execute)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
