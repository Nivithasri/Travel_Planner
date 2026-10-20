"""Mutation checks for Phase 3 (and the Phase 2 hold-out regression), D54.

    python scripts/mutation_checks.py

Applies ONE textual mutation at a time to a source file, runs the enrichment/process/facts tests,
prints the failing test names, and restores the file byte-for-byte (sha256 compared before and
after). A mutation that no test catches is reported as SURVIVED and the exit code is 1.
Nothing is committed or left changed: `git diff --stat` must be empty afterwards.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_enrich.py", "tests/test_process.py", "tests/test_facts.py",
         "tests/test_llm_api.py"]

MUTATIONS = [
    ("M1 leak scan disabled (leak_scan returns None)", "src/process/enrich.py",
     'def leak_scan(text: str, holdout: HoldoutFilter) -> str | None:\n'
     '    """Reason string when `text` names a hold-out place (HoldoutFilter or the independent'
     ' regex)."""\n',
     'def leak_scan(text: str, holdout: HoldoutFilter) -> str | None:\n    return None\n'),
    ("M2 fallback disabled (a bad note is kept)", "src/process/enrich.py",
     '    if problem:\n        detail = {"detail"',
     '    if False:\n        detail = {"detail"'),
    ("M15 grounding check disabled (ungrounded names pass)", "src/process/enrich.py",
     '("ungrounded" if ungrounded else None)',
     'None'),
    ("M16 hold-out stems match inside words again (Gitanjali is a leak)", "src/process/enrich.py",
     'r"\\b(?:chikmagal|chikkamagal|chickmagal|araku',
     'r"(?:chikmagal|chikkamagal|chickmagal|araku'),
    ("M3 enriched-vs-frozen check disabled", "src/process/process.py",
     '    e = {c["chunk_id"]: c for c in enriched}\n',
     '    return\n    e = {c["chunk_id"]: c for c in enriched}\n'),
    ("M4 Phase 2 hold-out regression (text_match returns None)", "src/facts/facts.py",
     '        m = self._names.search(text)\n        if m:\n            term = m.group(0)',
     '        return None\n        m = self._names.search(text)\n        if m:\n'
     '            term = m.group(0)'),
    ("M5 note-copy rule disabled", "src/process/enrich.py",
     '    if copy_share(note, chunk_text) > COPY_MAX_SHARE:\n',
     '    if False:\n'),
    ("M6 transport failure turned into a fallback", "src/process/enrich.py",
     "    except JSONValidationError:\n",
     "    except (JSONValidationError, LLMError):\n"),
    ("M7 mentioned_places hold-out filter disabled", "src/process/enrich.py",
     '        if leak_scan(name, holdout):\n            dropped["holdout"] += 1\n'
     '            continue\n',
     ''),
    ("M8 rate limiter pacing disabled (never sleeps)", "src/common/llm.py",
     '            if wait <= 0:\n',
     '            if True:\n'),
    ("M9 scrubbing disabled (keys, headers and URLs pass through)", "src/common/llm.py",
     '    for pattern in _SECRET_PATTERNS:\n        text = pattern.sub("<redacted>", text)\n',
     '    return text\n    for pattern in _SECRET_PATTERNS:\n'
     '        text = pattern.sub("<redacted>", text)\n'),
    ("M10 reasoning text stored with the reply", "src/common/llm.py",
     '    return content\n\n\nclass OpenAICompatClient',
     '    return content + str(message.get("reasoning", ""))\n\n\nclass OpenAICompatClient'),
    ("M11 daily token budget not enforced", "src/common/llm.py",
     '        if not self.trust_provider and self.day_tokens + est_tokens > self.tpd_eff:\n',
     '        if False:\n'),
    ("M12 Retry-After ignored", "src/common/llm.py",
     '            if retry_after is not None:  # the server said how long to wait: honour it\n',
     '            if False:\n'),
    ("M13 checkpoint model/prompt mismatch not refused", "src/process/enrich.py",
     '        if prov.get("model") != model or prov.get("prompt_version") != PROMPT_VERSION:\n',
     '        if False:\n'),
    ("M14 blog chunks sent although --include-blog is off", "src/process/enrich.py",
     '    if chunk["source_type"] == "blog" and not include_blog:\n',
     '    if False:\n'),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests() -> tuple[str, list[str]]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rf",
                        *TESTS], cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    failed = [ln.split(" ", 1)[1].split(" - ")[0] for ln in lines if ln.startswith("FAILED ")]
    return (lines[-1] if lines else "no output"), failed


def main() -> int:
    survived = 0
    rows = []
    for name, rel, old, new in MUTATIONS:
        path = ROOT / rel
        original = path.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")
        if text.count(old) != 1:
            print(f"{name}: PATTERN NOT FOUND ({text.count(old)} matches); nothing run")
            survived += 1
            continue
        path.write_bytes(text.replace(old, new).encode("utf-8"))
        try:
            summary, failed = run_tests()
        finally:
            path.write_bytes(original)
        restored = sha(path) == before
        rows.append((name, len(failed), restored))
        print(f"\n{name}\n  {summary}\n  restored byte-identical: {restored}")
        for f in failed:
            print(f"  FAILED {f}")
        if not failed:
            survived += 1
            print("  SURVIVED: no test caught this mutation")
    print("\n| mutation | failing tests | restored |\n|---|---|---|")
    for name, n, ok in rows:
        print(f"| {name} | {n} | {ok} |")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
