"""scripts/purge_cache.py (D115): finds garbled LLM replies and app answers and, separately,
LLM prompts that hold live text; deletes only with --execute. No network, no real cache."""
import importlib.util
import json
from pathlib import Path

from src.common.schemas import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "purge_cache.py"
spec = importlib.util.spec_from_file_location("purge_cache", SCRIPT)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

GARBLED = "6AM" + chr(0xE2) + chr(0x80) + chr(0x93) + "12"
CLEAN = "6AM" + chr(0x2013) + "12"


def put(base: Path, kind: str, name: str, entry):
    d = base / kind / name[:2]
    d.mkdir(parents=True, exist_ok=True)
    (d / (name + ".json")).write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")


def make_cache(tmp_path):
    base = tmp_path / ".cache"
    live = "<<L1 | Hampi | live | live, fetched 2026-10-23>>" + chr(10) + "snippet" + chr(10)
    put(base, "llm_chat", "aa11", {"inputs": {"messages": []}, "value": GARBLED})
    put(base, "llm_chat", "bb22", {"inputs": {"messages": []}, "value": CLEAN})
    put(base, "llm_chat", "cc33", {"inputs": {"messages": [{"role": "user", "content": live}]},
                                   "value": CLEAN})
    put(base, "llm_json", "dd44", {"inputs": {"messages": [{"role": "user", "content": "Request"}]},
                                   "value": "{}"})
    app = base / "app"
    app.mkdir(parents=True)
    (app / "e1.json").write_text(json.dumps({"final": {"text": GARBLED}}), encoding="utf-8")
    (app / "e2.json").write_text(json.dumps({"final": {"text": CLEAN}}), encoding="utf-8")
    return base


def test_the_scan_finds_garbled_entries_and_live_prompts_separately(tmp_path):
    base = make_cache(tmp_path)
    found = P.scan(base, garbled=True, live_prompts=True)
    assert sorted(p.stem for p in found["garbled"]) == ["aa11", "e1"]
    assert [p.stem for p in found["live_prompts"]] == ["cc33"]
    only = P.scan(base, garbled=True, live_prompts=False)
    assert only["live_prompts"] == []


def test_every_c1_control_character_counts_and_ordinary_text_does_not():
    for code in (0x80, 0x93, 0x9F):
        assert P.has_c1("x" + chr(code) + "y")
    assert not P.has_c1(CLEAN + " caf" + chr(0xE9) + " " + chr(0x20B9) + "1,000")
    assert not P.has_c1("plain ascii") and not P.has_c1({"a": ["fine", 3]})
    assert P.has_c1({"a": [{"b": GARBLED}]})


def test_a_dry_run_deletes_nothing_and_execute_deletes_only_the_selected_kind(tmp_path, capsys):
    base = make_cache(tmp_path)
    assert P.main(["--garbled", "--live-prompts", "--cache-dir", str(base)]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and len(list(base.rglob("*.json"))) == 6
    assert GARBLED not in out and "snippet" not in out  # counts and file names only
    assert P.main(["--garbled", "--execute", "--cache-dir", str(base)]) == 0
    left = sorted(p.stem for p in base.rglob("*.json"))
    assert left == ["bb22", "cc33", "dd44", "e2"]  # the live prompt was not selected


def test_live_prompts_are_deleted_only_when_asked(tmp_path):
    base = make_cache(tmp_path)
    assert P.main(["--live-prompts", "--execute", "--cache-dir", str(base)]) == 0
    assert sorted(p.stem for p in base.rglob("*.json")) == ["aa11", "bb22", "dd44", "e1", "e2"]
