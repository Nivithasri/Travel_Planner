"""Pure parts of scripts/rescore_dev.py: the D84 keep rule and the memo wrapper."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rescore_dev", ROOT / "scripts" / "rescore_dev.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_keep_rule_boundary_is_minus_0_005():
    assert rs.verdict(0.0) == "KEEP" and rs.verdict(0.2) == "KEEP"
    assert rs.verdict(-0.005) == "KEEP"  # exactly at the tolerance is kept
    assert rs.verdict(-0.0051) == "DROP" and rs.verdict(-0.188) == "DROP"
    assert rs.verdict(None) == "n/a"


def test_delta_handles_missing_values():
    assert rs.delta(0.5, 0.75) == pytest.approx(0.25)
    assert rs.delta(None, 0.5) is None and rs.delta(0.5, None) is None


def test_memo_runs_each_query_once_and_keeps_k_separate():
    calls = []

    def fn(query, filters=None, k=10):
        calls.append((query, k))
        return [query] * k

    m = rs.memo(fn)
    assert m("q", None, 3) == m("q", None, 3) == ["q", "q", "q"]
    m("q", None, 5)
    m("r", None, 3)
    assert calls == [("q", 3), ("q", 5), ("r", 3)]


def test_every_decision_row_exists_in_the_rows_files():
    names = set()
    for f in ("rows_dev.json", "rows_dev_diagnostics.json"):
        names |= {n for n, _ in json.loads((ROOT / "eval" / f).read_text(encoding="utf-8"))}
    names.add("hybrid_filters")  # added by the script itself
    for comp, a, b in rs.DECISIONS:
        assert a in names and b in names, comp


def test_the_script_reads_no_test_judgments():
    src = (ROOT / "scripts" / "rescore_dev.py").read_text(encoding="utf-8")
    assert "allow_test" not in src and "qrels_test" not in src
    assert 'load_queries("dev")' in src
