"""Pure helpers of scripts/parse_queries.py (no network)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "parse_queries", Path(__file__).resolve().parents[1] / "scripts" / "parse_queries.py")
pq = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pq)


def test_flag_table_counts_and_lists_the_wrong_ones():
    queries = [{"id": "A", "time_sensitive": True}, {"id": "B", "time_sensitive": True},
               {"id": "C", "time_sensitive": False}, {"id": "D", "time_sensitive": False}]
    parsed = {"A": {"time_sensitive": True}, "B": {"time_sensitive": False},
              "C": {"time_sensitive": True}, "D": {"time_sensitive": False}}
    t = pq.flag_table(queries, parsed)
    assert (t["tp"], t["fn"], t["fp"], t["tn"]) == (1, 1, 1, 1)
    assert t["accuracy"] == 0.5 and t["wrong"] == ["B", "C"]


def test_quota_state_defaults_outside_data_enriched():
    assert "enriched" not in str(pq.QUOTA_STATE) and ".cache" in str(pq.QUOTA_STATE)


def test_note_counts_groups_by_kind():
    parsed = {"A": {"notes": ["place 'X' dropped: hold-out", "place 'Y' dropped: hold-out"]},
              "B": {"notes": ["month 13 dropped: outside 1-12"]}}
    c = pq.note_counts(parsed)
    assert sum(c.values()) == 3
