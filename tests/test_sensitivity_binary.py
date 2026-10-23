"""scripts/sensitivity_binary.py: collapse, the binary view of the spot-check, chat share, and the
committed output (identical recall/P@5/MRR before and after the collapse)."""
import json
from pathlib import Path

import pytest

from scripts import sensitivity_binary as sb

ROOT = Path(__file__).resolve().parents[1]


def test_collapse_turns_two_into_one_and_keeps_zero_and_one():
    q = {"Q1": {"a": 2, "b": 1, "c": 0}, "Q2": {"d": 2}}
    assert sb.collapse(q) == {"Q1": {"a": 1, "b": 1, "c": 0}, "Q2": {"d": 1}}
    assert q["Q1"]["a"] == 2  # the input is not modified


def test_binary_agreement_and_kappa_match_a_hand_computation():
    # the owner/chat spot-check (D98): 25 of 30 agree on relevant-or-not; the owner called 19 pairs
    # relevant and the chat 20, so pe = (19*20 + 11*10) / 900 = 490/900 and
    # kappa = (25/30 - 490/900) / (1 - 490/900) = 0.6341
    got = sb.binary_from_confusion([[8, 3, 0], [1, 6, 0], [1, 11, 0]])
    assert got["table_2x2"] == [[8, 3], [2, 17]] and got["n"] == 30
    assert got["agreement"] == pytest.approx(25 / 30)
    assert got["kappa"] == pytest.approx(0.6341, abs=1e-4)
    perfect = sb.binary_from_confusion([[5, 0, 0], [0, 3, 1], [0, 0, 6]])
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == pytest.approx(1.0)


def test_chat_share_counts_slots_filled_by_round_two_pairs():
    ranked = {"Q1": [f"c{i}" for i in range(10)], "Q2": [f"d{i}" for i in range(10)]}
    r2 = {"Q1": {"c0": 1, "c3": 0, "zz": 1}, "Q2": {"d9": 2}}
    assert sb.chat_share(ranked, r2) == pytest.approx(3 / 20)
    assert sb.chat_share({}, r2) == 0.0


def test_significant_and_rank_order():
    comps = [{"comparison": "a -> b", "p_holm": 0.049}, {"comparison": "b -> c", "p_holm": 0.05}]
    assert sb.significant(comps) == {"a -> b"}
    assert sb.rank_order({"x": 0.5, "y": 0.7, "z": 0.5, "w": None}) == ["y", "x", "z", "w"]


def test_family_rows_are_all_analysed():
    rows = set(sb.ROWS)
    assert all(a in rows and b in rows for a, b in sb.FAMILY) and len(sb.FAMILY) == 11


def test_committed_output_keeps_binary_metrics_identical_and_carries_the_heading():
    path = ROOT / "eval" / "ablation" / "sensitivity_binary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["heading"].startswith("Sensitivity analysis, run after the test split was seen")
    assert "frozen retrieve() choice is unchanged" in data["heading"]
    for split in ("dev", "test"):
        for row, v in data["results"][split]["rows"].items():
            for metric in ("mrr", "p@5", "recall@10"):
                assert v["original"][metric] == pytest.approx(v["collapsed"][metric]), (row, metric)
    md = (ROOT / "eval" / "ablation" / "sensitivity_binary.md").read_text(encoding="utf-8")
    assert md.startswith("# Sensitivity analysis, run after the test split was seen")
    assert data["spot_check"]["binary"]["n"] == 30
