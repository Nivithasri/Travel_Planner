"""scripts/run_pipeline.py on fake dependencies: flags, exit codes, the timing log, no request text
in the log."""
import json

from scripts import run_pipeline as rp
from tests.test_graph import FakeClient, make_deps


def factory(**kw):
    def build(*, live, parse, on_piece):
        return make_deps(client=FakeClient(), on_piece=on_piece, **kw)
    return build


def test_flags_become_overrides():
    class A:
        no_llm_parse = True
        days, budget, month, start, end, party = 5, "luxury", 7, "Chennai", None, "family"
        cap_inr, max_daily_hours = 40000, 4.0
        interests, constraints, named, states = "temples, food", "avoid hill roads", "Hampi", "ka"
        time_sensitive = True

    o = rp.overrides_from(A)
    assert o["duration_days"] == 5 and o["budget_tier"] == "luxury" and o["month"] == 7
    assert o["interests"] == ["temples", "food"] and o["constraints"] == ["avoid hill roads"]
    assert o["named_places"] == ["Hampi"] and o["states"] == ["KA"] and o["time_sensitive"]
    assert "end_city" not in o and o["no_llm"] is True


def test_a_run_prints_the_answer_writes_a_log_without_the_request_text(tmp_path, monkeypatch,
                                                                       capsys):
    monkeypatch.setattr(rp, "RUNS", tmp_path / "runs")
    out = tmp_path / "answer.md"
    secret = "Four days of old temples and ruins for Asha"
    code = rp.main(["--query", secret, "--days", "4", "--month", "1", "--no-llm-parse",
                    "--no-live", "--local", "--out", out.as_posix()], deps_factory=factory())
    shown = capsys.readouterr().out
    assert code == 0 and "# South India itinerary" in shown and "Timing per node" in shown
    assert out.read_text(encoding="utf-8").startswith("# South India itinerary")
    log = next((tmp_path / "runs").glob("run_*.jsonl")).read_text(encoding="utf-8")
    lines = [json.loads(x) for x in log.splitlines()]
    assert {x.get("node") for x in lines if "node" in x} >= {"parse", "retrieve", "write",
                                                              "critic"}
    assert lines[-1]["summary"] and lines[-1]["critic"] == "passed"
    assert "Asha" not in log and "temples" not in log


def test_exit_codes_for_infeasible_and_bad_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "RUNS", tmp_path / "runs")
    code = rp.main(["--query", "far away", "--days", "1", "--start", "Hyderabad", "--party",
                    "family", "--named", "Gokarna", "--no-llm-parse", "--no-live", "--local"],
                   deps_factory=factory())
    assert code == 5 and "No feasible itinerary" in capsys.readouterr().out
    bad = rp.main(["--query", "x", "--days", "99", "--no-llm-parse", "--no-live", "--local"],
                  deps_factory=factory())
    assert bad == 2 and "bad input" in capsys.readouterr().err
