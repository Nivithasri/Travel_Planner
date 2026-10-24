"""The Phase 9 command-line tool (scripts/phase9.py): parsing, guards and the request check."""
import pytest

from eval.phase9 import store
from scripts import phase9 as P


def test_every_subcommand_parses_and_a_bad_arm_is_refused():
    ap = P.build_parser()
    assert ap.parse_args(["run", "--arm", "full_live", "--ids", "P01,P02"]).arm == "full_live"
    assert ap.parse_args(["sheet", "--set", "live"]).set == "live"
    assert ap.parse_args(["crosscheck", "sample", "--n", "8"]).n == 8
    assert ap.parse_args(["citations", "sample"]).n == 50
    assert ap.parse_args(["judge"]).budget_tokens == 120_000
    assert ap.parse_args(["latency", "--yes", "--with-live"]).with_live
    for argv in (["run", "--arm", "nope"], ["run"], ["sheet"], ["bogus"]):
        with pytest.raises(SystemExit):
            ap.parse_args(argv)


def test_pick_validates_ids_and_keeps_the_given_order():
    reqs = {"P01": {"id": "P01"}, "P02": {"id": "P02"}, "P03": {"id": "P03"}}
    assert [r["id"] for r in P.pick(reqs, None)] == ["P01", "P02", "P03"]
    assert [r["id"] for r in P.pick(reqs, "P03,P01")] == ["P03", "P01"]
    with pytest.raises(SystemExit, match="unknown request id"):
        P.pick(reqs, "P01,P99")


def test_a_sheet_needs_every_arm_for_every_request(monkeypatch):
    monkeypatch.setattr(store, "load_arm", lambda arm, ids=None, root=None: [])
    with pytest.raises(SystemExit, match="needs 30 records"):
        P.records_for("main", {f"P{i:02d}": {} for i in range(1, 11)})
    with pytest.raises(SystemExit, match="needs 20 records"):
        P.records_for("live", {f"P{i:02d}": {} for i in range(1, 11)})


def test_the_limiter_wrapper_adds_up_the_sleeps():
    slept = []

    class Lim:
        def _sleep(self, s):
            slept.append(s)

    client = type("C", (), {"limiter": Lim()})()
    waited = P.wrap_limiter(client)
    client.limiter._sleep(1.5)
    client.limiter._sleep(0.25)
    assert waited[0] == 1.75 and slept == [1.5, 0.25]
    assert P.wrap_limiter(type("N", (), {})()) == [0.0]


def test_every_live_request_gets_a_fresh_live_retriever_and_keeps_the_llm_client(monkeypatch):
    from types import SimpleNamespace

    import src.agents.deps as D

    made = []
    monkeypatch.setattr(D, "build_live", lambda registry, enabled=True: made.append(object())
                        or made[-1])
    import src.retrieval.query_understanding as QU

    monkeypatch.setattr(QU, "load_core_registry", lambda: SimpleNamespace(entries=[]))
    client = object()
    deps = SimpleNamespace(live=None, client=client)
    P.prepare_request(deps, True)
    first = deps.live
    P.prepare_request(deps, True)
    assert deps.live is not first and len(made) == 2 and deps.live is made[1]
    assert deps.client is client  # the rate limiter window must carry over


def test_a_corpus_request_never_builds_a_live_retriever(monkeypatch):
    from types import SimpleNamespace

    import src.agents.deps as D

    monkeypatch.setattr(D, "build_live", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    deps = SimpleNamespace(live=None)
    P.prepare_request(deps, False)
    assert deps.live is None


def test_latency_refuses_without_the_cost_acknowledgement(capsys):
    assert P.main(["latency"]) == 2
    assert "--yes" in capsys.readouterr().out


def test_check_requests_passes_on_the_committed_requests(capsys):
    assert P.main(["check-requests"]) == 0
    assert "OK: 10 requests" in capsys.readouterr().out
