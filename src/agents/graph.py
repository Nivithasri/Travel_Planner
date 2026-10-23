"""Graph wiring (Phase 7): LangGraph with a fan-out (retrieve || live) and a local sequential
fallback that runs the SAME node functions. Per-node timing is recorded by `instrument`.

    parse -> (retrieve || live) -> merge_grade -> plan -> write -> critic -> finalize
"""
from __future__ import annotations

import operator
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, TypedDict

from src.agents.nodes import NODES, Deps, apply_update
from src.agents.state import RunState

ORDER = ["parse", "retrieve", "live", "merge_grade", "plan", "write", "critic", "finalize"]


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class GState(TypedDict, total=False):
    input_text: str
    overrides: dict
    request: Any
    candidates: list
    corpus_evidence: dict
    live_evidence: list
    live_status: str
    live_note: str | None
    grades: dict
    plan: Any
    packed: dict
    packed_obj: Any
    draft: str
    critic: Any
    repaired: bool
    final: dict
    warnings: Annotated[list, operator.add]
    timings: Annotated[dict, _merge]
    tokens: Annotated[dict, _merge]


def instrument(name: str, fn: Callable[[RunState, Deps], dict], deps: Deps
               ) -> Callable[[RunState], dict]:
    """Wrap a node: time it and add the seconds to the update."""

    def run(state: RunState) -> dict:
        t0 = deps.clock()
        upd = fn(state, deps)
        upd = dict(upd)
        upd["timings"] = {**upd.get("timings", {}), name: round(deps.clock() - t0, 3)}
        return upd

    return run


def initial_state(text: str, overrides: dict | None = None) -> RunState:
    return RunState(input_text=text, overrides=dict(overrides or {}))


def run_local(deps: Deps, text: str, overrides: dict | None = None) -> RunState:
    """Sequential fallback: parse, then retrieve and live in two threads, then the rest."""
    state = initial_state(text, overrides)
    nodes = {n: instrument(n, NODES[n], deps) for n in ORDER}
    apply_update(state, nodes["parse"](state))
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ret, f_live = pool.submit(nodes["retrieve"], state), pool.submit(nodes["live"], state)
        updates = [f_ret.result(), f_live.result()]
    for upd in updates:
        apply_update(state, upd)
    for name in ("merge_grade", "plan", "write", "critic", "finalize"):
        apply_update(state, nodes[name](state))
    return state


def build_graph(deps: Deps):
    """The compiled LangGraph. Node wrappers rebuild a RunState view from the graph's dict state
    and return only the keys the node changed."""
    from langgraph.graph import END, START, StateGraph

    def wrap(name: str):
        inner = instrument(name, NODES[name], deps)

        def node(d: GState) -> dict:
            return inner(RunState.model_construct(**d))

        node.__name__ = name
        return node

    g = StateGraph(GState)
    for n in ORDER:
        g.add_node(n, wrap(n))
    g.add_edge(START, "parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("parse", "live")
    g.add_edge(["retrieve", "live"], "merge_grade")
    for a, b in (("merge_grade", "plan"), ("plan", "write"), ("write", "critic"),
                 ("critic", "finalize"), ("finalize", END)):
        g.add_edge(a, b)
    return g.compile()


def run_graph(deps: Deps, text: str, overrides: dict | None = None) -> RunState:
    out = build_graph(deps).invoke({"input_text": text, "overrides": dict(overrides or {}),
                                    "warnings": [], "timings": {}, "tokens": {}})
    return RunState.model_construct(**out)
