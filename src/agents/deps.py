"""The real dependencies of the pipeline (Phase 7): frozen retrieve(), the routing tools, place
cards, the hosted LLM client (writer token cap raised for this client only, D103) and the live
retriever behind its gates (D100). Nothing here calls the network by itself; clients are built
lazily and every call goes through src/common/llm.py or src/common/http.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.agents.evidence import load_pipeline_config
from src.agents.nodes import Deps
from src.agents.state import load_budget_tiers, load_planner_config
from src.common.schemas import REPO_ROOT, load_live_settings
from src.facts.facts import HoldoutFilter
from src.live.config import load_live_config
from src.live.search import LiveRetriever, TavilyClient
from src.retrieval.filters import load_provenance
from src.tools.check_season import check_season
from src.tools.place_cards import CARDS_PATH
from src.tools.travel_time import travel_time

QUOTA_STATE = REPO_ROOT / ".cache" / "llm_quota" / "quota_state.json"
EXCLUDED = REPO_ROOT / "data" / "manifests" / "d33_excluded.json"
PROVENANCE = REPO_ROOT / "data" / "enriched" / "provenance.jsonl"


def build_live(registry, *, enabled: bool = True) -> LiveRetriever | None:
    if not enabled:
        return None
    cfg = load_live_config()
    key = load_live_settings().api_key
    names = [e.name for e in registry.entries if not e.holdout]
    labels = [a for e in registry.entries if not e.holdout for a in e.aliases]
    client = None
    if cfg.real_calls_status() == "ok" and key:  # an HTTP client exists only when a call may happen
        from src.collect.run import make_http
        from src.common.schemas import load_collect_settings

        client = TavilyClient(make_http(load_collect_settings()), key, cfg)
    amap = {a: e.name for e in registry.entries if not e.holdout for a in e.aliases}
    return LiveRetriever(cfg, api_key=key, client=client, place_labels=labels, place_names=names,
                         alias_map=amap)


def build_default_deps(*, live: bool = True, parse: bool = True, on_piece=None) -> Deps:
    from src.common.llm import OpenAICompatClient
    from src.common.schemas import load_llm_settings
    from src.retrieval.query_understanding import load_core_registry, parse_query
    from src.retrieval.retrieve import retrieve

    os.environ.setdefault("LLM_QUOTA_STATE", str(QUOTA_STATE))  # D93: never data/enriched
    registry = load_core_registry()
    model = load_llm_settings().generator_model
    client = OpenAICompatClient.from_settings(max_tokens_cap=2000)  # writer cap (D103)
    holdout = HoldoutFilter.from_registry(registry)
    cards = json.loads(Path(CARDS_PATH).read_text(encoding="utf-8"))["places"]
    excluded = frozenset(json.loads(EXCLUDED.read_text(encoding="utf-8"))["excluded"])
    return Deps(
        registry_entries=registry.entries, excluded=excluded, cards=cards,
        provenance=load_provenance(PROVENANCE), retrieve=retrieve, hours=travel_time,
        season=check_season, planner_cfg=load_planner_config(),
        pipeline_cfg=load_pipeline_config(), budget_cfg=load_budget_tiers(), client=client,
        model=model,
        parse=(lambda text: parse_query(text, client, model, registry, holdout)) if parse else None,
        live=build_live(registry, enabled=live), on_piece=on_piece,
        tokens_used=lambda: int(client.usage["total"]))
