"""Live search configuration and its gates (D100). Read from config/live.json; the API key comes
from .env through src/common/schemas.py and is never read here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.schemas import REPO_ROOT

LIVE_CONFIG = REPO_ROOT / "config" / "live.json"
# Shown with every answer whose live step ran under the owner's risk acceptance (D109).
TERMS_NOTICE = "live search: terms not verified, owner accepted risk"


class LiveConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: Literal["tavily"] = "tavily"
    enabled: bool = True
    terms_status: Literal["not_verified", "owner_accepted_risk"] = "not_verified"
    terms_evidence: str = ""
    endpoint_host: str
    endpoint_path: str
    credits_per_search: int = Field(gt=0)
    default_searches_per_request: int = Field(ge=0)
    max_searches_per_request: int = Field(ge=0)
    max_credits_per_request: int = Field(ge=0)
    monthly_credits: int = Field(gt=0)
    monthly_stop_at: int = Field(gt=0)
    max_results_per_search: int = Field(gt=0, le=10)
    max_live_chunks: int = Field(gt=0)
    snippet_max_chars: int = Field(gt=0, le=2000)
    timeout_s: float = Field(gt=0)
    latency_budget_s: float = Field(gt=0)
    cache_enabled: bool = True
    cache_dir: str = ".cache/live"

    @model_validator(mode="after")
    def _consistent(self) -> LiveConfig:
        if self.default_searches_per_request > self.max_searches_per_request:
            raise ValueError("default_searches_per_request exceeds max_searches_per_request")
        if self.max_searches_per_request * self.credits_per_search > self.max_credits_per_request:
            raise ValueError("max_credits_per_request is below max searches x credits per search")
        if self.monthly_stop_at > self.monthly_credits:
            raise ValueError("monthly_stop_at exceeds monthly_credits")
        return self

    def real_calls_status(self) -> str:
        """'ok' when a real network call is permitted by the configuration, else why not:
        'disabled' or 'terms_not_verified' (the key is checked separately). The terms are never
        'verified' (D109): a real call needs the owner's recorded risk acceptance."""
        if not self.enabled:
            return "disabled"
        if self.terms_status != "owner_accepted_risk":
            return "terms_not_verified"
        return "ok"

    @property
    def cache_path(self) -> Path:
        return REPO_ROOT / self.cache_dir


def load_live_config(path: Path = LIVE_CONFIG) -> LiveConfig:
    return LiveConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
