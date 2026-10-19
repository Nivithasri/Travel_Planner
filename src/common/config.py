"""Project settings, loaded from the environment / .env. Nothing is hardcoded here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED = (
    "OLLAMA_BASE_URL",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "EMBED_MODEL",
    "EVAL_REFERENCE_DATE",
)


DEFAULT_LLM_SEED = 42  # fixed so cached LLM outputs stay reproducible
DEFAULT_RETRY_BUDGET_S = 300.0  # total wall-clock time one LLM call may spend, retries included


class ConfigError(RuntimeError):
    pass


def _optional_number(name: str, default, cast):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    generator_model: str
    judge_model: str
    embed_model: str
    eval_reference_date: date
    cache_dir: Path
    llm_seed: int = DEFAULT_LLM_SEED
    llm_retry_budget_s: float = DEFAULT_RETRY_BUDGET_S


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    """Read settings. Real environment variables win over .env values."""
    load_dotenv(env_file if env_file is not None else REPO_ROOT / ".env", override=False)
    missing = [k for k in _REQUIRED if not os.environ.get(k, "").strip()]
    if missing:
        raise ConfigError(f"Missing required settings: {', '.join(missing)} (see .env.example)")

    generator = os.environ["GENERATOR_MODEL"].strip()
    judge = os.environ["JUDGE_MODEL"].strip()
    if generator == judge:
        raise ConfigError(f"GENERATOR_MODEL and JUDGE_MODEL must differ (both are {generator!r})")

    try:
        ref_date = date.fromisoformat(os.environ["EVAL_REFERENCE_DATE"].strip())
    except ValueError as exc:
        raise ConfigError(f"EVAL_REFERENCE_DATE must be YYYY-MM-DD: {exc}") from exc

    cache_dir = Path(os.environ.get("CACHE_DIR", "").strip() or REPO_ROOT / ".cache")
    budget = _optional_number("LLM_RETRY_BUDGET_S", DEFAULT_RETRY_BUDGET_S, float)
    if budget <= 0:
        raise ConfigError("LLM_RETRY_BUDGET_S must be positive")
    return Settings(
        llm_seed=_optional_number("LLM_SEED", DEFAULT_LLM_SEED, int),
        llm_retry_budget_s=budget,
        ollama_base_url=os.environ["OLLAMA_BASE_URL"].strip().rstrip("/"),
        generator_model=generator,
        judge_model=judge,
        embed_model=os.environ["EMBED_MODEL"].strip(),
        eval_reference_date=ref_date,
        cache_dir=cache_dir,
    )


@dataclass(frozen=True)
class CollectSettings:
    """Settings for data collection. Separate from Settings so collection does not need Ollama."""

    contact_email: str
    user_agent: str
    cache_dir: Path
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    youtube_api_key: str | None = None
    search_api_key: str | None = None


def _opt(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def load_collect_settings(env_file: str | os.PathLike | None = None) -> CollectSettings:
    """CONTACT_EMAIL is required: Wikimedia and polite crawling need a way to reach us. The other
    credentials are optional; a collector without its credentials is skipped, never faked."""
    load_dotenv(env_file if env_file is not None else REPO_ROOT / ".env", override=False)
    email = os.environ.get("CONTACT_EMAIL", "").strip()
    if "@" not in email:
        raise ConfigError("CONTACT_EMAIL is required for data collection (see .env.example)")
    cache_dir = Path(os.environ.get("CACHE_DIR", "").strip() or REPO_ROOT / ".cache")
    return CollectSettings(
        contact_email=email,
        user_agent=f"TravelPlannerResearch/0.1 (academic IR project; {email})",
        cache_dir=cache_dir,
        reddit_client_id=_opt("REDDIT_CLIENT_ID"),
        reddit_client_secret=_opt("REDDIT_CLIENT_SECRET"),
        youtube_api_key=_opt("YOUTUBE_API_KEY"),
        search_api_key=_opt("SEARCH_API_KEY"),
    )
