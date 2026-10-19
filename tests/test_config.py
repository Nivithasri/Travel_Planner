from datetime import date

import pytest

from src.common.config import ConfigError, load_settings

KEYS = [
    "OLLAMA_BASE_URL",
    "GENERATOR_MODEL",
    "JUDGE_MODEL",
    "EMBED_MODEL",
    "EVAL_REFERENCE_DATE",
    "CACHE_DIR",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def write_env(tmp_path, **over):
    vals = {
        "OLLAMA_BASE_URL": "http://host:11434/",
        "GENERATOR_MODEL": "gen",
        "JUDGE_MODEL": "judge",
        "EMBED_MODEL": "emb",
        "EVAL_REFERENCE_DATE": "2026-10-18",
    }
    vals.update(over)
    p = tmp_path / ".env"
    p.write_text("\n".join(f"{k}={v}" for k, v in vals.items() if v is not None), encoding="utf-8")
    return p


def test_loads_and_normalises(tmp_path, monkeypatch):
    s = load_settings(write_env(tmp_path))
    assert s.ollama_base_url == "http://host:11434"
    assert s.eval_reference_date == date(2026, 10, 18)
    assert s.generator_model == "gen" and s.judge_model == "judge"
    monkeypatch.delenv("OLLAMA_BASE_URL")  # load_dotenv put these in os.environ


def test_missing_var_raises(tmp_path):
    with pytest.raises(ConfigError, match="JUDGE_MODEL"):
        load_settings(write_env(tmp_path, JUDGE_MODEL=None))


def test_generator_equals_judge_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must differ"):
        load_settings(write_env(tmp_path, JUDGE_MODEL="gen"))


def test_bad_date_rejected(tmp_path):
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        load_settings(write_env(tmp_path, EVAL_REFERENCE_DATE="18/10/2026"))


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GENERATOR_MODEL", "from-env")
    assert load_settings(write_env(tmp_path)).generator_model == "from-env"


# ---- merged from tests/test_config_optional.py: 

@pytest.fixture(autouse=True)
def clean_env_optional(monkeypatch):
    for k in [*KEYS, "LLM_SEED", "LLM_RETRY_BUDGET_S"]:
        monkeypatch.delenv(k, raising=False)


def test_seed_and_retry_budget_defaults_and_overrides(tmp_path, monkeypatch):
    s = load_settings(write_env(tmp_path))
    assert s.llm_seed == 42 and s.llm_retry_budget_s == 300
    monkeypatch.setenv("LLM_SEED", "7")
    monkeypatch.setenv("LLM_RETRY_BUDGET_S", "90.5")
    s = load_settings(write_env(tmp_path))
    assert s.llm_seed == 7 and s.llm_retry_budget_s == 90.5


@pytest.mark.parametrize(
    "name,value", [("LLM_SEED", "abc"), ("LLM_RETRY_BUDGET_S", "x"), ("LLM_RETRY_BUDGET_S", "0")]
)
def test_bad_optional_numbers_rejected(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        load_settings(write_env(tmp_path))
