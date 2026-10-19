"""CLAUDE.md rule 11: only values the user put in .env may be sent anywhere."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.common.cache import DiskCache
from src.common.config import ConfigError, load_collect_settings
from src.common.http import HttpClient, TransportResponse

ROOT = Path(__file__).resolve().parents[1]
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
AMBIENT = ("CONTACT_EMAIL", "USER_EMAIL", "EMAIL", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL")


def _clear(monkeypatch):
    for k in AMBIENT:
        monkeypatch.delenv(k, raising=False)


def test_missing_contact_email_is_an_error_not_a_guess(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USER_EMAIL", "someone@example.org")  # ambient identifiers are never used
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.org")
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n")
    with pytest.raises(ConfigError, match="CONTACT_EMAIL"):
        load_collect_settings(env)


def test_user_agent_contains_only_the_env_file_address(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USER_EMAIL", "ambient@example.org")
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=owner@example.com\n")
    cfg = load_collect_settings(env)
    assert EMAIL.findall(cfg.user_agent) == ["owner@example.com"]


def test_http_client_sends_exactly_the_settings_user_agent(tmp_path, monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "ambient@example.org")
    env = tmp_path / ".env"
    env.write_text("CONTACT_EMAIL=owner@example.com\n")
    cfg = load_collect_settings(env)
    seen = []

    def transport(method, url, headers, timeout):
        seen.append(dict(headers))
        return TransportResponse(200, {}, "ok")

    client = HttpClient(DiskCache(tmp_path / "c"), cfg.user_agent, transport=transport,
                        sleep=lambda s: None, min_delay_s=0)
    client.fetch("https://example.org/x", check_robots=False)
    assert seen[0]["User-Agent"] == cfg.user_agent
    assert all(EMAIL.findall(v) in ([], ["owner@example.com"]) for v in seen[0].values())


def test_http_module_does_not_read_the_environment_or_contact_settings():
    src = (ROOT / "src" / "common" / "http.py").read_text(encoding="utf-8")
    for forbidden in ("environ", "getenv", "dotenv", "CONTACT_EMAIL", "git config", "subprocess"):
        assert forbidden not in src, forbidden


def test_only_config_reads_contact_email():
    hits = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if "CONTACT_EMAIL" in p.read_text(encoding="utf-8")
    ]
    assert hits == [str(Path("src") / "common" / "config.py")]


def test_no_real_email_address_is_committed_in_source_or_docs():
    allowed = {"example.com", "example.org"}
    offenders = []
    for base in ("src", "scripts", "tests", "docs", "config"):
        for p in (ROOT / base).rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".md", ".json", ".txt"}:
                continue
            for addr in EMAIL.findall(p.read_text(encoding="utf-8", errors="ignore")):
                if addr.split("@")[1] not in allowed:
                    offenders.append((str(p.relative_to(ROOT)), addr))
    assert offenders == []


def test_env_example_and_readme_hold_only_placeholder_addresses():
    """.env.example is committed: its CONTACT_EMAIL must be a placeholder, never a real address."""
    allowed = {"example.com", "example.org"}
    for name in (".env.example", "README.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        found = EMAIL.findall(text)
        assert all(a.split("@")[1] in allowed for a in found), name
    assert "CONTACT_EMAIL= you@example.com" in (ROOT / ".env.example").read_text(encoding="utf-8")
