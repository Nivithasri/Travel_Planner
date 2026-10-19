#!/usr/bin/env python3
"""Smoke test for the live Ollama endpoint. Run once the server / ngrok URL is up:

    python scripts/smoke_ollama.py

Checks (PASS/FAIL each, exit code 1 if any fails):
  1. /api/tags is reachable and lists the configured generator, judge and embed models
  2. one short chat call returns text
  3. one format=json chat call returns valid JSON matching a tiny schema

Uses a throw-away cache directory so every call really hits the server. Settings come from .env
(OLLAMA_BASE_URL, GENERATOR_MODEL, JUDGE_MODEL, EMBED_MODEL).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from src.common.cache import DiskCache  # noqa: E402
from src.common.config import load_settings  # noqa: E402
from src.common.llm import OllamaClient  # noqa: E402


class Pong(BaseModel):
    ok: bool
    word: str


def model_present(available: list[str], wanted: str) -> bool:
    """Ollama lists 'name:tag'; a bare configured name means ':latest'."""
    if wanted in available:
        return True
    return ":" not in wanted and f"{wanted}:latest" in available


def report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return passed


def main() -> int:
    s = load_settings()
    print(f"Endpoint: {s.ollama_base_url}")
    with tempfile.TemporaryDirectory() as tmp:
        client = OllamaClient(
            s.ollama_base_url,
            DiskCache(tmp),
            seed=s.llm_seed,
            retry_budget_s=min(s.llm_retry_budget_s, 120),
        )
        results = []

        try:
            available = client.list_models()
        except Exception as exc:  # report, then continue so every check shows a result
            available = []
            results.append(report("GET /api/tags reachable", False, repr(exc)))
        else:
            results.append(report("GET /api/tags reachable", True, f"{len(available)} models"))
        for label, name in (
            ("generator", s.generator_model),
            ("judge", s.judge_model),
            ("embed", s.embed_model),
        ):
            results.append(report(f"{label} model pulled: {name}", model_present(available, name)))

        try:
            text = client.chat(
                [{"role": "user", "content": "Reply with the single word: pong"}],
                s.generator_model,
                options={"num_predict": 16},
            )
            results.append(report("chat call returns text", bool(text.strip()), repr(text[:40])))
        except Exception as exc:
            results.append(report("chat call returns text", False, repr(exc)))

        try:
            pong = client.chat_json(
                [
                    {
                        "role": "user",
                        "content": 'Reply with JSON only: {"ok": true, "word": "pong"}',
                    }
                ],
                s.generator_model,
                Pong,
            )
            results.append(report("format=json call validates", pong.ok is True, repr(pong)))
        except Exception as exc:
            results.append(report("format=json call validates", False, repr(exc)))

    failed = results.count(False)
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
