#!/usr/bin/env python3
"""Smoke test for the hosted LLM API (D59). Run once after filling LLM_* in .env:

    python scripts/smoke_llm.py

Checks (PASS/FAIL each, exit 1 if any fails, exit 5 if the daily quota is already used up):
  1. GET /models lists the configured generator and judge model
  2. one short chat call returns text
  3. one JSON-mode call returns valid JSON matching a tiny schema

Uses a throw-away cache so the calls really reach the API (3 requests, a few hundred tokens,
counted by the client-side limiter). Prints no URL, key or header; error text is scrubbed.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from src.common.cache import DiskCache  # noqa: E402
from src.common.llm import OpenAICompatClient, QuotaExhausted  # noqa: E402
from src.common.schemas import ConfigError, load_llm_settings  # noqa: E402

OPTIONS = {"max_tokens": 300, "reasoning_effort": "low"}


class Pong(BaseModel):
    ok: bool
    word: str


def report(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return passed


def main() -> int:
    try:
        s = load_llm_settings()
    except ConfigError as exc:
        print(f"REFUSING TO RUN: {exc}")
        return 4
    print(f"generator {s.generator_model}; judge {s.judge_model}; limits RPM {s.rpm} TPM {s.tpm} "
          f"RPD {s.rpd} TPD {s.tpd}")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        client = OpenAICompatClient.from_settings(
            cache=DiskCache(tmp), retry_budget_s=min(s.retry_budget_s, 120))
        try:
            available = client.list_models()
        except QuotaExhausted as exc:
            print(f"DAILY QUOTA USED UP: {exc.reason}")
            return 5
        except Exception as exc:  # report, then continue so every check shows a result
            available = []
            results.append(report("GET /models reachable", False,
                                  f"{type(exc).__name__}: {client.scrub(str(exc))[:200]}"))
        else:
            missing = [m for m in (s.generator_model, s.judge_model) if m not in available]
            results.append(report("GET /models lists generator and judge", not missing,
                                  f"{len(available)} models" if not missing
                                  else f"not listed: {missing}"))

        try:
            text = client.chat([{"role": "user", "content": "Reply with the single word pong."}],
                               s.generator_model, options=OPTIONS)
            results.append(report("chat call returns text", bool(text.strip()),
                                  f"{len(text)} chars, tokens {client.last_usage}"))
        except QuotaExhausted as exc:
            print(f"DAILY QUOTA USED UP: {exc.reason}")
            return 5
        except Exception as exc:
            results.append(report("chat call returns text", False,
                                  f"{type(exc).__name__}: {client.scrub(str(exc))[:200]}"))

        try:
            out = client.chat_json(
                [{"role": "user", "content": 'Reply with JSON only: {"ok": true, "word": "pong"}'}],
                s.generator_model, Pong, options=OPTIONS)
            results.append(report("JSON call matches the schema", out.ok is True,
                                  f"tokens {client.last_usage}"))
        except QuotaExhausted as exc:
            print(f"DAILY QUOTA USED UP: {exc.reason}")
            return 5
        except Exception as exc:
            results.append(report("JSON call matches the schema", False,
                                  f"{type(exc).__name__}: {client.scrub(str(exc))[:200]}"))
        print(f"usage this run {dict(client.usage)}; limiter {client.limiter.snapshot()}")
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
