"""Redaction of URLs, query strings and secrets in text that may be logged, stored or printed.

Exception messages from `requests` and from libraries embed the full request URL, including
query strings such as `?key=...`. Every message that can reach a log, a status file, a note or the
terminal goes through `redact()` first.
"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://[^\s'\"<>)\]]+")
_SECRET_KV = re.compile(
    r"(?i)\b(key|api_key|apikey|token|access_token|auth|secret|password|sig|signature)"
    r"\s*[=:]\s*[^&\s'\"<>)\]]+"
)


def strip_query(url: str) -> str:
    """scheme://host/path with the query string and fragment removed."""
    return re.split(r"[?#]", url, maxsplit=1)[0]


def redact(text: object) -> str:
    out = _URL.sub(lambda m: strip_query(m.group(0)), str(text))
    return _SECRET_KV.sub(lambda m: m.group(1) + "=<redacted>", out)
