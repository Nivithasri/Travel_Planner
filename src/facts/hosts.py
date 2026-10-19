"""The API host allowlist (config/sources.json -> api_hosts, D37). Any other host is refused."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from src.common.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config" / "sources.json"
_PLAIN_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?")


@dataclass(frozen=True)
class ApiHost:
    host: str
    path_prefix: str
    min_delay_s: float
    max_requests_per_run: int
    licence: str
    terms_url: str
    robots_exempt: bool
    read_timeout_s: float | None = None  # optional per-host read timeout
    path_prefixes: tuple[str, ...] = ()  # every prefix configured for this host (D43)


def load_api_hosts(path: Path = CONFIG_PATH) -> dict[str, ApiHost]:
    doc = json.loads(path.read_text(encoding="utf-8")).get("api_hosts", {})
    out = {}
    for h in doc.get("hosts", []):
        name = str(h["host"]).lower()
        if not _PLAIN_HOST.fullmatch(name):  # no markdown, scheme, path or port
            raise ValueError(f"api_hosts host must be a plain hostname, got {h['host']!r}")
        if h["host"].lower() in out:  # a further path prefix for a host already listed
            old = out[h["host"].lower()]
            out[old.host] = ApiHost(old.host, old.path_prefix, old.min_delay_s,
                                    old.max_requests_per_run, old.licence, old.terms_url,
                                    old.robots_exempt, old.read_timeout_s,
                                    old.path_prefixes + (h["path_prefix"],))
            continue
        out[h["host"].lower()] = ApiHost(
            h["host"].lower(), h["path_prefix"], float(h["min_delay_s"]),
            int(h["max_requests_per_run"]), h["licence"], h["terms_url"],
            bool(h.get("robots_exempt", False)),
            float(h["read_timeout_s"]) if h.get("read_timeout_s") else None,
            (h["path_prefix"],))
    return out


def require_host(hosts: dict[str, ApiHost], url: str) -> ApiHost:
    parts = urlsplit(url)
    host = hosts.get(parts.netloc.lower())
    if host is None:
        raise ValueError(f"host {parts.netloc!r} is not in api_hosts (D37); refused")
    if not any(parts.path.startswith(p) for p in host.path_prefixes):
        raise ValueError(
            f"path {parts.path!r} is outside {list(host.path_prefixes)} on {host.host}")
    return host
