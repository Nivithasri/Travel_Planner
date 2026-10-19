"""The request allowlist (D43): which hosts and paths a production HttpClient may contact.

Built from config/sources.json: `web_allowlist` domains, `api_hosts` entries (host + path prefix,
several prefixes per host allowed) and the `policy_fetch` hosts. Matching is explicit and exact:
a configured web domain also admits its `www.`/bare twin (nothing else); there is no suffix match,
no subdomain match and no "same family" guess. Every request and redirect hop is checked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from src.common.config import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config" / "sources.json"


def host_key(host: str) -> str:
    """The configured-domain rule: lower case, one leading "www." removed. Nothing else."""
    h = host.lower()
    return h[4:] if h.startswith("www.") else h


def _under(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


class Allowlist:
    def __init__(self, web_domains=(), api_prefixes: dict[str, list[str]] | None = None,
                 api_caps: dict[str, int] | None = None, api_delays: dict[str, float] | None = None,
                 policy_hosts=()):
        self.web = {host_key(d) for d in web_domains}
        self.api = {h.lower(): list(p) for h, p in (api_prefixes or {}).items()}
        self.caps = {h.lower(): int(c) for h, c in (api_caps or {}).items()}
        self.delays = {h.lower(): float(d) for h, d in (api_delays or {}).items()}
        self.policy = {h.lower() for h in policy_hosts}

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH) -> Allowlist:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        prefixes: dict[str, list[str]] = {}
        caps: dict[str, int] = {}
        delays: dict[str, float] = {}
        for h in doc.get("api_hosts", {}).get("hosts", []):
            host = str(h["host"]).lower()
            prefixes.setdefault(host, []).append(h["path_prefix"])
            caps[host] = max(caps.get(host, 0), int(h["max_requests_per_run"]))
            delays[host] = max(delays.get(host, 0.0), float(h["min_delay_s"]))
        return cls([d["domain"] for d in doc.get("web_allowlist", [])], prefixes, caps, delays,
                   doc.get("policy_fetch", {}).get("hosts", []))

    def reason(self, url: str) -> str | None:
        """None when the URL may be requested, else why it is refused."""
        parts = urlsplit(url)
        host, path = parts.netloc.lower(), parts.path or "/"
        if parts.scheme not in ("http", "https"):
            return f"scheme {parts.scheme!r} is not http or https"
        if "@" in parts.netloc:
            return "user information (user@host) in the URL is not accepted"
        if host_key(host) in self.web:
            return None
        if host in self.api:
            if path == "/robots.txt" or any(_under(path, p) for p in self.api[host]):
                return None
            return f"path {path!r} on {host} is outside {self.api[host]}"
        if host in self.policy:
            return None
        return f"host {host!r} is not in the allowlist"


# ---- merged from src/common/guard.py: guard for historical one-off scripts that build their own
# HttpClient (D43).

def refuse_without_historical(script: str, historical: bool) -> int | None:
    """Return exit code 2 (and say why on stderr) unless the --historical flag was given."""
    if historical:
        return None
    print(f"{script}: refused (exit 2). This is a historical one-off script that builds its own "
          "HttpClient and BYPASSES the D43 request allowlist. Pass --historical to run it anyway.",
          file=sys.stderr)
    return 2
