"""Every network client must log. This guard fails if any code in src/ or scripts/ builds an
HttpClient without a request_log (make_http is the approved way and passes one)."""
from __future__ import annotations

import ast
import json

from src.collect import orchestrator
from src.common.config import REPO_ROOT, CollectSettings
from src.common.http import TransportResponse

FIELDS = {"ts", "run_id", "host", "path", "kind", "status", "from_cache", "attempt", "source",
          "note"}


def unlogged_constructions(roots=("src", "scripts")) -> list[str]:
    bad = []
    for root in roots:
        for f in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == "HttpClient" and "request_log" not in {k.arg for k in node.keywords}:
                    bad.append(f"{f.relative_to(REPO_ROOT)}:{node.lineno}")
    return bad


def test_every_httpclient_construction_in_src_and_scripts_passes_a_request_log():
    assert unlogged_constructions() == []


def test_make_http_with_no_log_arguments_logs_a_fetch_with_all_fields(tmp_path, monkeypatch):
    log_path = tmp_path / "request_log.jsonl"
    monkeypatch.setattr(orchestrator, "REQUEST_LOG", log_path)
    cfg = CollectSettings("someone@example.org", "UA/0.1 (someone@example.org)", tmp_path / "c")

    def fake(method, url, headers, timeout):
        return TransportResponse(200, {}, "hello")

    http = orchestrator.make_http(cfg, transport=fake)
    http.fetch("https://sahapedia.org/some/page", {"q": "secret-looking-value"}, check_robots=False)
    http.fetch("https://sahapedia.org/some/page", {"q": "secret-looking-value"}, check_robots=False)
    rows = [json.loads(x) for x in log_path.read_text("utf-8").splitlines()]
    assert len(rows) == 2 and all(set(r) == FIELDS for r in rows)
    first, second = rows
    assert (first["host"], first["path"], first["kind"]) == ("sahapedia.org", "/some/page", "page")
    assert (first["status"], first["from_cache"], first["attempt"]) == (200, False, 1)
    assert (second["from_cache"], second["attempt"]) == (True, 0)
    raw = log_path.read_text("utf-8")
    assert "secret-looking-value" not in raw and "?" not in raw and "hello" not in raw
    assert first["run_id"] == second["run_id"] and first["ts"].endswith("Z")
