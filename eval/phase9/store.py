"""Run records on disk and the guards of the live day (D123, D126).

Records are JSON files under .cache/phase9/<arm>/<request id>.json (gitignored: they hold answer
text and source excerpts, D100, O12). A run never overwrites a record unless forced.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from src.common.schemas import REPO_ROOT

CACHE = REPO_ROOT / ".cache" / "phase9"
SHEETS = REPO_ROOT / "eval" / "sheets" / "phase9"
RESULTS = REPO_ROOT / "eval" / "phase9" / "results"
REQUEST_LOG = REPO_ROOT / "data" / "manifests" / "request_log.jsonl"
LEDGER = REPO_ROOT / ".cache" / "live" / "ledger.json"


class RecordExists(RuntimeError):
    """A record for this arm and request already exists and --force was not given."""


def record_path(arm: str, request_id: str, root: Path = CACHE) -> Path:
    return root / arm / f"{request_id}.json"


def save_record(rec: dict, root: Path = CACHE, *, force: bool = False) -> Path:
    path = record_path(rec["arm"], rec["request_id"], root)
    if path.exists() and not force:
        raise RecordExists(f"{path.name} already exists for arm {rec['arm']}: pass --force to "
                           "replace it (a rerun changes what every later table is built from)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_arm(arm: str, ids: Iterable[str] | None = None, root: Path = CACHE) -> list[dict]:
    """The records of one arm, sorted by request id; `ids` restricts the set."""
    wanted = set(ids) if ids is not None else None
    out = []
    for p in sorted((root / arm).glob("*.json")) if (root / arm).is_dir() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if wanted is None or rec["request_id"] in wanted:
            out.append(rec)
    return out


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def live_guard(*, terms_status: str, key_present: bool, ledger: dict | None,
               planned_searches: int, credits_per_search: int, credit_cap: int,
               monthly_stop_at: int, existing_dates: Iterable[str], today: str) -> list[str]:
    """Reasons the live day must not start (D126); an empty list means it may. `ledger` is the
    parsed .cache/live/ledger.json (credits used this month) or None."""
    problems = []
    if terms_status != "ok":
        problems.append(f"live calls are not allowed by config/live.json ({terms_status})")
    if not key_present:
        problems.append("no live search key in .env")
    need = planned_searches * credits_per_search
    if need > credit_cap:
        problems.append(f"the plan needs {need} assumed credits, above the cap of {credit_cap}")
    used = int((ledger or {}).get("credits", 0))
    if used + need > monthly_stop_at:
        problems.append(f"{used} credits used + {need} planned exceeds the monthly stop at "
                        f"{monthly_stop_at}")
    other = sorted({d for d in existing_dates if d != today})
    if other:
        problems.append(f"live records from another day exist ({other}): the live arm must be "
                        "one UTC day (D126); remove them or run with --force on a new set")
    return problems
