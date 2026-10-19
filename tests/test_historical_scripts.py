from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["build_core_list", "gather_source_evidence"]


def load(name):
    path = ROOT / "scripts" / "historical" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_unreachable(mod, monkeypatch):
    """Anything the guard must prevent: a client or settings object being built."""
    def boom(*a, **k):
        raise AssertionError("reached client or settings construction before the guard")

    monkeypatch.setattr(mod, "HttpClient", boom)
    monkeypatch.setattr(mod, "load_collect_settings", boom)


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_exits_2_without_historical_before_building_anything(name, monkeypatch, capsys):
    mod = load(name)
    make_unreachable(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", [f"{name}.py"])
    assert mod.main() == 2
    err = capsys.readouterr().err
    assert "--historical" in err and "D43" in err and "BYPASSES" in err


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_passes_the_guard_with_historical(name, monkeypatch):
    mod = load(name)

    class PastTheGuard(Exception):
        pass

    def stop(*a, **k):
        raise PastTheGuard

    monkeypatch.setattr(mod, "load_collect_settings", stop)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", "--historical"])
    with pytest.raises(PastTheGuard):
        mod.main()
