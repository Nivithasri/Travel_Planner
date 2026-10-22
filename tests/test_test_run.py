import pytest

from scripts import test_run as T


def test_second_run_is_refused(tmp_path):
    manifest = tmp_path / "test_run_manifest.json"
    T.ensure_not_yet_run(manifest)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        T.ensure_not_yet_run(manifest)


def test_family_rows_are_all_scored():
    scored = {n for n, _ in T.ROWS}
    assert all(a in scored and b in scored for a, b in T.FAMILY)
    assert T.FROZEN_ROW in scored
    assert len(set(T.FAMILY)) == len(T.FAMILY) == 11
