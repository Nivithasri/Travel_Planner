"""Qrels and query loaders (D70, D71): the test split is refused unless explicitly allowed."""
import json

import pytest

from eval import qrels as q


def write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")


def good_query(**kw):
    base = {"id": "Q1", "text": "a query", "split": "dev", "type": "family",
            "time_sensitive": False, "freshness_slice": False, "place_scope": "free", "places": []}
    base.update(kw)
    return base


def test_loader_refuses_test_without_flag(tmp_path):
    write(tmp_path, "qrels_test.json", {"Q2": {"c1": 2}})
    with pytest.raises(q.TestQrelsRefused):
        q.load_qrels("test", directory=tmp_path)
    assert q.load_qrels("test", allow_test=True, directory=tmp_path) == {"Q2": {"c1": 2}}


def test_dev_loader_never_opens_the_test_file(tmp_path):
    write(tmp_path, "qrels_dev.json", {"Q1": {"c1": 1}})
    # no qrels_test.json exists: loading dev must not need it
    assert q.load_qrels("dev", directory=tmp_path) == {"Q1": {"c1": 1}}
    assert not (tmp_path / "qrels_test.json").exists()


def test_refusal_happens_before_any_file_access(tmp_path):
    # even when the file is missing the error is the refusal, not FileNotFoundError
    with pytest.raises(q.TestQrelsRefused):
        q.load_qrels("test", directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        q.load_qrels("test", allow_test=True, directory=tmp_path)


def test_bad_split_and_bad_labels(tmp_path):
    with pytest.raises(ValueError):
        q.load_qrels("train", directory=tmp_path)
    write(tmp_path, "qrels_dev.json", {"Q1": {"c1": 3}})
    with pytest.raises(ValueError):
        q.load_qrels("dev", directory=tmp_path)
    write(tmp_path, "qrels_dev.json", {"Q1": {"c1": True}})
    with pytest.raises(ValueError):
        q.load_qrels("dev", directory=tmp_path)


def test_query_validation():
    assert q.validate_queries([good_query()]) == []
    assert q.validate_queries([good_query(), good_query()])  # duplicate id
    assert q.validate_queries([good_query(place_scope="named")])  # named without places
    assert q.validate_queries([good_query(places=["Hampi"])])  # free with places
    assert q.validate_queries([good_query(freshness_slice=True)])  # slice must be time_sensitive
    assert q.validate_queries([good_query(split="train")])
    assert q.validate_queries([good_query(type="luxury")])
    bad = good_query()
    del bad["places"]
    assert q.validate_queries([bad])


def test_load_queries_by_split(tmp_path):
    write(tmp_path, "queries.json", [good_query(), good_query(id="Q2", split="test")])
    path = tmp_path / "queries.json"
    assert [x["id"] for x in q.load_queries("dev", path)] == ["Q1"]
    assert [x["id"] for x in q.load_queries(None, path)] == ["Q1", "Q2"]
    write(tmp_path, "queries.json", [good_query(split="x")])
    with pytest.raises(ValueError):
        q.load_queries(None, path)
