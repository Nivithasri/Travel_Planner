"""Phase 6 variant wiring (eval/variants.py), tested with a fake index so no model is needed."""
import pytest

from eval import variants


class FakeIndex:
    def __init__(self):
        self.chunks = {c: {"chunk_id": c, "place": "Hampi", "text": f"text {c}"}
                       for c in ("a", "b", "c", "d")}
        self.calls: list[tuple[str, int]] = []

    def table(self, name, query, k):
        self.calls.append((name, k))
        rankings = {"bm25_plain": ["a", "b", "c"], "dense_plain": ["b", "c", "d"],
                    "bm25_contextual": ["c", "a"], "dense_contextual": ["a", "d"]}
        return [{"chunk_id": cid, "score": 1.0 / (i + 1)}
                for i, cid in enumerate(rankings[name][:k])]


def test_hybrid_plain_fuses_the_two_plain_arms(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    out = variants.hybrid_plain("q", None, 10)
    # bm25_plain [a,b,c] + dense_plain [b,c,d]: b is in both top ranks -> first
    assert [r["chunk_id"] for r in out][0] == "b"
    assert out[0]["place"] == "Hampi" and "score" in out[0]  # full chunk fields + score
    assert all(name in ("bm25_plain", "dense_plain") for name, _ in fake.calls)


def test_hybrid_arms_pull_depth_100(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    variants.hybrid_contextual("q", None, 10)
    assert all(k == variants.RRF_DEPTH for _, k in fake.calls)  # fuse deep, cut to k after


def test_hybrid_contextual_uses_the_contextual_tables(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_index", lambda: fake)
    variants.hybrid_contextual("q", None, 5)
    assert {name for name, _ in fake.calls} == {"bm25_contextual", "dense_contextual"}


def test_control_arms_use_the_control_index(monkeypatch):
    fake = FakeIndex()
    monkeypatch.setattr(variants, "get_control_index", lambda: fake)
    out = variants.dense_control("q", None, 2)
    assert fake.calls == [("dense_contextual", 2)]
    assert [r["chunk_id"] for r in out] == ["a", "d"]


def test_filters_raise_until_row_5():
    for fn in (variants.hybrid_plain, variants.hybrid_contextual, variants.hybrid_control,
               variants.dense_control, variants.bm25_control):
        with pytest.raises(NotImplementedError):
            fn("q", {"state": ["KA"]}, 10)
