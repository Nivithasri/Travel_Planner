"""Pooling and judging sheets (D72): union, dedup, blindness, seeded shuffle, counts."""
import csv

from eval import pooling

QUERIES = [
    {"id": "Q1", "text": "one", "split": "dev"},
    {"id": "Q2", "text": "two", "split": "test"},
]
CHUNKS = {
    f"c{i}": {"chunk_id": f"c{i}", "place": "Hampi" if i < 4 else "Munnar", "state": "KA",
              "source_type": "reference", "url": f"https://x/{i}", "date_kind": "revised",
              "text": f"text of chunk {i}", "context_note": f"SECRET-NOTE-{i}"}
    for i in range(1, 8)
}
INFO = {cid: {"place": c["place"], "source_type": c["source_type"],
              "provenance": "rule_only" if cid == "c7" else "llm"} for cid, c in CHUNKS.items()}
ATTR = {f"https://x/{i}": {"title": f"Title {i}"} for i in range(1, 8)}
RESULTS = {  # (query, system) -> ids
    ("one", "bm25_contextual"): ["c1", "c2"], ("one", "bm25_plain"): ["c2", "c3"],
    ("one", "dense_plain"): ["c1", "c7"], ("one", "dense_contextual"): ["c2"],
    ("one", "rrf4"): ["c2", "c1", "c3"],
    ("two", "bm25_contextual"): ["c4"], ("two", "bm25_plain"): ["c5"],
    ("two", "dense_plain"): ["c4", "c5"], ("two", "dense_contextual"): ["c6"],
    ("two", "rrf4"): ["c4"],
}


def search(system, query, k):
    return [(cid, 1.0) for cid in RESULTS[(query, system)][:k]]


def pool():
    return pooling.build_pool(QUERIES, search, INFO, {"Munnar"})


def test_pool_is_the_deduplicated_union_with_contributors():
    p = pool()["pairs"]
    assert set(p["Q1"]) == {"c1", "c2", "c3", "c7"}
    assert p["Q1"]["c2"]["systems"] == ["bm25_contextual", "bm25_plain", "dense_contextual",
                                        "rrf4"]
    assert p["Q1"]["c7"]["systems"] == ["dense_plain"]
    assert p["Q1"]["c7"]["provenance"] == "rule_only" and p["Q1"]["c7"]["excluded_place"] is True
    assert p["Q1"]["c1"]["excluded_place"] is False
    assert all(x["round"] == 1 for q in p.values() for x in q.values())


def test_depth_limits_each_system():
    p = pooling.build_pool(QUERIES, search, INFO, set(), depth=1)["pairs"]
    assert set(p["Q1"]) == {"c1", "c2"}  # only the first result of each system


def test_counts_by_split_and_depth_sizes():
    pl = pool()
    assert pooling.counts_by_split(pl, QUERIES) == {"dev": 4, "test": 3, "total": 7}
    assert pooling.pool_size_at_depth(pl["top"], 1) == {"Q1": 2, "Q2": 3}


def test_sheet_is_blind():
    rows = pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR, "dev", seed=1)
    assert len(rows) == 4
    assert set(rows[0]) == set(pooling.SHEET_COLUMNS)
    blob = " ".join(str(v) for r in rows for v in r.values())
    for secret in ("SECRET-NOTE", "bm25", "dense_", "rrf", "rule_only", "fallback",
                   "context_note", "provenance", "score"):
        assert secret not in blob, secret
    assert rows[0]["title"].startswith("Title") and "labelling_rubric.md" in rows[0]["rubric"]
    assert "excluded-place chunks are judged like any other" in rows[0]["rubric"]
    assert all(r["label"] == "" and r["note"] == "" for r in rows)


def test_sheet_splits_and_row_ids():
    dev = pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR, "dev", seed=1)
    test = pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR, "test", seed=1)
    assert {r["query_id"] for r in dev} == {"Q1"} and {r["query_id"] for r in test} == {"Q2"}
    assert [r["row_id"] for r in dev] == [1, 2, 3, 4]


def test_shuffle_is_seeded_and_hides_system_order():
    a = [r["chunk_id"] for r in pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR, "dev", 7)]
    b = [r["chunk_id"] for r in pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR, "dev", 7)]
    assert a == b  # same seed, same order
    orders = {tuple(r["chunk_id"] for r in pooling.sheet_rows(pool(), QUERIES, CHUNKS, ATTR,
                                                              "dev", s)) for s in range(30)}
    assert len(orders) > 1  # the seed changes the order
    assert all(sorted(o) == ["c1", "c2", "c3", "c7"] for o in orders)


def test_sheet_csv_round_trip_keeps_text(tmp_path):
    chunks = {**CHUNKS, "c1": {**CHUNKS["c1"], "text": 'quote " comma, and\nnewline ☎'}}
    rows = pooling.sheet_rows(pool(), QUERIES, chunks, ATTR, "dev", 1)
    path = tmp_path / "s.csv"
    pooling.write_sheet(path, rows)
    back = pooling.read_sheet(path)
    assert [r["text"] for r in back] == [r["text"] for r in rows]
    with path.open(encoding="utf-8-sig", newline="") as f:
        assert next(csv.reader(f)) == list(pooling.SHEET_COLUMNS)


def test_report_states_counts_and_depth_options():
    text = pooling.report(pool(), QUERIES, INFO, {"Munnar"})
    assert "dev 4, test 3, total 7" in text
    assert "depth  5" in text and "NOT applied" in text
