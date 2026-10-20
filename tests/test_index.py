"""Phase 3 indexes (D51, D52, D55, D61): BM25 contextual/plain, dense matrices, manifest, verify,
search. Toy corpora and a hashing 'encoder' stand in for bge-m3; nothing is downloaded or called."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.common.cache import DiskCache
from src.retrieval import build_index as bi
from src.retrieval.search import Index, rank

DIM = bi.DENSE_PARAMS["dim"]


def make_chunk(cid, place, text, note, source="reference"):
    return {"chunk_id": cid, "place": place, "source_type": source, "text": text,
            "context_note": note, "url": f"https://x/{cid}", "state": "KL"}


CHUNKS = [
    make_chunk("c01", "Hampi", "The ruins of the Vijayanagara empire spread over boulders.",
               "Reference article about Hampi and its temple ruins."),
    make_chunk("c02", "Munnar", "Tea plantations cover the hills above the town.",
               "Wikivoyage guide about Munnar tea estates.", "wikivoyage"),
    make_chunk("c03", "Alappuzha", "Houseboats leave from the jetty every morning.",
               "Wikivoyage guide about Alappuzha backwaters and canals.", "wikivoyage"),
    make_chunk("c04", "Kochi", "The old harbour has Chinese fishing nets.",
               "Reference article about Kochi."),
    make_chunk("c05", "Kochi", "The old harbour has Chinese fishing nets.",
               "Reference article about Kochi."),  # identical to c04: a tie
    make_chunk("c06", "Ooty", "Temples and churches stand beside the lake.",
               "Blog post about Ooty.", "blog"),
]


def fake_encode(texts):
    """Hashing bag-of-words embedding: texts sharing words get a high cosine (unit norm, 1024-d)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        out[i] = out[i] / n if n else out[i]
    return out


def write_enriched(tmp_path, chunks=CHUNKS, name="chunks.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks), encoding="utf-8")
    return p


def build(tmp_path, chunks=CHUNKS, out_name="index"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    enriched = write_enriched(tmp_path, chunks)
    out = tmp_path / out_name
    out.mkdir(exist_ok=True)
    for kind in bi.BM25_KINDS:
        bi.build_bm25([bi.bm25_text(c, kind) for c in chunks], out / f"bm25_{kind}")
    (out / bi.IDS_FILE).write_text(json.dumps([c["chunk_id"] for c in chunks]), encoding="utf-8")
    for kind in bi.DENSE_KINDS:
        bi.build_dense([bi.dense_text(c, kind) for c in chunks], out / f"dense_{kind}.npy",
                       fake_encode, log=lambda m: None)
    bi.write_manifest(out, enriched, None, {"scratch": True})
    return enriched, out


# ---- passage texts and parameters ------------------------------------------------------

def test_the_passages_follow_the_approved_formats_and_parameters():
    c = CHUNKS[0]
    assert bi.bm25_text(c, "contextual") == c["context_note"] + "\n" + c["text"]
    assert bi.bm25_text(c, "plain") == c["text"]
    assert bi.dense_text(c, "contextual") == c["context_note"] + "\n\n" + c["text"]
    assert bi.dense_text(c, "plain") == c["text"]
    assert (bi.BM25_PARAMS["method"], bi.BM25_PARAMS["k1"], bi.BM25_PARAMS["b"]) == (
        "lucene", 1.5, 0.75)
    assert bi.BM25_PARAMS["stopwords"] == "english" and "snowball" in bi.BM25_PARAMS["stemmer"]
    d = bi.DENSE_PARAMS
    assert (d["max_seq_length"], d["normalize"], d["dtype"], d["device"], d["dim"]) == (
        512, True, "float32", "cpu", 1024)


# ---- BM25 ------------------------------------------------------------------------------

def test_bm25_known_answers_with_stemming_and_stopwords(tmp_path):
    enriched, out = build(tmp_path)
    index = Index(out, enriched)
    top = index.bm25("plain", "ruins of the empire", 3)
    assert top[0]["chunk_id"] == "c01" and top[0]["place"] == "Hampi" and top[0]["score"] > 0
    # "temples" in c06 matches the query word "temple" only because of stemming
    ids = [r["chunk_id"] for r in index.bm25("plain", "temple", 5)]
    assert ids == ["c06"]
    # stop words alone match nothing
    assert index.bm25("plain", "the of and", 5) == []


def test_contextual_bm25_sees_words_that_only_the_note_has(tmp_path):
    enriched, out = build(tmp_path)
    index = Index(out, enriched)
    assert index.bm25("plain", "backwaters canals", 5) == []  # not in any chunk text
    hit = index.bm25("contextual", "backwaters canals", 5)
    assert [r["chunk_id"] for r in hit] == ["c03"]


def test_ties_are_broken_by_chunk_id_ascending(tmp_path):
    enriched, out = build(tmp_path)
    index = Index(out, enriched)
    rows = index.bm25("plain", "harbour fishing nets", 5)
    assert [r["chunk_id"] for r in rows[:2]] == ["c04", "c05"]
    assert rows[0]["score"] == rows[1]["score"]
    assert rank(np.array([1.0, 1.0, 2.0]), ["b", "a", "c"], 3) == [("c", 2.0), ("a", 1.0),
                                                                    ("b", 1.0)]


def test_bm25_unknown_words_give_no_results_instead_of_an_error(tmp_path):
    enriched, out = build(tmp_path)
    assert Index(out, enriched).bm25("contextual", "zzzqqq", 5) == []


# ---- dense -----------------------------------------------------------------------------

def test_dense_known_answers_and_exact_cosine(tmp_path):
    enriched, out = build(tmp_path)
    index = Index(out, enriched, encode=fake_encode)
    rows = index.dense("plain", "tea plantations hills", 3)
    assert rows[0]["chunk_id"] == "c02"
    assert all(-1.0001 <= r["score"] <= 1.0001 for r in rows)
    m = np.load(out / "dense_plain.npy")
    q = fake_encode(["tea plantations hills"])[0]
    assert rows[0]["score"] == pytest.approx(float(m[1] @ q), abs=1e-4)
    ctx = index.dense("contextual", "backwaters canals", 1)
    assert ctx[0]["chunk_id"] == "c03"  # only the note carries those words
    assert index.dense("plain", "backwaters canals", 1)[0]["score"] <= ctx[0]["score"]


def test_query_vectors_are_cached_on_disk(tmp_path):
    enriched, out = build(tmp_path)
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return fake_encode(texts)

    cache = DiskCache(tmp_path / "cache")
    a = Index(out, enriched, encode=counting, cache=cache, model_id="m")
    a.dense("plain", "tea hills", 3)
    b = Index(out, enriched, encode=counting, cache=cache, model_id="m")
    b.dense("contextual", "tea hills", 3)
    assert calls == [["tea hills"]]  # second index object: served from the disk cache


def test_dense_without_an_encoder_says_so(tmp_path):
    enriched, out = build(tmp_path)
    with pytest.raises(RuntimeError, match="no encoder"):
        Index(out, enriched).dense("plain", "x", 3)


def test_build_dense_checkpoints_and_resumes_without_recomputing(tmp_path):
    texts = [f"passage number {i} about place {i}" for i in range(8)]
    out = tmp_path / "dense.npy"
    seen = []

    def dying(batch):
        seen.append(len(batch))
        if len(seen) == 3:
            raise RuntimeError("laptop slept")
        return fake_encode(batch)

    with pytest.raises(RuntimeError):
        bi.build_dense(texts, out, dying, checkpoint_rows=3, log=lambda m: None)
    assert not out.exists() and out.with_name("dense.npy.partial.npy").exists()
    assert len(np.load(out.with_name("dense.npy.partial.npy"))) == 6
    seen.clear()

    def counting(batch):
        seen.append(len(batch))
        return fake_encode(batch)

    stats = bi.build_dense(texts, out, counting, checkpoint_rows=3, log=lambda m: None)
    assert seen == [2] and stats["resumed_from"] == 6  # only the last 2 rows were encoded
    assert np.array_equal(np.load(out), fake_encode(texts))
    assert not list(tmp_path.glob("*.partial*"))


def test_a_partial_file_for_different_texts_is_discarded(tmp_path):
    out = tmp_path / "dense.npy"
    texts = [f"text {i}" for i in range(6)]
    with pytest.raises(RuntimeError):
        n = {"i": 0}

        def dying(batch):
            n["i"] += 1
            if n["i"] == 2:
                raise RuntimeError("stop")
            return fake_encode(batch)

        bi.build_dense(texts, out, dying, checkpoint_rows=3, log=lambda m: None)
    changed = ["DIFFERENT"] + texts[1:]
    msgs = []
    stats = bi.build_dense(changed, out, fake_encode, checkpoint_rows=3, log=msgs.append)
    assert stats["resumed_from"] == 0 and any("discarding a stale partial" in m for m in msgs)
    assert np.array_equal(np.load(out), fake_encode(changed))


# ---- manifest and verify ---------------------------------------------------------------

def test_a_good_index_verifies_and_the_manifest_records_what_was_read(tmp_path):
    enriched, out = build(tmp_path)
    assert bi.verify_index(out, enriched) == []
    m = json.loads((out / bi.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert m["enriched_sha256"] == hashlib.sha256(enriched.read_bytes()).hexdigest()
    assert m["rows"] == len(CHUNKS) and set(m["versions"]) >= {
        "bm25s", "PyStemmer", "sentence-transformers", "torch"}
    assert m["bm25_params"]["k1"] == 1.5 and m["dense_params"]["max_seq_length"] == 512
    assert "dense_plain.npy" in m["artifacts_sha256"] and "ids.json" in m["artifacts_sha256"]
    assert any(k.startswith("bm25_contextual/") for k in m["artifacts_sha256"])


def problems_after(tmp_path, mutate):
    enriched, out = build(tmp_path)
    mutate(out, enriched)
    return bi.verify_index(out, enriched)


def test_verify_catches_a_changed_enriched_file(tmp_path):
    def mutate(out, enriched):
        text = enriched.read_text(encoding="utf-8").replace("Hampi", "Hampy", 1)
        enriched.write_text(text, encoding="utf-8")

    assert any("sha256 differs" in p for p in problems_after(tmp_path, mutate))


def test_verify_catches_an_id_order_change(tmp_path):
    def mutate(out, enriched):
        ids = json.loads((out / bi.IDS_FILE).read_text(encoding="utf-8"))
        ids[0], ids[1] = ids[1], ids[0]
        (out / bi.IDS_FILE).write_text(json.dumps(ids), encoding="utf-8")

    found = problems_after(tmp_path, mutate)
    assert any("ids.json does not equal" in p for p in found)


@pytest.mark.parametrize("what,expect", [
    ("shape", "shape"), ("dtype", "dtype"), ("nan", "NaN"), ("norm", "unit-norm")])
def test_verify_catches_bad_dense_matrices(tmp_path, what, expect):
    def mutate(out, enriched):
        path = out / "dense_plain.npy"
        m = np.load(path)
        if what == "shape":
            m = m[:-1]
        elif what == "dtype":
            m = m.astype(np.float64)
        elif what == "nan":
            m[2, 3] = np.nan
        else:
            m = m * 2.0
        np.save(path, m)

    found = problems_after(tmp_path, mutate)
    assert any(expect in p for p in found), found
    assert any("dense_plain.npy is missing or changed" in p for p in found)  # and the hash


def test_verify_catches_a_missing_artifact_and_a_missing_manifest(tmp_path):
    def mutate(out, enriched):
        (out / "dense_contextual.npy").unlink()

    assert any("dense_contextual.npy" in p for p in problems_after(tmp_path / "a", mutate))
    (tmp_path / "b").mkdir()
    enriched, out = build(tmp_path / "b")
    (out / bi.MANIFEST_FILE).unlink()
    assert bi.verify_index(out, enriched) == [f"{bi.MANIFEST_FILE} is missing"]


def test_model_file_hashes_cover_the_files_that_exist(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    h = bi.model_file_hashes(tmp_path)
    assert set(h) == {"tokenizer.json", "config.json"}
    assert h["config.json"] == hashlib.sha256(b"{}").hexdigest()


# ---- scripts ---------------------------------------------------------------------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"script_{name}", Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_script_stops_when_the_tokenizer_hash_differs(tmp_path, capsys):
    script = load_script("build_index")
    (tmp_path / "tokenizer.json").write_text('{"not": "the bge-m3 tokenizer"}', encoding="utf-8")
    enriched = write_enriched(tmp_path)
    with pytest.raises(SystemExit) as exc:
        script.main(["--source", str(enriched), "--out", str(tmp_path / "o"),
                     "--model-dir", str(tmp_path)])
    assert "STOP" in str(exc.value) and "tokenizer" in str(exc.value)


def test_build_script_refuses_the_real_index_from_another_file_and_a_missing_source(
        tmp_path, capsys):
    script = load_script("build_index")
    assert script.main(["--source", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path)]) == 2
    assert "does not exist" in capsys.readouterr().out
    other = write_enriched(tmp_path)
    assert script.main(["--source", str(other), "--out", str(script.DEFAULT_OUT)]) == 2
    assert "only from data/enriched/chunks.jsonl" in capsys.readouterr().out


def test_bm25_only_scratch_build_and_verify_through_the_script(tmp_path, capsys):
    script = load_script("build_index")
    enriched = write_enriched(tmp_path)
    out = tmp_path / "scratch"
    assert script.main(["--source", str(enriched), "--out", str(out), "--bm25-only"]) == 0
    printed = capsys.readouterr().out
    assert "SCRATCH BUILD" in printed and "bm25_contextual" in printed
    assert (out / "bm25_plain").is_dir() and (out / bi.IDS_FILE).exists()


def test_query_script_prints_four_tables_in_the_agreed_order(tmp_path, capsys, monkeypatch):
    script = load_script("query_index")
    enriched, out = build(tmp_path)
    monkeypatch.setattr(script, "load_encoder", lambda d: fake_encode)
    code = script.main(["tea plantations", "--index-dir", str(out), "--enriched", str(enriched),
                        "--model-dir", str(tmp_path), "--k", "3"])
    text = capsys.readouterr().out
    assert code == 0
    heads = [ln for ln in text.splitlines() if ln.startswith("== ")]
    assert heads == ["== bm25_contextual", "== bm25_plain", "== dense_plain", "== dense_contextual"]
    assert "c02" in text and "Munnar" in text and "wikivoyage" in text
    assert text.count("\n 1  ") == 4  # a first-ranked row in every table
    assert script.main(["x", "--index-dir", str(tmp_path / "empty")]) == 2


# ---- fix round: UTF-8 output and the query-vector cache key ---------------------------------

def test_query_script_prints_characters_the_console_cannot_encode(tmp_path, monkeypatch):
    import io

    script = load_script("query_index")
    chunks = [make_chunk("c01", "Hampi", "Contact ☏ +91 80 2558 3262 for the ferry timings.",
                         "Wikivoyage guide about Hampi.", "wikivoyage"),
              make_chunk("c02", "Munnar", "Tea plantations above the town.", "Munnar tea.")]
    enriched, out = build(tmp_path, chunks)
    monkeypatch.setattr(script, "load_encoder", lambda d: fake_encode)
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr("sys.stdout", console)  # what a default Windows console gives
    with pytest.raises(UnicodeEncodeError):
        print("☏")  # proves the stream cannot print the character on its own
    code = script.main(["ferry timings", "--index-dir", str(out), "--enriched", str(enriched),
                        "--model-dir", str(tmp_path), "--k", "2"])
    console.flush()
    text = raw.getvalue().decode("utf-8")
    assert code == 0 and "Contact ☏ +91" in text
    assert [ln for ln in text.splitlines() if ln.startswith("== ")] == [
        "== bm25_contextual", "== bm25_plain", "== dense_plain", "== dense_contextual"]


def test_the_query_cache_namespace_depends_on_the_recorded_weight_hashes(tmp_path):
    from src.retrieval.search import model_id_from_manifest

    def manifest(files):
        d = tmp_path / ("m" + str(abs(hash(json.dumps(files)))))
        d.mkdir()
        (d / bi.MANIFEST_FILE).write_text(json.dumps({"model_files_sha256": files}),
                                          encoding="utf-8")
        return d

    a = model_id_from_manifest(manifest({"pytorch_model.bin": "aaa", "tokenizer.json": "t"}))
    same = model_id_from_manifest(manifest({"tokenizer.json": "t", "pytorch_model.bin": "aaa"}))
    other_weights = model_id_from_manifest(manifest({"pytorch_model.bin": "bbb",
                                                     "tokenizer.json": "t"}))
    assert a == same and a != other_weights and a.startswith("bge-m3-local-")
    assert model_id_from_manifest(manifest({})) == "bge-m3-local-unrecorded"
    # and the namespace really is part of the cache key: other weights, other cached vectors
    enriched, out = build(tmp_path / "idx")
    calls = []

    def counting(texts):
        calls.append(1)
        return fake_encode(texts)

    cache = DiskCache(tmp_path / "cache")
    Index(out, enriched, encode=counting, cache=cache, model_id=a).dense("plain", "tea", 2)
    Index(out, enriched, encode=counting, cache=cache, model_id=a).dense("plain", "tea", 2)
    Index(out, enriched, encode=counting, cache=cache, model_id=other_weights).dense(
        "plain", "tea", 2)
    assert len(calls) == 2  # the second namespace could not reuse the first one's vector
