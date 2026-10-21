"""Cross-encoder reranking, the model-independent part (D88; docs/papers/cross_encoder.md). Pure.

The reranker scores (query, chunk) pairs; `apply_scores` re-orders the top `depth` candidates by
those scores and leaves everything below them where it was (plan.md item 7: rerank the top 30,
ranks 31-100 keep their order).

A scorer is any callable (query: str, chunk_ids: list[str]) -> {chunk_id: logit}.
CrossEncoderScorer is the real one (BAAI/bge-reranker-v2-m3 from a LOCAL folder, offline, CPU):
the score is the model's single raw logit; every score is cached on disk (CLAUDE.md rule 4) under
a key made of the sha256 of the weight and tokenizer files, the query, the chunk id and a hash of
the passage text, so other weights, another passage text or another query never reuse a score.
The model is loaded only on the first cache miss, so a fully cached run needs no torch.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

DEPTH = 30
MAX_LENGTH = 512
BATCH_SIZE = 8
CACHE_KIND = "rerank"
MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "sentencepiece.bpe.model", "model.safetensors", "pytorch_model.bin")
Scorer = Callable[[str, list[str]], dict[str, float]]


def apply_scores(ranked: list[tuple[str, float]], logits: dict[str, float], depth: int = DEPTH
                 ) -> list[tuple[str, float]]:
    """The first `depth` of `ranked`, ordered by logit descending (ties by their prior rank, then
    chunk id), carrying the LOGIT as the new score; the rest unchanged below them. Every
    candidate in the head must have a score."""
    head, tail = ranked[:depth], ranked[depth:]
    missing = [cid for cid, _ in head if cid not in logits]
    if missing:
        raise KeyError(f"no reranker score for {missing[:3]}")
    pos = {cid: i for i, (cid, _) in enumerate(head)}
    order = sorted(head, key=lambda t: (-logits[t[0]], pos[t[0]], t[0]))
    return [(cid, logits[cid]) for cid, _ in order] + tail


# ==== the cross-encoder ==============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weight_hashes(model_dir: Path) -> dict[str, str]:
    """sha256 of every config, tokenizer and weight file of the local folder that exists."""
    return {n: sha256_file(Path(model_dir) / n) for n in MODEL_FILES
            if (Path(model_dir) / n).exists()}


def model_id_from(hashes: dict[str, str]) -> str:
    """Cache namespace: a hash of the file hashes, never a path."""
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return "bge-reranker-" + digest[:16]


def passage_text(chunk: dict, text_kind: str) -> str:
    """The text the reranker reads: the chunk, or the model-written note then the chunk (the
    contextual form of Phase 3, D88). Same separator as the dense contextual text."""
    if text_kind == "contextual":
        return chunk["context_note"] + "\n\n" + chunk["text"]
    if text_kind == "plain":
        return chunk["text"]
    raise ValueError("text_kind must be 'contextual' or 'plain'")


def load_model(model_dir: Path):
    """(tokenizer, model) from a local folder; never downloads (D88)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_TF", "0")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir),
                                                               local_files_only=True)
    model.eval()
    if getattr(model.config, "num_labels", 1) != 1:
        raise RuntimeError("expected a single-logit reranker (num_labels == 1)")
    return tok, model


class CrossEncoderScorer:
    """Scores (query, chunk) pairs with bge-reranker-v2-m3. `score_fn(query, passages) -> logits`
    can be injected (tests); by default it loads the local model on the first cache miss."""

    def __init__(self, model_dir: Path, chunks: dict[str, dict], cache, *,
                 text_kind: str = "contextual", batch_size: int = BATCH_SIZE,
                 max_length: int = MAX_LENGTH, score_fn=None,
                 log: Callable[[str], None] = lambda s: None):
        self.model_dir = Path(model_dir)
        self.chunks = chunks
        self.cache = cache
        self.text_kind = text_kind
        self.batch_size, self.max_length = batch_size, max_length
        self._score_fn = score_fn
        self._log = log
        self.hashes = weight_hashes(self.model_dir)
        if not self.hashes:
            raise FileNotFoundError("the reranker folder holds none of the expected model files")
        self.model_id = model_id_from(self.hashes)
        self.n_scored = 0  # pairs that needed the model (cache misses), for the run report
        self.n_cached = 0
        self._model = None

    def _key(self, query: str, cid: str, passage: str) -> dict:
        return {"query": query, "chunk_id": cid, "max_length": self.max_length,
                "text_kind": self.text_kind,
                "text_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest()}

    def _model_scores(self, query: str, passages: list[str]) -> list[float]:
        if self._score_fn is not None:
            return list(self._score_fn(query, passages))
        import torch

        if self._model is None:
            self._log("loading the reranker (first cache miss)")
            self._model = load_model(self.model_dir)
        tok, model = self._model
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(passages), self.batch_size):
                batch = passages[i:i + self.batch_size]
                enc = tok([query] * len(batch), batch, padding=True, truncation="only_second",
                          max_length=self.max_length, return_tensors="pt")
                out.extend(float(x) for x in model(**enc).logits.view(-1).tolist())
        return out

    def __call__(self, query: str, ids: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        todo: list[tuple[str, str, dict]] = []
        for cid in ids:
            passage = passage_text(self.chunks[cid], self.text_kind)
            key = self._key(query, cid, passage)
            hit = self.cache.get(CACHE_KIND, self.model_id, key, default=None)
            if hit is None:
                todo.append((cid, passage, key))
            else:
                scores[cid] = float(hit)
                self.n_cached += 1
        if todo:
            logits = self._model_scores(query, [p for _, p, _ in todo])
            if len(logits) != len(todo):
                raise RuntimeError("the scorer returned a different number of scores")
            for (cid, _, key), logit in zip(todo, logits):
                self.cache.set(CACHE_KIND, self.model_id, key, float(logit))
                scores[cid] = float(logit)
            self.n_scored += len(todo)
        return scores
