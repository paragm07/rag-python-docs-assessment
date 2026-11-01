"""Local index: SQLite for chunks and embeddings, BM25 and dense search.

Hybrid retrieval fuses two ranked lists with reciprocal rank fusion. BM25
catches exact identifiers (``%j``, ``sort_keys``, ``r.json()``). Dense search
catches paraphrases. Fusion needs no score calibration between the two.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .providers import Embedder

TOKEN_RE = re.compile(r"%[a-zA-Z]|[a-z0-9_]+")
STOP = frozenset("a an and are as at be by do does for from how in is it of on or that the this to what which with you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass
class ChunkRow:
    chunk_id: str
    doc: str
    file: str
    section: str
    text: str
    index_text: str
    meta: dict[str, Any]


class Index:
    def __init__(self, db_path: Path, embedder: Embedder | None, use_dense: bool, k1: float = 1.5, b: float = 0.75):
        self.db_path = db_path
        self.embedder = embedder if use_dense else None
        self.k1, self.b = k1, b
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        manifest = con.execute("SELECT * FROM manifest ORDER BY name").fetchall()
        con.close()
        self.chunks: dict[str, ChunkRow] = {}
        self.order: list[str] = []
        doc_vecs: list[np.ndarray] = []
        for r in rows:
            meta = json.loads(r["meta"])
            self.chunks[r["chunk_id"]] = ChunkRow(r["chunk_id"], r["doc"], r["file"], r["section"],
                                                  r["text"], r["index_text"], meta)
            self.order.append(r["chunk_id"])
            if r["embedding"] is not None:
                doc_vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
        self.manifest = [dict(m) for m in manifest]
        self._build_bm25()
        self.dense_ready = bool(self.embedder) and len(doc_vecs) == len(self.order) and bool(doc_vecs)
        self.dense = np.vstack(doc_vecs) if self.dense_ready else None

    def _build_bm25(self) -> None:
        self.doc_tokens = {cid: tokenize(self.chunks[cid].index_text) for cid in self.order}
        self.doc_len = {cid: len(t) for cid, t in self.doc_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / max(len(self.order), 1)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens.values():
            df.update(set(toks))
        n = len(self.order)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = {cid: Counter(toks) for cid, toks in self.doc_tokens.items()}

    def bm25_rank(self, query: str) -> list[str]:
        q_terms = set(tokenize(query))
        scores: dict[str, float] = {}
        for cid in self.order:
            tf = self.tf[cid]
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[cid] / self.avgdl)
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scores[cid] = s
        return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def dense_rank(self, query: str) -> list[str]:
        if not self.dense_ready or self.embedder is None:
            return []
        qv = self.embedder.embed([query])[0]
        sims = self.dense @ qv
        return [self.order[i] for i in np.argsort(-sims)]

    def search(self, query: str, top_k: int, rrf_k: int = 60) -> list[str]:
        """Return the top_k chunk ids for the query, best first."""
        lists = [self.bm25_rank(query)]
        if self.dense_ready:
            lists.append(self.dense_rank(query))
        fused: dict[str, float] = {}
        for ranked in lists:
            for rank, cid in enumerate(ranked[: max(top_k * 4, 50)], start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        return [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:top_k]

    def get(self, chunk_id: str) -> ChunkRow:
        return self.chunks[chunk_id]
