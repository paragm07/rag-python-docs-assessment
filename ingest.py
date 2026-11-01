"""Ingest the pinned corpus into the local index. Run once before serving.

    python ingest.py [--config config.yaml] [--no-embeddings]

Writes one SQLite file with a ``chunks`` table and a ``manifest`` table. If
dense retrieval is enabled, each chunk also gets an embedding from the local
embedder (Ollama). ``--no-embeddings`` builds a BM25-only index, which is
useful for checking the pipeline on a machine without the embedding model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path

import numpy as np

from ragdocs.chunking import chunk_file
from ragdocs.config import load_config, resolve
from ragdocs.providers import make_embedder

SCHEMA = """
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS manifest;
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY, doc TEXT NOT NULL, file TEXT NOT NULL, section TEXT NOT NULL,
    text TEXT NOT NULL, index_text TEXT NOT NULL, meta TEXT NOT NULL, embedding BLOB
);
CREATE TABLE manifest (
    name TEXT PRIMARY KEY, version TEXT NOT NULL, source TEXT NOT NULL,
    chunk_count INTEGER NOT NULL, ingested_at TEXT NOT NULL
);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    corpus_dir = resolve(cfg, cfg["corpus"]["dir"])
    db_path = resolve(cfg, cfg["index"]["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    chunks = []
    manifest = []
    for doc in cfg["corpus"]["documents"]:
        doc_chunks = []
        for fname in doc["files"]:
            rel = f"{doc['name']}/{fname}"
            doc_chunks += chunk_file(corpus_dir / rel, doc=doc["name"], version=doc["version"],
                                     rel_file=rel, cfg_index=cfg["index"])
        chunks += doc_chunks
        manifest.append((doc["name"], doc["version"], doc["source"], len(doc_chunks),
                         dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
        print(f"{doc['name']}: {len(doc_chunks)} chunks")

    embeddings = None
    if cfg["retrieval"]["use_dense"] and not args.no_embeddings:
        embedder = make_embedder(cfg["models"]["embedder"])
        vectors = []
        for start in range(0, len(chunks), 32):
            batch = [c.index_text for c in chunks[start:start + 32]]
            vectors.append(embedder.embed(batch))
            print(f"embedded {min(start + 32, len(chunks))}/{len(chunks)}", flush=True)
        embeddings = np.vstack(vectors).astype(np.float32)

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for i, c in enumerate(chunks):
        meta = {k: v for k, v in c.to_dict().items()
                if k not in ("text", "index_text", "chunk_id", "doc", "file", "section")}
        emb = embeddings[i].tobytes() if embeddings is not None else None
        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.doc, c.file, c.section, c.text, c.index_text,
                     json.dumps(meta), emb))
    con.executemany("INSERT INTO manifest VALUES (?,?,?,?,?)", manifest)
    con.commit()
    con.close()
    print(f"wrote {len(chunks)} chunks to {db_path} (embeddings: {embeddings is not None})")


if __name__ == "__main__":
    main()
