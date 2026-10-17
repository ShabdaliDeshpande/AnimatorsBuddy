"""Hybrid retrieval: dense (Chroma) + keyword (BM25), merged with reciprocal rank fusion.

Embeddings are good at meaning ("how do I make a walk feel heavy?"); BM25 is good
at exact names and terms ("Ken Harris", "X-sheet") that embeddings can blur.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.embeddings.text_embedder import embed_query
from src.vectorstore.store import get_collection, search

ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "data" / "processed" / "chunks"

RRF_K = 60            # standard reciprocal-rank-fusion constant
CANDIDATES = 20       # results taken from each retriever before fusion

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_chunks(chunk_dir=CHUNK_DIR):
    chunks = []
    for path in sorted(Path(chunk_dir).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f if line.strip())
    return chunks


class HybridSearcher:
    def __init__(self, chunks=None, collection=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.collection = collection or get_collection()

    def dense(self, question, k=CANDIDATES):
        return search(self.collection, embed_query(question), k=k)

    def keyword(self, question, k=CANDIDATES):
        scores = self.bm25.get_scores(tokenize(question))
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            {**self.chunks[i], "score": float(scores[i])} for i in best if scores[i] > 0
        ]

    def search(self, question, k=5, mode="hybrid"):
        """Top-k chunks; mode is 'hybrid', 'dense' or 'keyword'."""
        if mode == "dense":
            return self.dense(question, k)
        if mode == "keyword":
            return self.keyword(question, k)

        fused = {}
        for hits in (self.dense(question), self.keyword(question)):
            for rank, hit in enumerate(hits):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{**self.by_id[i], "score": fused[i]} for i in best]
