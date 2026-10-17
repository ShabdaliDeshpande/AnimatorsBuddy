"""Embed every chunk in data/processed/chunks and (re)build the Chroma index.

Usage:
    python scripts/ingest.py

Run after scripts/chunk_pages.py. Always rebuilds the whole collection, so the
index never keeps chunks that no longer exist.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings.text_embedder import MODEL, embed_documents  # noqa: E402
from src.retrieval.hybrid_search import load_chunks  # noqa: E402
from src.vectorstore.store import PERSIST_DIR, add_chunks, reset_collection  # noqa: E402


def main():
    chunks = load_chunks()
    if not chunks:
        print("No chunks found; run scripts/chunk_pages.py first")
        return

    print(f"Embedding {len(chunks)} chunks with {MODEL} ...")
    t = time.time()
    vectors = embed_documents(
        [c["text"] for c in chunks],
        progress=lambda done, total: print(f"  {done}/{total}", end="\r"),
    )
    print(f"  {len(vectors)} vectors, dim {len(vectors[0])}, {time.time() - t:.0f}s")

    collection = reset_collection()
    add_chunks(collection, chunks, vectors)
    print(f"Indexed {collection.count()} chunks -> {PERSIST_DIR}")


if __name__ == "__main__":
    main()
