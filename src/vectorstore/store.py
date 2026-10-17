"""Chroma vector store for chunks (persisted under vectorstore/)."""

import json
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIR = ROOT / "vectorstore"
COLLECTION = "chunks"

# Chunk fields stored as Chroma metadata (Chroma does not accept None values)
METADATA_FIELDS = (
    "source", "pdf_page", "printed_page", "chapter", "section",
    "chunk_index", "chunk_count", "method", "image_path", "has_description",
)
JSON_FIELDS = ("figures",)  # lists, stored as JSON strings


def get_client(persist_dir=PERSIST_DIR):
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(client=None, name=COLLECTION):
    client = client or get_client()
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def reset_collection(client=None, name=COLLECTION):
    """Drop and recreate the collection (used before a full re-index)."""
    client = client or get_client()
    try:
        client.delete_collection(name)
    except Exception:  # collection did not exist yet
        pass
    return get_collection(client, name)


def _metadata(chunk):
    meta = {}
    for field in METADATA_FIELDS:
        value = chunk.get(field)
        if value is None:
            value = -1 if field == "printed_page" else ""
        meta[field] = value
    for field in JSON_FIELDS:
        meta[field] = json.dumps(chunk.get(field) or [], ensure_ascii=False)
    return meta


def add_chunks(collection, chunks, embeddings, batch_size=500):
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[start:start + batch_size],
            metadatas=[_metadata(c) for c in batch],
        )


def search(collection, query_embedding, k=5, where=None):
    """Top-k chunks as [{"id", "text", "score", **metadata}], best first."""
    res = collection.query(
        query_embeddings=[query_embedding], n_results=k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for id_, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        meta = {**meta}
        for field in JSON_FIELDS:
            meta[field] = json.loads(meta.get(field) or "[]")
        hits.append({"id": id_, "text": doc, "score": 1 - dist, **meta})
    return hits
