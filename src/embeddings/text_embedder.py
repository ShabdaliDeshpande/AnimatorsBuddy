"""Text embeddings from a local Ollama model (default qwen3-embedding:0.6b).

Documents are embedded as-is; queries get a task instruction in front, which
Qwen3-Embedding is trained to use (improves retrieval a few points).
Vectors come back L2-normalised, so cosine similarity = dot product.
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
NUM_CTX = 2048        # chunks are < 700 tokens; the model default (32k) wastes VRAM
BATCH_SIZE = 32
TIMEOUT = 300

QUERY_INSTRUCTION = (
    "Given a question about animation technique, retrieve passages from an "
    "animation instruction book that answer it"
)


def _embed(texts, model=MODEL, url=OLLAMA_URL):
    body = {
        "model": model,
        "input": texts,
        "truncate": True,
        "options": {"num_ctx": NUM_CTX},
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)["embeddings"]


def embed_documents(texts, model=MODEL, batch_size=BATCH_SIZE, progress=None):
    """Embed a list of chunk texts, batch_size at a time."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(_embed(texts[start:start + batch_size], model=model))
        if progress:
            progress(len(vectors), len(texts))
    return vectors


def embed_query(question, model=MODEL, instruction=QUERY_INSTRUCTION):
    """Embed a search question, with the retrieval instruction prepended."""
    text = f"Instruct: {instruction}\nQuery: {question}" if instruction else question
    return _embed([text], model=model)[0]
