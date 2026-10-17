"""Chat with a local Ollama model (default gemma4:26b), optionally with images."""

import base64
import json
import urllib.request
from pathlib import Path

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:26b"
NUM_CTX = 16384       # ~8 pages of text + 3 page images + answer
NUM_PREDICT = 1500
TEMPERATURE = 0.2
TIMEOUT = 600


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def chat(messages, model=MODEL, on_token=None):
    """Send messages to Ollama and return the full reply.

    on_token: optional callable(str), called with each piece of text as it streams.
    """
    parts = []
    for piece in chat_stream(messages, model=model):
        parts.append(piece)
        if on_token:
            on_token(piece)
    return "".join(parts).strip()


def chat_stream(messages, model=MODEL):
    """Yield the reply piece by piece as Ollama generates it.

    Image paths in a message's "images" list are base64-encoded here.
    """
    payload_messages = []
    for m in messages:
        m = dict(m)
        if m.get("images"):
            m["images"] = [encode_image(p) for p in m["images"]]
        payload_messages.append(m)

    body = {
        "model": model,
        "messages": payload_messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        for line in resp:
            if not line.strip():
                continue
            event = json.loads(line)
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                break
