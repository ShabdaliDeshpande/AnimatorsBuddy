"""Structured (JSON-schema constrained) generation with a local Ollama model.

Ollama's "format" option accepts a JSON schema and constrains decoding to it,
so the reply always parses - essential for dataset generation and judging.
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
NUM_CTX = 16384
NUM_PREDICT = 2048
TIMEOUT = 600


def generate_json(prompt, schema, model=MODEL, temperature=0.0, url=OLLAMA_URL):
    """Return the parsed JSON reply (dict) matching schema (a JSON-schema dict)."""
    body = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(json.load(resp)["response"])
