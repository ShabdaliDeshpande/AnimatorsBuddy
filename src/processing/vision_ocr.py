"""Transcribe hard pages (hand lettering, text inside drawings) with a local vision LLM.

Uses Ollama's /api/generate endpoint with a vision-capable model (default gemma4:26b).
"""

import base64
import json
import urllib.request

import pymupdf

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
TIMEOUT = 300  # seconds per page
NUM_CTX = 8192        # one page image + prompt + answer fits easily; default 262k wastes VRAM
NUM_PREDICT = 2048    # hard cap on answer length: stops runaway repetition loops

PROMPT = (
    "Transcribe ALL text on this book page exactly as written, including hand-lettered "
    "notes and labels on drawings, captions and page numbers. Output plain text only, "
    "one line per text block, top to bottom. Do not describe the drawings. "
    "Ignore '+' registration crosses and other marks that are not words or numbers."
)


def transcribe_page(page, model=MODEL, url=OLLAMA_URL, zoom=1.0):
    """Render a PyMuPDF page to PNG and return the model's transcription.

    Only when the answer hit NUM_PREDICT (a runaway loop) is the repeated tail
    trimmed; pages that really repeat a label ("BDWN" x 11) are left alone.
    """
    text, hit_limit = _generate(page, PROMPT, model, url, zoom)
    return trim_repetition(text, drop_partial_line=True) if hit_limit else text


def trim_repetition(text, max_period=20, min_repeats=4, drop_partial_line=False):
    """Cut a runaway loop at the end of the text down to a single copy.

    If the last lines are the same block of 1..max_period lines repeated at
    least min_repeats times (e.g. "1/2, 3/2, 2/2, ..." forever), keep one block.
    drop_partial_line: also try without the last line, which a length cut-off
    usually leaves half-written.
    """
    all_lines = text.splitlines()
    candidates = (all_lines, all_lines[:-1]) if drop_partial_line else (all_lines,)
    for lines in candidates:
        for period in range(1, max_period + 1):
            block = lines[-period:]
            repeats, end = 0, len(lines)
            while end - period >= 0 and lines[end - period:end] == block:
                repeats += 1
                end -= period
            if repeats >= min_repeats:
                return "\n".join(lines[:end] + block).strip()
    return text


def ask_about_page(page, prompt, model=MODEL, url=OLLAMA_URL, zoom=1.0):
    """Render a PyMuPDF page to PNG, send it with prompt, return the model's answer."""
    return _generate(page, prompt, model, url, zoom)[0]


def _generate(page, prompt, model, url, zoom):
    """(answer, True if the answer was cut off at NUM_PREDICT)."""
    png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    body = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(png).decode()],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        result = json.load(resp)
    return result["response"].strip(), result.get("done_reason") == "length"
