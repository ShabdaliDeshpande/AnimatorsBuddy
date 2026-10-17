"""Describe what the drawings on a page show, using a local vision LLM.

Complements vision_ocr.transcribe_page(): that one copies the words, this one
explains the pictures, so retrieval can match questions answered by a drawing.
"""

from src.processing.vision_ocr import MODEL, OLLAMA_URL, ask_about_page

PROMPT = (
    "This is a page from an animation instruction book. Describe what the drawings "
    "and images on this page show: the characters, their poses and motion, any "
    "timing, spacing, arcs or key/breakdown positions indicated, and the animation "
    "technique or principle being demonstrated. Write 2-5 plain sentences. Do not "
    "transcribe the text word for word. If the page has no drawings or images, "
    "reply with exactly: NO_IMAGES"
)


def describe_page(page, model=MODEL, url=OLLAMA_URL, zoom=1.0):
    """Return a short description of the page's drawings, or "" if there are none."""
    text = ask_about_page(page, PROMPT, model=model, url=url, zoom=zoom)
    return "" if text.strip() == "NO_IMAGES" else text
