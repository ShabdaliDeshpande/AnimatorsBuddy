"""Build the Gemma prompt: retrieved pages as text, page images, numbered figures."""

SYSTEM_PROMPT = """You are an animation tutor answering questions from the book \
"The Animator's Survival Kit" by Richard Williams.

Rules:
- Answer ONLY from the book pages provided below and the attached page images. \
If they don't contain the answer, say so plainly.
- Explain clearly and practically, like a teacher: steps, timing numbers, \
key poses, what to watch out for.
- Cite pages inline as (p. 118). Use the printed page number given for each page.
- When a figure helps, point to it inline by its id, e.g. [F3]. Only use figure \
ids from the list provided.
- End with a line exactly like:  FIGURES: F1, F3   (the figures most worth \
looking at, best first, at most 4; or FIGURES: none)"""


def page_label(page):
    printed = page.get("printed_page")
    if printed not in (None, -1):
        return f"p. {printed} (PDF {page['pdf_page']})"
    return f"PDF page {page['pdf_page']}"


def build_messages(question, pages, figures, image_paths):
    """pages: [{"pdf_page", "printed_page", "chapter", "section", "text"}] in reading order
    figures: [{"id", "page", "label", "file"}]
    image_paths: page images attached for the model to look at
    """
    blocks = []
    for page in pages:
        where = " > ".join(x for x in (page.get("chapter"), page.get("section")) if x)
        blocks.append(f"=== {page_label(page)} | {where} ===\n{page['text']}")

    fig_lines = [
        f"[{f['id']}] {page_label(f['page'])}: {f['label']}" for f in figures
    ] or ["(no figures)"]

    attached = ", ".join(page_label(p) for p in pages if p.get("image_attached"))
    user = (
        "BOOK PAGES:\n\n" + "\n\n".join(blocks)
        + "\n\nFIGURES YOU CAN REFER TO:\n" + "\n".join(fig_lines)
        + (f"\n\nAttached images are the full scans of: {attached}." if attached else "")
        + f"\n\nQUESTION: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user, "images": list(image_paths)},
    ]
