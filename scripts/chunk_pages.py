"""Build retrieval chunks from OCR text + drawing descriptions.

Input:  data/processed/text/<name>.json                (OCR text, from ocr_pdfs.py)
        data/processed/text/<name>_descriptions.json   (optional, from describe_pages.py)
        data/processed/text/<name>_figures.json        (optional, from crop_figures.py)
        data/raw/<name>.pdf                            (to render page images)
Output: data/processed/chunks/<name>.jsonl             (one chunk per line)
        data/processed/pages/<name>_p0001.png ...      (page images shown with answers)

Safe to re-run: it rebuilds the chunks from scratch and only renders missing images.
Re-run after describe_pages.py finishes so every chunk gets its description.

Usage:
    python scripts/chunk_pages.py                      # every <name>.json in data/processed/text
    python scripts/chunk_pages.py The                  # one book
"""

import json
import statistics
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processing.chunker import build_chunks  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
TEXT_DIR = ROOT / "data" / "processed" / "text"
CHUNK_DIR = ROOT / "data" / "processed" / "chunks"
PAGE_DIR = ROOT / "data" / "processed" / "pages"


def render_pages(pdf_path, stem):
    """Render every page at its scanned size; skip images that already exist."""
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            out = PAGE_DIR / f"{stem}_p{page.number + 1:04d}.png"
            if not out.exists():
                page.get_pixmap().save(out)


def process(stem):
    with open(TEXT_DIR / f"{stem}.json", encoding="utf-8") as f:
        text_records = json.load(f)

    desc_path = TEXT_DIR / f"{stem}_descriptions.json"
    descriptions = {}
    if desc_path.exists():
        with open(desc_path, encoding="utf-8") as f:
            descriptions = {r["page"]: r["description"] for r in json.load(f)}

    fig_path = TEXT_DIR / f"{stem}_figures.json"
    figures = {}
    if fig_path.exists():
        with open(fig_path, encoding="utf-8") as f:
            figures = {r["page"]: r["figures"] for r in json.load(f)}

    pdf_path = RAW_DIR / text_records[0]["source"]
    render_pages(pdf_path, stem)

    chunks = build_chunks(
        text_records,
        descriptions,
        image_path_for=lambda p: str((PAGE_DIR / f"{stem}_p{p:04d}.png").relative_to(ROOT)),
        figures=figures,
    )

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNK_DIR / f"{stem}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    words = sorted(len(c["text"].split()) for c in chunks)
    pages = {c["pdf_page"] for c in chunks}
    split_pages = {c["pdf_page"] for c in chunks if c["chunk_count"] > 1}
    n_desc = sum(c["has_description"] for c in chunks)
    print(f"{stem}: {len(chunks)} chunks from {len(pages)} pages "
          f"({len(split_pages)} pages split) -> {out_path}")
    print(f"  words per chunk: min {words[0]}, median {statistics.median(words):.0f}, "
          f"max {words[-1]}  (~{int(words[-1] * 1.35)} tokens max)")
    print(f"  chunks with a drawing description: {n_desc}/{len(chunks)}")
    print(f"  chunks with cut-out figures: {sum(bool(c['figures']) for c in chunks)}/{len(chunks)}")
    print(f"  chunks without chapter: {sum(c['chapter'] is None for c in chunks)}, "
          f"without printed page: {sum(c['printed_page'] is None for c in chunks)}")


def main():
    stems = sys.argv[1:] or sorted(
        p.stem for p in TEXT_DIR.glob("*.json")
        if not p.stem.endswith(("_descriptions", "_figures"))
    )
    if not stems:
        print(f"No OCR text found in {TEXT_DIR}; run scripts/ocr_pdfs.py first")
        return
    for stem in stems:
        process(stem)


if __name__ == "__main__":
    main()
