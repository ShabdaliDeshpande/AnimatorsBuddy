"""Add a searchable text layer to scanned PDFs with OCRmyPDF (Tesseract).

The output PDF looks identical to the input, but every page gets invisible,
positioned text behind the scan, so PyMuPDF's page.get_text() returns real text.

Requires the Tesseract binary on PATH. Ghostscript is not needed: pages are
rasterized with pypdfium2 and the output is a plain (non PDF/A) PDF.
"""

import os
import re
from pathlib import Path

import ocrmypdf

MIN_CHARS = 150           # OCR text shorter than this -> page is "hard"
MIN_CLEAN_RATIO = 0.75    # share of tokens that look like real words/numbers

WORD_RE = re.compile(r"^[(\"']*([A-Za-z][A-Za-z'\-]*|\d+([.,/]\d+)*)[)\"'.,;:!?\-]*$")


def ocr_pdf(pdf_path, out_dir, language="eng", oversample=150, jobs=None, force=False):
    """OCR one PDF into out_dir/<same name>.pdf and return the output path.

    Pages that already contain text are skipped (skip_text), so re-running on a
    partially OCR'd file is safe.

    oversample: scans are upsampled to this DPI before OCR. The scans in data/raw
    are stored with a 72 DPI page size (real resolution is ~150-200 DPI), so 150
    here means roughly 2x upscaling, which is about 300 real DPI for Tesseract.
    The original page images are kept in the output; only the OCR input is upscaled.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / pdf_path.name

    if out_path.exists() and not force:
        return out_path

    ocrmypdf.ocr(
        pdf_path,
        out_path,
        language=language,
        skip_text=True,
        oversample=oversample,
        optimize=1,
        rasterizer="pypdfium",  # no Ghostscript needed
        output_type="pdf",      # plain PDF; PDF/A would need Ghostscript
        jobs=jobs or os.cpu_count(),
        progress_bar=True,
    )
    return out_path


def clean_ratio(text):
    """Share of whitespace-separated tokens that look like real words or numbers."""
    tokens = text.split()
    if not tokens:
        return 0.0
    good = sum(
        1 for t in tokens
        if WORD_RE.match(t) and (not t[0].isalpha() or re.search(r"[aeiouyAEIOUY]", t) or len(t) == 1)
    )
    return good / len(tokens)


def is_hard_page(text, min_chars=MIN_CHARS, min_clean_ratio=MIN_CLEAN_RATIO):
    """True when Tesseract output is too short or too garbled to trust."""
    text = text.strip()
    return len(text) < min_chars or clean_ratio(text) < min_clean_ratio
