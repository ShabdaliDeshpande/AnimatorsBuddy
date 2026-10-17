"""OCR scanned PDFs: Tesseract (OCRmyPDF) first, then Gemma for pages it can't read.

Pass 1: data/raw/<name>.pdf -> data/processed/ocr/<name>.pdf  (searchable text layer)
Pass 2: pages whose OCR text is short or garbled are re-read by gemma4:26b via Ollama.
Result: data/processed/text/<name>.json, one record per page:
    {"source", "page", "method": "tesseract" | "gemma", "text", "ocr_text"}

The JSON is saved after every Gemma page, so an interrupted run resumes where it stopped.

Usage:
    python scripts/ocr_pdfs.py                          # all PDFs in data/raw
    python scripts/ocr_pdfs.py data/raw/book.pdf
    python scripts/ocr_pdfs.py --force                  # redo Tesseract pass and all Gemma pages
    python scripts/ocr_pdfs.py --gemma-all              # send every page to Gemma
    python scripts/ocr_pdfs.py --pages 12,40-45         # force these pages through Gemma
    python scripts/ocr_pdfs.py --no-gemma               # Tesseract only

Needs Tesseract on PATH, `pip install ocrmypdf`, and Ollama running
with a vision model (ollama pull gemma4:26b).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processing.ocr import is_hard_page, ocr_pdf  # noqa: E402
from src.processing.vision_ocr import transcribe_page  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OCR_DIR = ROOT / "data" / "processed" / "ocr"
TEXT_DIR = ROOT / "data" / "processed" / "text"


def parse_pages(spec):
    """'12,40-45' -> {12, 40, 41, 42, 43, 44, 45} (1-based page numbers)."""
    pages = set()
    for part in filter(None, spec.split(",")):
        start, _, end = part.partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return pages


def save(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def process(pdf, args, forced_pages):
    print(f"[1/2] Tesseract OCR {pdf.name} ...")
    ocr_path = ocr_pdf(pdf, OCR_DIR, force=args.force)

    out_path = TEXT_DIR / f"{pdf.stem}.json"
    previous = {}
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            previous = {r["page"]: r for r in json.load(f)}

    with pymupdf.open(ocr_path) as doc:
        records = []
        for page in doc:
            page_no = page.number + 1
            ocr_text = page.get_text().strip()
            records.append({
                "source": pdf.name,
                "page": page_no,
                "method": "tesseract",
                "text": ocr_text,
                "ocr_text": ocr_text,
            })

        if args.no_gemma:
            todo = []
        else:
            todo = [
                r for r in records
                if args.gemma_all or r["page"] in forced_pages or is_hard_page(r["ocr_text"])
            ]

        # Reuse Gemma results from an earlier (interrupted) run
        for r in todo:
            prev = previous.get(r["page"])
            if prev and prev["method"] == "gemma" and r["page"] not in forced_pages:
                r["method"], r["text"] = "gemma", prev["text"]
        pending = [r for r in todo if r["method"] != "gemma"]

        print(f"[2/2] Gemma: {len(todo)} hard pages, {len(pending)} still to do")
        save(records, out_path)
        for i, r in enumerate(pending, start=1):
            t = time.time()
            try:
                text = transcribe_page(doc[r["page"] - 1])
            except Exception as e:  # keep the Tesseract text and move on
                print(f"  page {r['page']}: Gemma failed ({e}), keeping Tesseract text")
                continue
            r["method"], r["text"] = "gemma", text
            save(records, out_path)
            print(f"  [{i}/{len(pending)}] page {r['page']} ({time.time() - t:.0f}s)")

    n_gemma = sum(r["method"] == "gemma" for r in records)
    n_empty = sum(not r["text"] for r in records)
    print(f"  {len(records)} pages: {len(records) - n_gemma} tesseract, {n_gemma} gemma, "
          f"{n_empty} empty -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--force", action="store_true", help="redo everything")
    parser.add_argument("--gemma-all", action="store_true", help="send every page to Gemma")
    parser.add_argument("--pages", default="", help="pages to force through Gemma, e.g. 12,40-45")
    parser.add_argument("--no-gemma", action="store_true", help="Tesseract only")
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}")
        return

    forced_pages = parse_pages(args.pages)
    for pdf in pdfs:
        process(pdf, args, forced_pages)


if __name__ == "__main__":
    main()
