"""Describe the drawings on every page with Gemma (via Ollama).

Input:  data/raw/<name>.pdf
Output: data/processed/text/<name>_descriptions.json, one record per page:
    {"source", "page", "description"}      ("" when the page has no drawings)

Kept separate from <name>.json (OCR text) so re-running OCR never wipes the
descriptions. Join the two files on "page".

The JSON is saved after every page, so an interrupted run resumes where it stopped.

Usage:
    python scripts/describe_pages.py                    # all PDFs in data/raw
    python scripts/describe_pages.py data/raw/book.pdf
    python scripts/describe_pages.py --pages 12,40-45   # redo these pages
    python scripts/describe_pages.py --force            # redo every page
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ocr_pdfs import parse_pages, save  # noqa: E402
from src.processing.image_captioner import describe_page  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
TEXT_DIR = ROOT / "data" / "processed" / "text"


def process(pdf, args, forced_pages):
    out_path = TEXT_DIR / f"{pdf.stem}_descriptions.json"
    done = {}
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            done = {r["page"]: r for r in json.load(f)}

    with pymupdf.open(pdf) as doc:
        records = [
            done.get(n) or {"source": pdf.name, "page": n, "description": None}
            for n in range(1, len(doc) + 1)
        ]
        pending = [
            r for r in records if r["description"] is None or r["page"] in forced_pages
        ]
        print(f"Describing {pdf.name}: {len(pending)} of {len(records)} pages to do")

        for i, r in enumerate(pending, start=1):
            t = time.time()
            try:
                r["description"] = describe_page(doc[r["page"] - 1])
            except Exception as e:  # leave as None so the next run retries it
                print(f"  page {r['page']}: Gemma failed ({e})")
                continue
            save(records, out_path)
            print(f"  [{i}/{len(pending)}] page {r['page']} ({time.time() - t:.0f}s)")

    save(records, out_path)
    n_img = sum(bool(r["description"]) for r in records)
    n_none = sum(r["description"] == "" for r in records)
    n_fail = sum(r["description"] is None for r in records)
    print(f"  {n_img} described, {n_none} without drawings, {n_fail} failed -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--force", action="store_true", help="redo every page")
    parser.add_argument("--pages", default="", help="pages to redo, e.g. 12,40-45")
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
