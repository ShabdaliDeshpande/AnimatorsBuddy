"""Cut out each drawing on pages that have drawings (Gemma boxes + ink snapping).

Input:  data/raw/<name>.pdf
        data/processed/text/<name>_descriptions.json   (pages with "" are skipped)
Output: data/processed/figures/<name>_p0001_fig01.png ...
        data/processed/text/<name>_figures.json, one record per page:
            {"source", "page", "figures": [{"index", "label", "bbox", "gemma_bbox", "file"}]}
            ("figures": null until the page is done)

Saved after every page, so an interrupted run resumes where it stopped.
Re-run scripts/chunk_pages.py and scripts/ingest.py afterwards to attach the crops.

Usage:
    python scripts/crop_figures.py                     # all PDFs in data/raw
    python scripts/crop_figures.py --pages 125,174     # redo these pages
    python scripts/crop_figures.py --force             # redo every page
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
from src.processing.figure_cropper import crop_figures  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
TEXT_DIR = ROOT / "data" / "processed" / "text"
FIGURE_DIR = ROOT / "data" / "processed" / "figures"


def process(pdf, args, forced_pages):
    stem = pdf.stem
    out_path = TEXT_DIR / f"{stem}_figures.json"
    done = {}
    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            done = {r["page"]: r for r in json.load(f)}

    desc_path = TEXT_DIR / f"{stem}_descriptions.json"
    with open(desc_path, encoding="utf-8") as f:
        has_drawings = {r["page"]: bool(r["description"]) for r in json.load(f)}

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        records = [
            done.get(n) or {"source": pdf.name, "page": n, "figures": None}
            for n in range(1, len(doc) + 1)
        ]
        for r in records:  # pages without drawings need no Gemma call
            if not has_drawings.get(r["page"]) and r["page"] not in forced_pages:
                r["figures"] = []
        pending = [
            r for r in records if r["figures"] is None or r["page"] in forced_pages
        ]
        print(f"Cropping {pdf.name}: {len(pending)} of {len(records)} pages to do")

        for i, r in enumerate(pending, start=1):
            t = time.time()
            for old in FIGURE_DIR.glob(f"{stem}_p{r['page']:04d}_fig*.png"):
                old.unlink()
            try:
                figures = crop_figures(doc[r["page"] - 1], FIGURE_DIR, stem)
            except Exception as e:  # leave as None so the next run retries it
                print(f"  page {r['page']}: failed ({e})")
                continue
            for fig in figures:
                fig["file"] = str(Path(fig["file"]).relative_to(ROOT))
            r["figures"] = figures
            save(records, out_path)
            print(f"  [{i}/{len(pending)}] page {r['page']}: {len(figures)} figures "
                  f"({time.time() - t:.0f}s)")

    save(records, out_path)
    n_figs = sum(len(r["figures"] or []) for r in records)
    n_fail = sum(r["figures"] is None for r in records)
    print(f"  {n_figs} figures from {sum(bool(r['figures']) for r in records)} pages, "
          f"{n_fail} failed -> {out_path}")


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
