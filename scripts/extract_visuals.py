"""Extract images and figures from every PDF in data/raw.

Usage:
    python scripts/extract_visuals.py                 # all PDFs in data/raw
    python scripts/extract_visuals.py data/raw/book.pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.figure_extractor import extract_figures  # noqa: E402
from src.ingestion.image_extractor import extract_images  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
IMAGES_DIR = ROOT / "data" / "processed" / "images"
FIGURES_DIR = ROOT / "data" / "processed" / "figures"
METADATA_DIR = ROOT / "data" / "processed" / "metadata"


def main():
    pdfs = [Path(p) for p in sys.argv[1:]] or sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}")
        return

    for pdf in pdfs:
        print(f"Processing {pdf.name} ...")
        images = extract_images(pdf, IMAGES_DIR, METADATA_DIR)
        figures = extract_figures(pdf, FIGURES_DIR, METADATA_DIR)
        print(f"  {len(images)} images  -> {IMAGES_DIR}")
        print(f"  {len(figures)} figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
