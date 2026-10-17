"""Extract embedded raster images (photos, scans, bitmaps) from a PDF."""

import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_images(pdf_path, out_dir, metadata_dir=None, min_width=100, min_height=100):
    """Save every embedded image in the PDF and return a list of metadata dicts.

    Images smaller than min_width x min_height (icons, bullets, logos) are skipped.
    Images reused on several pages (xref repeated) are saved once.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    seen_xrefs = set()

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                if xref in seen_xrefs:
                    continue

                pix = fitz.Pixmap(doc, xref)
                if pix.width < min_width or pix.height < min_height:
                    continue
                seen_xrefs.add(xref)

                # Convert CMYK / other colorspaces to RGB so PNG export works
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                filename = f"{pdf_path.stem}_p{page_no:04d}_img{img_index:02d}.png"
                pix.save(out_dir / filename)

                rects = page.get_image_rects(xref)
                bbox = [round(v, 2) for v in rects[0]] if rects else None

                records.append({
                    "type": "image",
                    "source": pdf_path.name,
                    "page": page_no,
                    "file": str(out_dir / filename),
                    "width": pix.width,
                    "height": pix.height,
                    "bbox": bbox,
                    "xref": xref,
                })

    if metadata_dir:
        metadata_dir = Path(metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_dir / f"{pdf_path.stem}_images.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    return records
