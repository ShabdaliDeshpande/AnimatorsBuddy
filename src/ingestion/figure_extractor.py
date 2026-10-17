"""Extract captioned figures from a PDF.

A "figure" is found by locating its caption ("Figure 3.2 ...", "Fig. 4 ...")
and grouping the visual elements (embedded images and vector drawings) next to it.
The page region is then rendered to PNG, so vector charts and diagrams that are
not stored as images are captured too.
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?)\s*(\d+(?:[.\-]\d+)*)", re.IGNORECASE)

MIN_VISUAL_SIZE = 20       # ignore drawings smaller than this (pt) in both directions
MAX_CAPTION_GAP = 150      # max vertical distance (pt) between caption and figure
PADDING = 4                # extra margin (pt) around the rendered figure


def _visual_rects(page):
    """Bounding boxes of images and drawing clusters on the page."""
    rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]

    if hasattr(page, "cluster_drawings"):
        rects += page.cluster_drawings()
    else:
        rects += [d["rect"] for d in page.get_drawings()]

    page_area = page.rect.get_area()
    return [
        r for r in rects
        if (r.width >= MIN_VISUAL_SIZE or r.height >= MIN_VISUAL_SIZE)
        and r.get_area() < 0.95 * page_area  # skip full-page backgrounds/frames
    ]


def _caption_blocks(page):
    """Text blocks that start with a figure caption label."""
    captions = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        match = CAPTION_RE.match(text)
        if match:
            captions.append({
                "rect": fitz.Rect(x0, y0, x1, y1),
                "label": match.group(2),
                "text": " ".join(text.split()),
            })
    return sorted(captions, key=lambda c: c["rect"].y0)


def _overlaps_horizontally(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1


def _figure_region(caption_rect, visuals, upper_limit, lower_limit):
    """Union of visuals directly above the caption, or below if none above."""
    above = [
        v for v in visuals
        if v.y1 <= caption_rect.y0 + 2
        and v.y0 >= upper_limit
        and caption_rect.y0 - v.y1 <= MAX_CAPTION_GAP
        and _overlaps_horizontally(v, caption_rect)
    ]
    below = [
        v for v in visuals
        if v.y0 >= caption_rect.y1 - 2
        and v.y1 <= lower_limit
        and v.y0 - caption_rect.y1 <= MAX_CAPTION_GAP
        and _overlaps_horizontally(v, caption_rect)
    ]
    chosen = above or below
    if not chosen:
        return None

    region = fitz.Rect(chosen[0])
    for r in chosen[1:]:
        region |= r
    return region


def extract_figures(pdf_path, out_dir, metadata_dir=None, dpi=200, include_caption=True):
    """Render every captioned figure to PNG and return a list of metadata dicts."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1
            captions = _caption_blocks(page)
            if not captions:
                continue
            visuals = _visual_rects(page)

            for i, cap in enumerate(captions):
                # Don't let a figure reach past the neighbouring captions
                upper = captions[i - 1]["rect"].y1 if i > 0 else page.rect.y0
                lower = captions[i + 1]["rect"].y0 if i + 1 < len(captions) else page.rect.y1

                region = _figure_region(cap["rect"], visuals, upper, lower)
                if region is None:
                    continue

                clip = region | cap["rect"] if include_caption else region
                clip = (clip + (-PADDING, -PADDING, PADDING, PADDING)) & page.rect

                label = cap["label"].replace(".", "_").replace("-", "_")
                filename = f"{pdf_path.stem}_p{page_no:04d}_fig{label}.png"
                page.get_pixmap(clip=clip, dpi=dpi).save(out_dir / filename)

                records.append({
                    "type": "figure",
                    "source": pdf_path.name,
                    "page": page_no,
                    "label": f"Figure {cap['label']}",
                    "caption": cap["text"],
                    "file": str(out_dir / filename),
                    "bbox": [round(v, 2) for v in clip],
                })

    if metadata_dir:
        metadata_dir = Path(metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_dir / f"{pdf_path.stem}_figures.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    return records
