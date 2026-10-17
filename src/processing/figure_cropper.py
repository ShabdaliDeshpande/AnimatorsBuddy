"""Cut individual drawings out of scanned pages.

1. Gemma (via Ollama) lists each drawing on the page with a rough box + short label.
2. The box is snapped to the actual ink: the page is binarised, strokes are merged
   into blobs, and the box grows to cover every blob that lies mostly inside it.
   (Gemma's boxes find the right drawing but often clip feet, arms or edges.)
3. The refined region is rendered to PNG.
"""

import json
import re

import numpy as np
import pymupdf
from scipy import ndimage

from src.processing.vision_ocr import MODEL, OLLAMA_URL, _generate

PROMPT = (
    "Find each separate drawing, illustration or photo on this page (not blocks of "
    "plain text). Return ONLY a JSON list, one object per drawing: "
    '{"box_2d": [ymin, xmin, ymax, xmax], "label": "short description"} '
    "with coordinates normalised to 0-1000. Include hand-lettered notes that belong "
    "to a drawing inside its box. If there are no drawings, return []."
)

INK_THRESHOLD = 200     # grey level below which a pixel counts as ink (0-255)
MERGE_PX = 12           # dilation radius that merges nearby strokes into one blob
MIN_BLOB_PX = 30        # ignore specks smaller than this (px, both sides)
INSIDE_SHARE = 0.5      # a blob joins a box when this share of it lies inside
PADDING_PX = 10         # margin around each crop
MAX_GROW_SHARE = 0.15   # edge growing may extend each side by this share of the box size
GAP_PX = 10             # edge growing stops after this many ink-free rows/columns
MIN_CROP_SHARE = 0.01   # drop crops smaller than this share of the page area


def locate_figures(page, model=MODEL, url=OLLAMA_URL):
    """Gemma's rough figure list: [{"box": (x0, y0, x1, y1) in pixels, "label"}]."""
    text, _ = _generate(page, PROMPT, model, url, 1.0)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    items = json.loads(match.group(0)) if match else []

    w, h = page.rect.width, page.rect.height  # pixmap at zoom 1 has the same size
    figures = []
    for item in items:
        try:
            y0, x0, y1, x1 = (float(v) for v in item["box_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
        if box[2] > box[0] and box[3] > box[1]:
            figures.append({"box": box, "label": str(item.get("label", "")).strip()})
    return figures


def ink_mask(pix):
    """Boolean array, True where the page has ink (dark or coloured strokes)."""
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3].min(axis=2) < INK_THRESHOLD


def ink_blobs(ink):
    """Bounding boxes (x0, y0, x1, y1) of merged ink blobs."""
    merged = ndimage.binary_dilation(ink, iterations=MERGE_PX)
    labels, _ = ndimage.label(merged)
    blobs = []
    for sl in ndimage.find_objects(labels):
        y, x = sl
        if x.stop - x.start >= MIN_BLOB_PX or y.stop - y.start >= MIN_BLOB_PX:
            # undo the dilation so boxes hug the real strokes
            blobs.append((x.start + MERGE_PX, y.start + MERGE_PX,
                          x.stop - MERGE_PX, y.stop - MERGE_PX))
    return blobs


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def grow_edges(box, ink, others=()):
    """Push each side outward while the strip just outside it still has ink.

    Handles drawings whose blob is merged with neighbours (shared ground line,
    overlapping figures), where the blob rule can't apply. Each side grows at
    most MAX_GROW_SHARE of the box size, stops after GAP_PX empty lines, and
    never enters another drawing's box (others).
    """
    height, width = ink.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    max_dx = int((x1 - x0) * MAX_GROW_SHARE)
    max_dy = int((y1 - y0) * MAX_GROW_SHARE)

    def blocked_row(y):
        return any(o[1] <= y <= o[3] and _overlap((x0, o[1], x1, o[3]), o) > 0.2 * (x1 - x0) * (o[3] - o[1])
                   for o in others)

    def blocked_col(x):
        return any(o[0] <= x <= o[2] and _overlap((o[0], y0, o[2], y1), o) > 0.2 * (y1 - y0) * (o[2] - o[0])
                   for o in others)

    def grow(edge, step, limit, has_ink, blocked):
        best, gap = edge, 0
        pos = edge
        while gap < GAP_PX and abs(pos - edge) < limit:
            pos += step
            if blocked(pos):
                break
            if not has_ink(pos):
                gap += 1
            else:
                best, gap = pos, 0
        return best

    y0 = grow(y0, -1, max_dy, lambda y: 0 <= y < height and ink[y, x0:x1].any(), blocked_row)
    y1 = grow(y1, +1, max_dy, lambda y: 0 <= y < height and ink[y, x0:x1].any(), blocked_row)
    x0 = grow(x0, -1, max_dx, lambda x: 0 <= x < width and ink[y0:y1, x].any(), blocked_col)
    x1 = grow(x1, +1, max_dx, lambda x: 0 <= x < width and ink[y0:y1, x].any(), blocked_col)
    return x0, y0, x1, y1


def refine_box(box, blobs, ink, others=()):
    """Cover every blob mostly inside the box, push edges out over ink, then pad.

    others: the other drawings' boxes on the page, which edge growing won't enter.
    """
    height, width = ink.shape
    x0, y0, x1, y1 = box
    for b in blobs:
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
        if _overlap(box, b) / area >= INSIDE_SHARE:
            x0, y0 = min(x0, b[0]), min(y0, b[1])
            x1, y1 = max(x1, b[2]), max(y1, b[3])
    x0, y0, x1, y1 = grow_edges((x0, y0, x1, y1), ink, others)
    return (max(x0 - PADDING_PX, 0), max(y0 - PADDING_PX, 0),
            min(x1 + PADDING_PX, width), min(y1 + PADDING_PX, height))


def crop_figures(page, out_dir, stem, model=MODEL, url=OLLAMA_URL):
    """Detect, refine and save each drawing on the page. Returns metadata records."""
    pix = page.get_pixmap()
    figures = locate_figures(page, model=model, url=url)
    ink = ink_mask(pix)
    blobs = ink_blobs(ink)
    page_area = pix.width * pix.height

    records = []
    for fig in figures:
        others = [f["box"] for f in figures if f is not fig]
        x0, y0, x1, y1 = refine_box(fig["box"], blobs, ink, others)
        if (x1 - x0) * (y1 - y0) < MIN_CROP_SHARE * page_area:
            continue
        index = len(records) + 1
        path = out_dir / f"{stem}_p{page.number + 1:04d}_fig{index:02d}.png"
        # pixmap coords == page coords at zoom 1 for these scans
        clip = pymupdf.Rect(x0, y0, x1, y1)
        page.get_pixmap(clip=clip).save(path)
        records.append({
            "page": page.number + 1,
            "index": index,
            "label": fig["label"],
            "bbox": [round(v) for v in (x0, y0, x1, y1)],
            "gemma_bbox": [round(v) for v in fig["box"]],
            "file": str(path),
        })
    return records
