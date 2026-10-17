"""Turn per-page OCR text + drawing descriptions into retrieval chunks.

Strategy (one book page = one teaching unit):
  * one chunk per page; pages longer than MAX_WORDS are split recursively
    (paragraphs, then sentences) into ~TARGET_WORDS pieces with OVERLAP_WORDS overlap
  * every chunk starts with a contextual header (chapter > section, printed + PDF page)
    and ends with the page's drawing description
  * chapter/section come from the book's own table of contents, which is skipped
  * printed page numbers are read from each page and smoothed against neighbours
"""

import re
from collections import Counter

MAX_WORDS = 400        # pages longer than this are split
TARGET_WORDS = 300     # size of each piece when splitting
OVERLAP_WORDS = 50     # words repeated between consecutive pieces

TOC_LINE_RE = re.compile(r"^(\d{1,3})\s+(.+)$")
PAGE_NO_RE = re.compile(r"^\d{1,3}$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Table of contents
# --------------------------------------------------------------------------- #

def parse_toc(records, min_share=0.5):
    """Find table-of-contents pages and parse them.

    The TOC starts at the page with a "CONTENTS" line and continues over the
    following pages while at least min_share of their lines look like
    "<page number> <title>" (numbered lists in the body don't count).
    UPPERCASE titles are chapters, others are sections.
    Returns (entries sorted by printed page, set of TOC pdf pages).
    """
    entries, toc_pages = [], set()
    in_toc = False
    for r in records:
        lines = [l.strip() for l in r["text"].splitlines() if l.strip()]
        starts_toc = any(l.upper() in ("CONTENTS", "TABLE OF CONTENTS") for l in lines)
        if not (in_toc or starts_toc):
            continue
        matches = [TOC_LINE_RE.match(l) for l in lines]
        hits = [m for m in matches if m and re.search(r"[A-Za-z]", m.group(2))]
        if len(hits) < 5 or len(hits) < min_share * len(lines):
            if in_toc:
                break
            continue
        in_toc = True
        toc_pages.add(r["page"])
        for m in hits:
            title = m.group(2).strip()
            entries.append({
                "page": int(m.group(1)),
                "title": title,
                "level": "chapter" if title.upper() == title else "section",
            })
    entries.sort(key=lambda e: e["page"])  # stable: keeps TOC order within a page
    return entries, toc_pages


def locate(printed_page, entries):
    """(chapter, section) that a printed page falls under, or (None, None)."""
    if printed_page is None:
        return None, None
    chapter, chapter_start, sections = None, 0, []
    for e in entries:
        if e["page"] > printed_page:
            break
        if e["level"] == "chapter":
            chapter, chapter_start, sections = e["title"], e["page"], []
        elif e["page"] >= chapter_start:
            # several sections can start on the same page: keep all of them
            if sections and sections[0]["page"] != e["page"]:
                sections = []
            sections.append(e)
    section = "; ".join(s["title"] for s in sections) or None
    return chapter, section


# --------------------------------------------------------------------------- #
# Printed page numbers
# --------------------------------------------------------------------------- #

def read_page_number(text):
    """The printed page number: a bare number on the last two lines or first line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[-2:][::-1] + lines[:1]:
        if PAGE_NO_RE.match(line):
            return int(line)
    return None


def printed_page_numbers(records, window=4):
    """{pdf page: printed page or None}, smoothed so OCR misreads don't break it.

    The PDF-to-printed offset changes a few times (missing scans), so each page
    uses the most common offset among its neighbours within +/- window pages.
    """
    offsets = {}
    for r in records:
        n = read_page_number(r["text"])
        if n is not None:
            offsets[r["page"]] = r["page"] - n

    result = {}
    for r in records:
        p = r["page"]
        near = [offsets[q] for q in range(p - window, p + window + 1) if q in offsets]
        if not near:
            result[p] = None
            continue
        offset = Counter(near).most_common(1)[0][0]
        printed = p - offset
        result[p] = printed if printed >= 1 else None
    return result


# --------------------------------------------------------------------------- #
# Splitting long pages
# --------------------------------------------------------------------------- #

def _units(text):
    """Paragraph-level units; paragraphs longer than TARGET_WORDS become sentences."""
    units = []
    for para in (p.strip() for p in text.splitlines()):
        if not para:
            continue
        if len(para.split()) <= TARGET_WORDS:
            units.append(para)
        else:
            units.extend(s for s in SENTENCE_RE.split(para) if s.strip())
    return units


def split_text(text, max_words=MAX_WORDS, target=TARGET_WORDS, overlap=OVERLAP_WORDS):
    """Return [text] if short enough, else ~target-word pieces with overlap."""
    if len(text.split()) <= max_words:
        return [text]

    pieces, current = [], []
    for unit in _units(text):
        if current and len(" ".join(current).split()) + len(unit.split()) > target:
            pieces.append("\n".join(current))
            # start the next piece with the tail of this one
            tail_words = " ".join(current).split()[-overlap:]
            current = [" ".join(tail_words)] if overlap else []
        current.append(unit)
    if current:
        pieces.append("\n".join(current))
    return pieces


# --------------------------------------------------------------------------- #
# Building chunks
# --------------------------------------------------------------------------- #

def _header(chapter, section, printed, pdf_page):
    parts = []
    if chapter:
        parts.append(f"[Chapter: {chapter}" + (f" > {section}" if section else "") + "]")
    page = f"Page {printed} (PDF {pdf_page})" if printed else f"PDF page {pdf_page}"
    parts.append(f"[{page}]")
    return "\n".join(parts)


def build_chunks(text_records, descriptions=None, image_path_for=None, figures=None):
    """Build chunk dicts from The.json records and {page: description}.

    image_path_for: optional callable pdf_page -> path of the rendered page image.
    figures: optional {pdf_page: [{"file", "label", ...}]} cut-out drawings per page.
    """
    descriptions = descriptions or {}
    figures = figures or {}
    entries, toc_pages = parse_toc(text_records)
    printed = printed_page_numbers(text_records)

    chunks = []
    for r in text_records:
        page = r["page"]
        if page in toc_pages:
            continue
        description = (descriptions.get(page) or "").strip()
        body = r["text"].strip()
        if not body and not description:
            continue

        chapter, section = locate(printed[page], entries)
        header = _header(chapter, section, printed[page], page)
        pieces = split_text(body) if body else [""]

        for i, piece in enumerate(pieces):
            text = header + "\n" + piece
            if description:
                text += "\n\nDrawings: " + description
            chunks.append({
                "id": f"{r['source']}#p{page:04d}-{i}",
                "source": r["source"],
                "pdf_page": page,
                "printed_page": printed[page],
                "chapter": chapter,
                "section": section,
                "chunk_index": i,
                "chunk_count": len(pieces),
                "method": r["method"],
                "image_path": image_path_for(page) if image_path_for else None,
                "figures": [
                    {"file": f["file"], "label": f["label"]} for f in figures.get(page) or []
                ],
                "has_description": bool(description),
                "text": text,
            })
    return chunks
