"""Split Gemma's reply into answer text + chosen figures, and format sources."""

import re

FIGURES_LINE_RE = re.compile(r"^\s*FIGURES:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
FIGURE_REF_RE = re.compile(r"\[?(F\d+)\]?")


def parse_reply(reply, figures):
    """Return (answer text without the FIGURES line, [chosen figure dicts]).

    Figures come from the final FIGURES: line; if it is missing, from [Fn]
    references in the text. Unknown ids are ignored.
    """
    by_id = {f["id"]: f for f in figures}
    match = FIGURES_LINE_RE.search(reply)
    if match:
        answer = (reply[:match.start()] + reply[match.end():]).strip()
        ids = FIGURE_REF_RE.findall(match.group(1))
    else:
        answer = reply.strip()
        ids = FIGURE_REF_RE.findall(reply)

    chosen, seen = [], set()
    for fid in ids:
        if fid in by_id and fid not in seen:
            chosen.append(by_id[fid])
            seen.add(fid)
    return answer, chosen


def format_sources(pages):
    lines = []
    for p in pages:
        printed = p.get("printed_page")
        page = f"p. {printed}" if printed not in (None, -1) else "front matter"
        where = " > ".join(x for x in (p.get("chapter"), p.get("section")) if x)
        lines.append(f"  {page:>8}  (PDF {p['pdf_page']:>3})  {where}")
    return "\n".join(lines)
