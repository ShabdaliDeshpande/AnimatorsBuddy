"""Store user feedback on answers; entries with correct pages form the test set.

File: data/eval/feedback.json, a list of records keyed by "id" (re-saving the
same answer replaces its record instead of adding a duplicate).
"""

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_PATH = ROOT / "data" / "eval" / "feedback.json"


def load_feedback(path=FEEDBACK_PATH):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_pages(text):
    """'118, 119-121' -> [118, 119, 120, 121]."""
    pages = []
    for start, end in re.findall(r"(\d+)\s*(?:-\s*(\d+))?", text or ""):
        pages.extend(range(int(start), int(end or start) + 1))
    return sorted(set(pages))


def save_feedback(record, path=FEEDBACK_PATH):
    """Insert or replace the record with the same id; returns the total count."""
    records = [r for r in load_feedback(path) if r["id"] != record["id"]]
    records.append({**record, "saved_at": datetime.now().isoformat(timespec="seconds")})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    return len(records)


def build_record(item_id, question, result, rating, correct_printed, comment, settings,
                 printed_to_pdf):
    """Everything needed later to re-test this question or judge the answer."""
    correct_printed = parse_pages(correct_printed)
    return {
        "id": item_id,
        "question": question,
        "rating": rating,                                  # "good" | "bad" | None
        "correct_printed_pages": correct_printed,
        "correct_pdf_pages": sorted(
            printed_to_pdf[p] for p in correct_printed if p in printed_to_pdf
        ),
        "comment": comment.strip(),
        "answer": result.get("answer"),
        "retrieved_pdf_pages": [h["pdf_page"] for h in result.get("hits", [])],
        "context_pdf_pages": [p["pdf_page"] for p in result.get("pages", [])],
        "figures_shown": [f["file"] for f in result.get("figures", [])],
        "settings": {k: settings[k] for k in ("mode", "k", "image_pages")},
    }
