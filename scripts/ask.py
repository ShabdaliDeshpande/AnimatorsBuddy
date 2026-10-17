"""Ask a question about the book and get a cited answer with figures.

Usage:
    python scripts/ask.py "how do I animate a double bounce walk?"
    python scripts/ask.py "what is a breakdown?" --open      # also open the figures
    python scripts/ask.py                                     # interactive: ask repeatedly
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.answer_formatter import format_sources  # noqa: E402
from src.pipeline.query_pipeline import QueryPipeline  # noqa: E402


def ask(pipeline, question, open_figures=False):
    print()
    result = pipeline.answer(
        question, on_token=lambda t: print(t, end="", flush=True)
    )
    print("\n")
    print("Sources:")
    print(format_sources(result["pages"]))
    if result["figures"]:
        print("\nFigures:")
        for f in result["figures"]:
            printed = f["page"].get("printed_page")
            page = f"p. {printed}" if printed not in (None, -1) else f"PDF {f['page']['pdf_page']}"
            print(f"  [{f['id']}] {page}: {f['label']}")
            print(f"       {ROOT / f['file']}")
            if open_figures and hasattr(os, "startfile"):
                os.startfile(ROOT / f["file"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--open", action="store_true", help="open the chosen figures")
    args = parser.parse_args()

    pipeline = QueryPipeline()
    if args.question:
        ask(pipeline, args.question, args.open)
        return

    print("Ask about the book (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(pipeline, question, args.open)


if __name__ == "__main__":
    main()
