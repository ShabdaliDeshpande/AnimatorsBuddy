"""Search the index from the command line (retrieval only, no answer generation yet).

Usage:
    python scripts/query.py "how do I make a walk feel heavy?"
    python scripts/query.py "Ken Harris" --mode keyword -k 3
    python scripts/query.py "what is a breakdown?" --mode dense
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_search import HybridSearcher  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    parser.add_argument("--width", type=int, default=300, help="characters shown per field")
    args = parser.parse_args()

    for i, hit in enumerate(HybridSearcher().search(args.question, k=args.k, mode=args.mode), 1):
        printed = hit["printed_page"] if hit["printed_page"] not in (None, -1) else "-"
        where = " > ".join(x for x in (hit["chapter"], hit["section"]) if x)
        print(f"{i}. PDF {hit['pdf_page']} (p.{printed})  score {hit['score']:.3f}  {where}")
        body = hit["text"].split("\n", 2)[-1]
        text, _, drawings = body.partition("\n\nDrawings: ")
        print(f"   text:     {text.replace(chr(10), ' ')[:args.width]}")
        if drawings:
            print(f"   drawings: {drawings[:args.width]}")
        print(f"   page:     {hit['image_path']}")
        for fig in hit.get("figures") or []:
            print(f"   figure:   {fig['file']}  ({fig['label'][:80]})")


if __name__ == "__main__":
    main()
