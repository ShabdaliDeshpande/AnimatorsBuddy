"""Build the golden evaluation dataset (default 200 questions) from the book.

Output: data/eval/golden.jsonl            accepted items (one JSON per line)
        data/eval/golden_candidates.jsonl every attempt, with why it was rejected

Resumable: re-running keeps accepted items and continues until the quotas are met.

Usage:
    python scripts/build_golden.py                 # full set (200)
    python scripts/build_golden.py --scale 0.05    # quick trial (~10 items)
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings.text_embedder import embed_documents  # noqa: E402
from src.eval.golden_generator import (  # noqa: E402
    DEICTIC_RE, critique, evidence_found, evidence_pages, generate_item,
    generate_out_of_scope,
)
from src.eval.ollama_json import MODEL, generate_json  # noqa: E402
from src.retrieval.hybrid_search import HybridSearcher  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
CANDIDATES_PATH = EVAL_DIR / "golden_candidates.jsonl"
TEXT_DIR = ROOT / "data" / "processed" / "text"

QUOTAS = {"factual": 80, "visual": 50, "multi_page": 35, "false_premise": 20, "out_of_scope": 15}
MIN_CRITIC = 4          # every critic score must be >= this
DUP_SIMILARITY = 0.90   # questions more similar than this to an accepted one are dropped
MIN_WORDS = 40          # pages with less text + description are skipped
MAX_ATTEMPTS = 3        # attempts per wanted item before giving up on a type
SEED = 7


def load_pages(searcher):
    """Eligible book pages: {pdf_page: {pdf_page, printed_page, chapter, section, where, text, description}}."""
    with open(TEXT_DIR / "The.json", encoding="utf-8") as f:
        texts = {r["page"]: r["text"] for r in json.load(f)}
    with open(TEXT_DIR / "The_descriptions.json", encoding="utf-8") as f:
        descs = {r["page"]: r["description"] or "" for r in json.load(f)}

    pages = {}
    for c in searcher.chunks:
        p = c["pdf_page"]
        if p in pages or c.get("printed_page") in (None, -1) or not c.get("chapter"):
            continue
        words = len(texts[p].split()) + len(descs.get(p, "").split())
        if words < MIN_WORDS:
            continue
        pages[p] = {
            "pdf_page": p,
            "printed_page": c["printed_page"],
            "chapter": c["chapter"],
            "section": c.get("section"),
            "where": " > ".join(x for x in (c["chapter"], c.get("section")) if x),
            "text": texts[p],
            "description": descs.get(p, ""),
        }
    return pages


def round_robin(pages, rng, need_description=False):
    """Page order that cycles through chapters, so every chapter gets questions."""
    by_chapter = defaultdict(list)
    for p in pages.values():
        if need_description and not p["description"]:
            continue
        by_chapter[p["chapter"]].append(p["pdf_page"])
    for lst in by_chapter.values():
        rng.shuffle(lst)
    chapters = list(by_chapter)
    rng.shuffle(chapters)
    order = []
    while any(by_chapter.values()):
        for ch in chapters:
            if by_chapter[ch]:
                order.append(by_chapter[ch].pop())
    return order


def page_pairs(pages, rng):
    """(page, other page of the same chapter): neighbours first, then random pairs."""
    pairs = []
    for p in round_robin(pages, rng):
        same = [q for q in pages if q != p and pages[q]["chapter"] == pages[p]["chapter"]]
        if not same:
            continue
        neighbour = p + 1 if p + 1 in same else None
        other = neighbour if neighbour and rng.random() < 0.6 else rng.choice(same)
        pairs.append(tuple(sorted((p, other))))
    return pairs


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Builder:
    def __init__(self, quotas, searcher):
        self.quotas = quotas
        self.searcher = searcher
        self.golden = load_jsonl(GOLDEN_PATH)
        self.vectors = (
            [np.array(v) for v in embed_documents([g["question"] for g in self.golden])]
            if self.golden else []
        )
        tried = load_jsonl(CANDIDATES_PATH)
        self.tried = {(c["type"], tuple(c["pdf_pages"])) for c in tried}

    def count(self, qtype):
        return sum(g["type"] == qtype for g in self.golden)

    def is_duplicate(self, question):
        v = np.array(embed_documents([question])[0])
        if any(float(v @ u) > DUP_SIMILARITY for u in self.vectors):
            return True, v
        return False, v

    def accept(self, record, vector):
        record["id"] = f"g{len(self.golden) + 1:03d}"
        self.golden.append(record)
        self.vectors.append(vector)
        append_jsonl(GOLDEN_PATH, record)

    def try_pages(self, qtype, page_list):
        """One generation attempt; returns (accepted?, reject reason)."""
        key = (qtype, tuple(p["pdf_page"] for p in page_list))
        self.tried.add(key)
        log = {"type": qtype, "pdf_pages": list(key[1])}
        try:
            item, material = generate_item(qtype, page_list)
        except Exception as e:
            append_jsonl(CANDIDATES_PATH, {**log, "rejected": f"generation failed: {e}"})
            return False, "generation failed"
        log.update(item)

        evidence = [e for e in item["evidence"] if e.strip()]
        deictic = DEICTIC_RE.search(item["question"])
        if deictic:
            reason = f"points at the picture ('{deictic.group(0)}')"
        elif not evidence or not all(evidence_found(e, material) for e in evidence):
            reason = "evidence not found on page"
        elif qtype == "multi_page" and len(evidence_pages(evidence, page_list)) < 2:
            reason = "evidence does not cover both pages"
        else:
            critic = critique(item, material, qtype)
            log["critic"] = critic
            scores = [critic["standalone"], critic["answerable"], critic["specific"],
                      critic["premise_false"]]
            if min(scores) < MIN_CRITIC:
                reason = f"critic {scores}: {critic['reason']}"
            else:
                dup, vector = self.is_duplicate(item["question"])
                reason = "near-duplicate question" if dup else None

        if reason:
            append_jsonl(CANDIDATES_PATH, {**log, "rejected": reason})
            return False, reason

        append_jsonl(CANDIDATES_PATH, {**log, "rejected": None})
        self.accept({
            "type": qtype,
            "question": item["question"].strip(),
            "reference_answer": item["reference_answer"].strip(),
            "evidence": evidence,
            "reference_pdf_pages": [p["pdf_page"] for p in page_list],
            "reference_printed_pages": [p["printed_page"] for p in page_list],
            "chapter": page_list[0]["chapter"],
            "section": page_list[0]["section"],
            "critic": log["critic"],
            "source": f"synthetic:{MODEL}",
        }, vector)
        return True, None

    def fill(self, qtype, candidates):
        want = self.quotas[qtype] - self.count(qtype)
        max_attempts = want * MAX_ATTEMPTS + 10   # fixed up front, not shrinking as we accept
        attempts = 0
        for page_list in candidates:
            if want <= 0 or attempts >= max_attempts:
                break
            key = (qtype, tuple(p["pdf_page"] for p in page_list))
            if key in self.tried:
                continue
            attempts += 1
            t = time.time()
            ok, reason = self.try_pages(qtype, page_list)
            want -= ok
            pages = ",".join(str(p["printed_page"]) for p in page_list)
            status = "OK " if ok else "-- "
            print(f"  {status}{qtype:13} p.{pages:9} {time.time() - t:4.0f}s  "
                  f"[{self.count(qtype)}/{self.quotas[qtype]}]" + (f"  {reason[:70]}" if reason else ""))

    def fill_out_of_scope(self):
        want = self.quotas["out_of_scope"] - self.count("out_of_scope")
        rounds = 0
        while want > 0 and rounds < 5:
            rounds += 1
            existing = [g["question"] for g in self.golden if g["type"] == "out_of_scope"]
            for q in generate_out_of_scope(want + 5, existing):
                if want <= 0:
                    break
                # make sure the book really doesn't answer it
                hits = self.searcher.search(q, k=3)
                context = "\n\n".join(h["text"][:1500] for h in hits)
                verdict = generate_json(
                    f"BOOK EXCERPTS:\n{context}\n\nQUESTION: {q}\n\n"
                    "Do these excerpts contain an answer to the question?",
                    {"type": "object", "properties": {"answered": {"type": "boolean"}},
                     "required": ["answered"]},
                )
                dup, vector = self.is_duplicate(q)
                reason = ("book answers it" if verdict["answered"]
                          else "near-duplicate question" if dup else None)
                append_jsonl(CANDIDATES_PATH, {"type": "out_of_scope", "pdf_pages": [],
                                               "question": q, "rejected": reason})
                if reason:
                    print(f"  -- out_of_scope  {q[:60]}  ({reason})")
                    continue
                self.accept({
                    "type": "out_of_scope",
                    "question": q.strip(),
                    "reference_answer": "The book does not cover this topic; the "
                                        "assistant should say so instead of guessing.",
                    "evidence": [],
                    "reference_pdf_pages": [],
                    "reference_printed_pages": [],
                    "chapter": None,
                    "section": None,
                    "critic": None,
                    "source": f"synthetic:{MODEL}",
                }, vector)
                want -= 1
                print(f"  OK out_of_scope  {q[:60]}  [{self.count('out_of_scope')}/"
                      f"{self.quotas['out_of_scope']}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply all quotas (0.05 = quick trial)")
    args = parser.parse_args()
    quotas = {k: max(1, round(v * args.scale)) for k, v in QUOTAS.items()}

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    searcher = HybridSearcher()
    pages = load_pages(searcher)
    builder = Builder(quotas, searcher)
    print(f"{len(pages)} eligible pages; target {sum(quotas.values())} items {quotas}; "
          f"{len(builder.golden)} already accepted")

    single = round_robin(pages, rng)
    visual = round_robin(pages, rng, need_description=True)
    pairs = page_pairs(pages, rng)
    builder.fill("factual", ([pages[p]] for p in single))
    builder.fill("visual", ([pages[p]] for p in visual))
    builder.fill("multi_page", ([pages[a], pages[b]] for a, b in pairs))
    builder.fill("false_premise", ([pages[p]] for p in reversed(single)))
    builder.fill_out_of_scope()

    counts = {t: builder.count(t) for t in quotas}
    print(f"\nDone: {len(builder.golden)} items {counts} -> {GOLDEN_PATH}")
    short = {t: quotas[t] - n for t, n in counts.items() if n < quotas[t]}
    if short:
        print(f"Short of quota: {short} (re-run to keep trying)")


if __name__ == "__main__":
    main()
