"""Exact retrieval metrics against the golden reference pages (no LLM involved)."""


def ranked_pages(hits):
    """Distinct PDF pages in rank order (a page can appear in several chunks)."""
    pages = []
    for h in hits:
        if h["pdf_page"] not in pages:
            pages.append(h["pdf_page"])
    return pages


def retrieval_scores(reference_pages, hits, context_pages, ks=(1, 3, 5)):
    """Scores for one question; empty dict when there is no reference page."""
    ref = set(reference_pages)
    if not ref:
        return {}
    ranked = ranked_pages(hits)
    scores = {}
    for k in ks:
        top = set(ranked[:k])
        scores[f"hit@{k}"] = float(bool(ref & top))
        scores[f"recall@{k}"] = len(ref & top) / len(ref)
    first = next((i for i, p in enumerate(ranked, 1) if p in ref), None)
    scores["mrr"] = 1.0 / first if first else 0.0
    # did the generator get to see the reference pages (after neighbour expansion)?
    scores["context_recall_pages"] = len(ref & set(context_pages)) / len(ref)
    return scores
