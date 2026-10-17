"""Question -> retrieved pages -> Gemma answer with page citations and figures.

1. Hybrid search finds the best chunks.
2. The top pages are expanded with their neighbouring pages (same chapter),
   because one movement often spans facing pages.
3. Gemma gets the page text + descriptions, the full scans of the top pages,
   and a numbered list of cut-out figures it may point to.
4. The reply is split into the answer and the figures Gemma picked.
"""

from src.generation.answer_formatter import parse_reply
from src.generation.llm import MODEL, chat, chat_stream
from src.generation.prompt_builder import build_messages
from src.retrieval.hybrid_search import HybridSearcher

TOP_K = 5             # chunks retrieved
EXPAND_TOP = 2        # hits whose previous/next page is added
MAX_PAGES = 8         # pages of text sent to the model
IMAGE_PAGES = 3       # full page scans attached (best hits first)
MAX_FIGURES = 15      # figures listed for the model to choose from


class QueryPipeline:
    def __init__(self, searcher=None, model=MODEL):
        self.searcher = searcher or HybridSearcher()
        self.model = model
        # all chunks of each page, in order, for neighbour expansion
        self.page_chunks = {}
        for c in self.searcher.chunks:
            self.page_chunks.setdefault(c["pdf_page"], []).append(c)
        for chunks in self.page_chunks.values():
            chunks.sort(key=lambda c: c["chunk_index"])

    def _page(self, pdf_page):
        """One record per page: metadata of its first chunk + text of all its chunks."""
        chunks = self.page_chunks[pdf_page]
        first = chunks[0]
        if len(chunks) == 1:
            text = first["text"]
        else:  # split pages: drop the repeated header/description from later pieces
            text = "\n".join(c["text"] for c in chunks)
        body = text.split("\n", 2)[-1] if text.startswith("[") else text
        return {**first, "text": body}

    def gather_pages(self, hits):
        """Hit pages (best first) plus neighbours of the top hits, capped at MAX_PAGES."""
        ordered = []
        for h in hits:
            if h["pdf_page"] not in ordered:
                ordered.append(h["pdf_page"])
        best = list(ordered)
        for h in hits[:EXPAND_TOP]:
            for n in (h["pdf_page"] - 1, h["pdf_page"] + 1):
                if n in self.page_chunks and n not in ordered:
                    if self.page_chunks[n][0]["chapter"] == h["chapter"]:
                        ordered.append(n)
        ordered = ordered[:MAX_PAGES]
        return [self._page(p) for p in ordered], best

    def prepare(self, question, k=TOP_K, image_pages=IMAGE_PAGES):
        """Retrieve and build the prompt; returns everything except the reply."""
        hits = self.searcher.search(question, k=k)
        pages, best = self.gather_pages(hits)

        # images + figures: from the best-ranked pages first
        n_images = image_pages
        image_pages = [p for p in best if p in {pg["pdf_page"] for pg in pages}][:n_images]
        by_page = {p["pdf_page"]: p for p in pages}
        for p in image_pages:
            by_page[p]["image_attached"] = True
        image_paths = [by_page[p]["image_path"] for p in image_pages]

        figures = []
        for p in best + [pg["pdf_page"] for pg in pages]:
            for fig in by_page.get(p, {}).get("figures") or []:
                if len(figures) >= MAX_FIGURES:
                    break
                if all(f["file"] != fig["file"] for f in figures):
                    figures.append({
                        "id": f"F{len(figures) + 1}",
                        "page": by_page[p],
                        "label": fig["label"],
                        "file": fig["file"],
                    })

        reading_order = sorted(pages, key=lambda p: p["pdf_page"])
        return {
            "question": question,
            "messages": build_messages(question, reading_order, figures, image_paths),
            "all_figures": figures,
            "image_paths": image_paths,
            "pages": reading_order,
            "hits": hits,
        }

    def stream(self, context):
        """Yield the reply for a prepare() result, piece by piece."""
        return chat_stream(context["messages"], model=self.model)

    def finish(self, context, reply):
        """Split a full reply into answer + chosen figures."""
        answer, chosen = parse_reply(reply, context["all_figures"])
        return {**context, "answer": answer, "figures": chosen}

    def answer(self, question, k=TOP_K, on_token=None):
        context = self.prepare(question, k=k)
        reply = chat(context["messages"], model=self.model, on_token=on_token)
        return self.finish(context, reply)
