"""Streamlit UI for testing the RAG pipeline: answers, figures, page scans, retrieval.

Run from the project root:
    streamlit run app/ui.py
Needs Ollama running with gemma4:26b and qwen3-embedding:0.6b, and the index built
(scripts/ingest.py).
"""

import re
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.answer_formatter import FIGURES_LINE_RE  # noqa: E402
from src.pipeline import query_pipeline  # noqa: E402
from src.pipeline.query_pipeline import QueryPipeline  # noqa: E402
from src.utils.feedback import (  # noqa: E402
    FEEDBACK_PATH, build_record, load_feedback, save_feedback,
)

st.set_page_config(page_title="Animator'sBuddy", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading index ...")
def get_pipeline():
    return QueryPipeline()


@st.cache_resource
def printed_to_pdf():
    """Printed page number -> PDF page, from the chunk metadata."""
    mapping = {}
    for c in get_pipeline().searcher.chunks:
        if c.get("printed_page") not in (None, -1):
            mapping.setdefault(c["printed_page"], c["pdf_page"])
    return mapping


def feedback_form(item, settings):
    """👍/👎 + correct pages + comment under an answer; saved to data/eval/feedback.json."""
    saved = st.session_state.setdefault("saved_feedback", {})
    with st.form(key=f"fb_{item['id']}", border=True):
        cols = st.columns([1, 2, 3])
        rating = cols[0].radio(
            "Answer was", ["👍 good", "👎 bad"], index=None, key=f"rate_{item['id']}",
        )
        pages = cols[1].text_input(
            "Correct pages", key=f"pages_{item['id']}", placeholder="e.g. 118, 119",
            help="Printed page numbers (as shown in the sources) that really answer the "
                 "question. These make the question a test case.",
        )
        comment = cols[2].text_input(
            "Comment", key=f"comment_{item['id']}",
            placeholder="what was wrong or missing?",
        )
        if st.form_submit_button("Save feedback"):
            record = build_record(
                item["id"], item["question"], item["result"],
                {"👍 good": "good", "👎 bad": "bad"}.get(rating), pages, comment,
                item["settings"], printed_to_pdf(),
            )
            total = save_feedback(record)
            saved[item["id"]] = record
            st.success(f"Saved ({total} entries in {FEEDBACK_PATH.relative_to(ROOT)})")
        elif item["id"] in saved:
            r = saved[item["id"]]
            st.caption(f"Saved: {r['rating'] or 'no rating'}"
                       + (f" · correct pages {r['correct_printed_pages']}" if r["correct_printed_pages"] else ""))


def page_name(page):
    printed = page.get("printed_page")
    return f"p. {printed}" if printed not in (None, -1) else f"PDF {page['pdf_page']}"


def where(page):
    return " › ".join(x for x in (page.get("chapter"), page.get("section")) if x)


def highlight_figure_refs(text):
    """Make [F3] references stand out in the answer."""
    return re.sub(r"\[(F\d+)\]", r"**[\1]**", text)


def show_figures(figures, columns=2):
    cols = st.columns(columns)
    for i, fig in enumerate(figures):
        with cols[i % columns]:
            st.image(
                str(ROOT / fig["file"]),
                caption=f"[{fig['id']}] {page_name(fig['page'])} — {fig['label']}",
                width="stretch",
            )


def show_result(result, settings):
    """Everything below the answer text: figures, sources, debug views."""
    if result.get("figures"):
        st.markdown("#### Figures")
        show_figures(result["figures"])

    with st.expander(f"Sources ({len(result['pages'])} pages)", expanded=False):
        for page in result["pages"]:
            attached = " · 🖼 sent to Gemma" if page.get("image_attached") else ""
            st.markdown(f"**{page_name(page)}** (PDF {page['pdf_page']}) — {where(page)}{attached}")
        st.markdown("---")
        cols = st.columns(3)
        for i, page in enumerate(result["pages"]):
            with cols[i % 3]:
                st.image(
                    str(ROOT / page["image_path"]),
                    caption=f"{page_name(page)} — {where(page)}",
                    width="stretch",
                )

    if settings["show_all_figures"] and result.get("all_figures"):
        with st.expander(f"All candidate figures ({len(result['all_figures'])})"):
            show_figures(result["all_figures"], columns=3)

    if settings["show_chunks"]:
        with st.expander("Retrieved chunks (hybrid search)"):
            for i, hit in enumerate(result["hits"], 1):
                st.markdown(
                    f"**{i}. {page_name(hit)}** (PDF {hit['pdf_page']}) — {where(hit)} "
                    f"· score `{hit['score']:.4f}`"
                )
                st.text(hit["text"][:1500])


def sidebar():
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Answer with Gemma", "Search only"], index=0)
        k = st.slider("Chunks retrieved (top-k)", 1, 10, query_pipeline.TOP_K)
        image_pages = st.slider(
            "Page scans sent to Gemma", 0, 4, query_pipeline.IMAGE_PAGES,
            help="Full page images Gemma can look at while answering. 0 = text only (faster).",
        )
        show_all_figures = st.checkbox("Show all candidate figures", value=False)
        show_chunks = st.checkbox("Show retrieved chunks", value=False)
        st.markdown("---")
        records = load_feedback()
        n_good = sum(r["rating"] == "good" for r in records)
        n_bad = sum(r["rating"] == "bad" for r in records)
        n_test = sum(bool(r["correct_pdf_pages"]) for r in records)
        st.markdown(f"**Feedback:** {len(records)} saved · 👍 {n_good} · 👎 {n_bad}  \n"
                    f"**Test cases** (with correct pages): {n_test}")
        if FEEDBACK_PATH.exists():
            st.download_button("Download feedback.json", FEEDBACK_PATH.read_bytes(),
                               file_name="feedback.json", mime="application/json")
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()
        st.caption("Models: gemma4:26b (answers) · qwen3-embedding:0.6b (search)")
    return {
        "mode": mode, "k": k, "image_pages": image_pages,
        "show_all_figures": show_all_figures, "show_chunks": show_chunks,
    }


def run_answer(pipeline, question, settings):
    with st.status("Searching the book ...", expanded=False) as status:
        context = pipeline.prepare(question, k=settings["k"], image_pages=settings["image_pages"])
        status.update(
            label=f"Found {len(context['pages'])} pages · asking Gemma "
                  f"({len(context['image_paths'])} page scans attached)",
            state="complete",
        )

    placeholder = st.empty()
    reply = ""
    for piece in pipeline.stream(context):
        reply += piece
        # hide the trailing FIGURES: line while streaming
        placeholder.markdown(highlight_figure_refs(FIGURES_LINE_RE.sub("", reply)) + " ▌")
    result = pipeline.finish(context, reply)
    placeholder.markdown(highlight_figure_refs(result["answer"]))
    return result


def run_search(pipeline, question, settings):
    hits = pipeline.searcher.search(question, k=settings["k"])
    pages, _ = pipeline.gather_pages(hits)
    st.markdown(f"Top {len(hits)} chunks for this question:")
    figures = []
    for i, hit in enumerate(hits, 1):
        st.markdown(f"**{i}. {page_name(hit)}** (PDF {hit['pdf_page']}) — {where(hit)}")
        text, _, drawings = hit["text"].split("\n", 2)[-1].partition("\n\nDrawings: ")
        st.caption(text[:400] + ("…" if len(text) > 400 else ""))
        if drawings:
            st.caption(f"🖼 {drawings}")
        for fig in hit.get("figures") or []:
            figures.append({"id": f"F{len(figures) + 1}", "page": hit, **fig})
    return {"answer": None, "figures": figures, "all_figures": [], "pages": pages, "hits": hits}


def main():
    settings = sidebar()
    st.title("🎬 Animator'sBuddy")
    st.caption("Ask animation questions, get answers with the book's own drawings. "
               "Local RAG over *The Animator's Survival Kit*: Gemma-read text, drawing "
               "descriptions and cut-out figures.")

    if "history" not in st.session_state:
        st.session_state.history = []

    pipeline = get_pipeline()

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            if item["result"]["answer"]:
                st.markdown(highlight_figure_refs(item["result"]["answer"]))
            show_result(item["result"], settings)
            if "id" in item:  # answers from before feedback existed have no id
                feedback_form(item, settings)

    question = st.chat_input("Ask about animation, e.g. how do I animate a double bounce walk?")
    if not question:
        if not st.session_state.history:
            st.info("Try: *how do I make a character feel heavy?* · *what is a breakdown?* · "
                    "*how many drawings does a run need?*")
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            if settings["mode"] == "Search only":
                result = run_search(pipeline, question, settings)
            else:
                result = run_answer(pipeline, question, settings)
        except Exception as e:  # Ollama down, model missing, ...
            st.error(f"Something went wrong: {e}. Is Ollama running?")
            return
        show_result(result, settings)
        item = {
            "id": uuid.uuid4().hex[:12],
            "question": question,
            "result": result,
            "settings": dict(settings),
        }
        st.session_state.history.append(item)
        feedback_form(item, settings)


main()
