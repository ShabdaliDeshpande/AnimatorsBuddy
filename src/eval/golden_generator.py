"""Generate golden Q&A items from the book pages with a local LLM.

Each item is grounded in specific pages, so retrieval can be scored exactly
(was the reference page retrieved?) and answers can be judged against a
reference answer. Quality gates, in order:
  1. JSON-schema constrained generation (always parses)
  2. evidence check: the quoted evidence must really appear on the page
  3. LLM critic: standalone, answerable from the pages, specific (1-5 each)
  4. near-duplicate filter on question embeddings

Item types:
  factual        one page, from the printed / lettered text
  visual         one page, answer lives in the drawings or hand-lettered notes
  multi_page     needs two pages of the same chapter
  false_premise  question assumes something the page contradicts
  out_of_scope   plausible animation question the book does not cover
"""

import re
from difflib import SequenceMatcher

from src.eval.ollama_json import generate_json

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question", "reference_answer", "evidence"],
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "standalone": {"type": "integer", "minimum": 1, "maximum": 5},
        "answerable": {"type": "integer", "minimum": 1, "maximum": 5},
        "specific": {"type": "integer", "minimum": 1, "maximum": 5},
        "premise_false": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["reason", "standalone", "answerable", "specific", "premise_false"],
}

# Questions that point at the picture instead of asking about the technique
DEICTIC_RE = re.compile(
    r"\b(?:the|this|that|these)\s+(?:\w+\s+)?(?:illustration|drawing|sketch|figure|image|"
    r"diagram|picture|page|speech bubble|panel|caption|photo)s?\b"
    r"|\b(?:shown|depicted|pictured|illustrated)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}

COMMON_RULES = """Rules for the question:
- It must make sense on its own to someone who has NOT seen the page: never say \
"this page", "the figure", "the drawing above", "according to the text".
- Ask the way a student animator would naturally ask.
- Do not mention page numbers.
- "evidence": 1-3 short phrases copied WORD FOR WORD from the page material below \
that prove the answer (copy exactly, including capitalisation).
- "reference_answer": a correct, complete answer in 1-4 sentences, using only the page."""

TYPE_INSTRUCTIONS = {
    "factual": "Write ONE question whose answer is stated in the page TEXT "
               "(a technique, a rule, a number, a definition, a name or a reason).",
    "visual": "Write ONE question whose answer comes from the DRAWINGS or the "
              "HAND-LETTERED NOTES on the drawings (poses, timing numbers, which part "
              "leads, what the labels say) - not from a printed paragraph. Use the "
              "drawing description and the short capitalised lines. Phrase it about "
              "the ANIMATION ACTION OR TECHNIQUE being shown (e.g. 'In a double bounce "
              "walk, on which frame is the passing position?'), never about the picture "
              "itself (not 'in the illustration', 'the sketch of the cat', 'the speech "
              "bubble').",
    "multi_page": "Write ONE question that can ONLY be answered by combining "
                  "information from BOTH pages (e.g. compare, or connect a principle "
                  "on one page with its example on the other). The evidence must "
                  "include at least one phrase from each page.",
    "false_premise": "Write ONE question that states a WRONG fact as if it were true, "
                     "where the page clearly states the opposite (e.g. swap a number, "
                     "reverse which part leads, attribute something to the wrong person: "
                     "'Why does Williams put the passing position UP in a double bounce?' "
                     "when the page says it is DOWN). A genuine doubt or yes/no question "
                     "is NOT enough - the wrong fact must be asserted. The "
                     "reference_answer must politely correct the false assumption and "
                     "give the right fact from the page. Evidence = the phrase showing "
                     "the right fact.",
}

CRITIC_PROMPT = """You are a STRICT reviewer of a question/answer pair for a test set \
about the book "The Animator's Survival Kit". Most pairs have at least one flaw; \
give 5 only when there is truly nothing to improve.

PAGE MATERIAL:
{material}

QUESTION: {question}
REFERENCE ANSWER: {answer}
QUESTION TYPE: {qtype}

First write a one-sentence reason naming the biggest flaw, then score 1-5:
- standalone: imagine a student who owns the book but is NOT looking at this page. \
Do they know exactly what is being asked? Mentions of a particular picture, sketch, \
character in a drawing, speech bubble or "the example" they cannot identify = 1-2.
- answerable: the reference answer is fully supported by the page material \
(5 = completely; 1 = not supported or wrong){false_premise_note}
- specific: one clear correct answer, not trivial (copying a label or caption \
verbatim is trivial = 2)
- premise_false: {premise_rule}"""


def page_material(page, label="PAGE"):
    """Text the generator sees for one page: lettered/printed text + drawing description."""
    parts = [f"=== {label} ({page['where']}) ===", "TEXT AND LETTERING:", page["text"]]
    if page.get("description"):
        parts += ["DRAWING DESCRIPTION:", page["description"]]
    return "\n".join(parts)


def generate_item(qtype, pages, temperature=0.7):
    material = "\n\n".join(
        page_material(p, f"PAGE {chr(65 + i)}") for i, p in enumerate(pages)
    )
    prompt = (
        "You write test questions for a question-answering system about the book "
        "\"The Animator's Survival Kit\" by Richard Williams.\n\n"
        f"{material}\n\nTASK: {TYPE_INSTRUCTIONS[qtype]}\n\n{COMMON_RULES}"
    )
    return generate_json(prompt, QUESTION_SCHEMA, temperature=temperature), material


def critique(item, material, qtype):
    note = ("\n  For a false-premise question: score whether the reference answer "
            "correctly REJECTS the false assumption using the page."
            if qtype == "false_premise" else "")
    premise_rule = (
        "the question ASSERTS a specific fact that the page clearly contradicts "
        "(5 = clear wrong fact stated as true; 1 = no wrong fact, just a doubt or "
        "an open question)"
        if qtype == "false_premise" else "not applicable - give 5"
    )
    prompt = CRITIC_PROMPT.format(
        material=material, question=item["question"], answer=item["reference_answer"],
        qtype=qtype, false_premise_note=note, premise_rule=premise_rule,
    )
    return generate_json(prompt, CRITIC_SCHEMA)


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def evidence_found(evidence, material, min_share=0.8):
    """True if the quote (or >= min_share of it, as one block) occurs in the material."""
    quote, haystack = _norm(evidence), _norm(material)
    if not quote:
        return False
    if quote in haystack:
        return True
    match = SequenceMatcher(None, quote, haystack, autojunk=False).find_longest_match(
        0, len(quote), 0, len(haystack)
    )
    return match.size >= min_share * len(quote)


def evidence_pages(evidence_list, pages):
    """Which of the pages each evidence phrase was found on."""
    found = set()
    for ev in evidence_list:
        for p in pages:
            if evidence_found(ev, page_material(p)):
                found.add(p["pdf_page"])
    return found


def generate_out_of_scope(n, existing=()):
    prompt = (
        "The book \"The Animator's Survival Kit\" by Richard Williams covers hand-drawn "
        "character animation: timing and spacing, keys and breakdowns, walks, runs, "
        "jumps, weight, flexibility, anticipation, takes, dialogue, acting, animal "
        "action and directing.\n\n"
        f"Write {n} realistic questions an animation student might ask that this book "
        "does NOT answer - e.g. specific 3D software (Blender, Maya rigging), render "
        "settings, game engines, studio salaries, file formats, AI tools, streaming "
        "delivery specs, anime production pipelines after 2000. Vary the topics. Each "
        "question must be standalone and specific."
        + (f"\nAvoid these existing questions: {list(existing)}" if existing else "")
    )
    return generate_json(prompt, OUT_OF_SCOPE_SCHEMA, temperature=0.9)["questions"]
