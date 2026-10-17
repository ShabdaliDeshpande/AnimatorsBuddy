"""DeepEval metrics used for the RAG evaluation, judged by a local Ollama model.

Which metrics run depends on the golden item type:
  factual / visual / multi_page : correctness, groundedness, answer_relevancy,
                                  contextual_precision, contextual_recall
  false_premise                 : the above + premise_correction
  out_of_scope                  : abstention
  user (feedback, no reference) : groundedness, answer_relevancy

Note: DeepEval's FaithfulnessMetric only penalises claims that CONTRADICT the
context, not claims the context doesn't support, so a stricter G-Eval
"groundedness" is used for hallucination instead.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

P = SingleTurnParams

ANSWERABLE = ("factual", "visual", "multi_page", "false_premise")


def build_metrics(judge):
    """name -> factory(), so every test case gets fresh metric objects."""
    common = {"model": judge, "async_mode": False}
    return {
        "correctness": lambda: GEval(
            name="Correctness",
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            evaluation_steps=[
                "Identify the key facts in the expected output that answer the input.",
                "Check whether the actual output states each key fact correctly "
                "(numbers, which body part leads, order of positions, names).",
                "Penalise heavily any statement in the actual output that contradicts "
                "the expected output.",
                "If the input contains a false assumption, rejecting that assumption "
                "is CORRECT and agrees with the expected output - do not penalise it.",
                "Do not penalise extra correct detail or different wording.",
            ],
            **common,
        ),
        "groundedness": lambda: GEval(
            name="Groundedness",
            evaluation_params=[P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT],
            evaluation_steps=[
                "List the factual claims made in the actual output.",
                "For each claim, check whether the retrieval context supports it.",
                "Claims that are not supported by the retrieval context (even if they "
                "might be true in general) count as hallucinations.",
                "Give the highest score when every claim is supported; lower it in "
                "proportion to unsupported or contradicted claims.",
            ],
            **common,
        ),
        "answer_relevancy": lambda: AnswerRelevancyMetric(**common),
        "contextual_precision": lambda: ContextualPrecisionMetric(**common),
        "contextual_recall": lambda: ContextualRecallMetric(**common),
        "premise_correction": lambda: GEval(
            name="PremiseCorrection",
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            evaluation_steps=[
                "The input contains a false assumption; the expected output shows the "
                "correction.",
                "Check whether the actual output explicitly points out that the "
                "assumption is wrong, rather than going along with it.",
                "Check whether it gives the correct fact from the expected output.",
                "Give the lowest score if it accepts the false assumption, the highest "
                "if it clearly rejects it and states the correct fact.",
            ],
            **common,
        ),
        "abstention": lambda: GEval(
            name="Abstention",
            evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
            evaluation_steps=[
                "The input asks about something the book (The Animator's Survival Kit) "
                "does not cover.",
                "Give the highest score if the actual output clearly says the book does "
                "not cover it (it may add a brief pointer to related book content).",
                "Give the lowest score if it answers the question as if the book "
                "covered it, or invents specific details.",
            ],
            **common,
        ),
    }


def metrics_for(item_type, selected=None):
    if item_type == "out_of_scope":
        names = ["abstention"]
    elif item_type == "user":
        names = ["groundedness", "answer_relevancy"]
    else:
        names = ["correctness", "groundedness", "answer_relevancy",
                 "contextual_precision", "contextual_recall"]
        if item_type == "false_premise":
            names.append("premise_correction")
    return [n for n in names if not selected or n in selected]


def test_case(item, answer_record, metric_name):
    """The DeepEval test case for one metric.

    Contextual precision/recall judge the ranked retrieval results; groundedness
    judges against everything the generator actually saw (incl. neighbour pages).
    """
    context = (
        answer_record["retrieval_context"]
        if metric_name in ("contextual_precision", "contextual_recall")
        else answer_record["generation_context"]
    )
    return LLMTestCase(
        input=item["question"],
        actual_output=answer_record["answer"] or "(no answer)",
        expected_output=item.get("reference_answer"),
        retrieval_context=context,
    )
