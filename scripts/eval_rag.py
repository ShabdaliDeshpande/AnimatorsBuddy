"""Evaluate the RAG pipeline on the golden dataset (DeepEval + exact retrieval metrics).

Stages (each resumable; results are cached per run under data/eval/runs/<run>/):
  answers    run the real pipeline on every question -> answers.jsonl
  judge      DeepEval metrics with the local judge     -> judgements.jsonl
  report     retrieval + judge summary per type        -> summary.md, summary.json

Usage:
    python scripts/eval_rag.py --run baseline                    # all stages, all items
    python scripts/eval_rag.py --run baseline --limit 20         # first 20 items
    python scripts/eval_rag.py --run baseline --stages answers   # answers only
    python scripts/eval_rag.py --run baseline --stages report    # re-print summary
    python scripts/eval_rag.py --run fast --image-pages 0 --metrics correctness,groundedness

Compare runs by giving each configuration its own --run name.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")   # keep everything local

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.retrieval_metrics import ranked_pages, retrieval_scores  # noqa: E402
from src.utils.feedback import load_feedback  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
RUNS_DIR = EVAL_DIR / "runs"
TYPES = ("factual", "visual", "multi_page", "false_premise", "out_of_scope", "user")


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_items(include_feedback=True):
    items = load_jsonl(GOLDEN_PATH)
    if include_feedback:  # questions saved from the UI with correct pages
        for r in load_feedback():
            if r.get("correct_pdf_pages"):
                items.append({
                    "id": f"fb_{r['id']}",
                    "type": "user",
                    "question": r["question"],
                    "reference_answer": None,
                    "reference_pdf_pages": r["correct_pdf_pages"],
                    "reference_printed_pages": r["correct_printed_pages"],
                })
    return items


# --------------------------------------------------------------------------- #
# Stage 1: answers
# --------------------------------------------------------------------------- #

def run_answers(items, run_dir, config):
    from src.generation.llm import chat
    from src.pipeline.query_pipeline import QueryPipeline

    path = run_dir / "answers.jsonl"
    done = {r["id"] for r in load_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[answers] {len(todo)} to do ({len(done)} cached)")
    if not todo:
        return
    pipeline = QueryPipeline()
    for i, item in enumerate(todo, 1):
        t = time.time()
        context = pipeline.prepare(item["question"], k=config["k"],
                                   image_pages=config["image_pages"])
        try:
            reply = chat(context["messages"])
        except Exception as e:
            print(f"  {item['id']}: generation failed ({e}); will retry next run")
            continue
        result = pipeline.finish(context, reply)
        append_jsonl(path, {
            "id": item["id"],
            "answer": result["answer"],
            "hits": [{"id": h["id"], "pdf_page": h["pdf_page"], "score": h["score"]}
                     for h in result["hits"]],
            "context_pdf_pages": [p["pdf_page"] for p in result["pages"]],
            "retrieval_context": [h["text"] for h in result["hits"]],
            "generation_context": [p["text"] for p in result["pages"]],
            "figures": [f["file"] for f in result["figures"]],
            "latency_s": round(time.time() - t, 1),
        })
        print(f"  [{i}/{len(todo)}] {item['id']} {item['type']:13} {time.time() - t:4.0f}s  "
              f"pages {ranked_pages(result['hits'])[:5]} ref {item['reference_pdf_pages']}")


# --------------------------------------------------------------------------- #
# Stage 2: judge
# --------------------------------------------------------------------------- #

def run_judge(items, run_dir, selected):
    from src.eval.judge import OllamaJudge
    from src.eval.metrics import build_metrics, metrics_for, test_case

    answers = {r["id"]: r for r in load_jsonl(run_dir / "answers.jsonl")}
    path = run_dir / "judgements.jsonl"
    done = {(r["id"], r["metric"]) for r in load_jsonl(path)}
    factories = build_metrics(OllamaJudge())

    jobs = [(it, m) for it in items if it["id"] in answers
            for m in metrics_for(it["type"], selected) if (it["id"], m) not in done]
    print(f"[judge] {len(jobs)} metric evaluations to do ({len(done)} cached)")
    for i, (item, name) in enumerate(jobs, 1):
        t = time.time()
        metric = factories[name]()
        try:
            metric.measure(test_case(item, answers[item["id"]], name))
            record = {"id": item["id"], "metric": name, "score": metric.score,
                      "reason": metric.reason}
        except Exception as e:
            print(f"  {item['id']} {name}: failed ({e}); will retry next run")
            continue
        append_jsonl(path, record)
        print(f"  [{i}/{len(jobs)}] {item['id']} {name:20} {record['score']:.2f}  "
              f"{time.time() - t:4.0f}s")


# --------------------------------------------------------------------------- #
# Stage 3: report
# --------------------------------------------------------------------------- #

def mean(values):
    return statistics.mean(values) if values else None


def fmt(v):
    return "–" if v is None else f"{v:.2f}"


def report(items, run_dir, config):
    answers = {r["id"]: r for r in load_jsonl(run_dir / "answers.jsonl")}
    judgements = defaultdict(dict)
    for r in load_jsonl(run_dir / "judgements.jsonl"):
        judgements[r["id"]][r["metric"]] = r

    rows = []
    for item in items:
        a = answers.get(item["id"])
        if not a:
            continue
        rows.append({
            "item": item,
            "answer": a,
            "retrieval": retrieval_scores(item["reference_pdf_pages"], a["hits"],
                                          a["context_pdf_pages"]),
            "judge": {m: j["score"] for m, j in judgements[item["id"]].items()},
        })

    retrieval_keys = ["hit@1", "hit@3", "hit@5", "mrr", "recall@5", "context_recall_pages"]
    judge_keys = ["correctness", "groundedness", "answer_relevancy", "contextual_precision",
                  "contextual_recall", "premise_correction", "abstention"]
    groups = [("ALL", rows)] + [
        (t, [r for r in rows if r["item"]["type"] == t]) for t in TYPES
    ]
    groups = [(name, g) for name, g in groups if g]

    def table(keys, source):
        lines = ["| type | n | " + " | ".join(keys) + " |",
                 "|---|---|" + "---|" * len(keys)]
        summary = {}
        for name, g in groups:
            vals = {k: mean([r[source][k] for r in g if k in r[source]]) for k in keys}
            summary[name] = {"n": len(g), **vals}
            lines.append(f"| {name} | {len(g)} | " + " | ".join(fmt(vals[k]) for k in keys) + " |")
        return "\n".join(lines), summary

    retrieval_table, retrieval_summary = table(retrieval_keys, "retrieval")
    judge_table, judge_summary = table(judge_keys, "judge")
    latency = mean([r["answer"]["latency_s"] for r in rows])

    worst = sorted(
        (r for r in rows if "correctness" in r["judge"]),
        key=lambda r: r["judge"]["correctness"],
    )[:10]
    worst_lines = []
    for r in worst:
        j = judgements[r["item"]["id"]]["correctness"]
        worst_lines.append(
            f"- **{r['item']['id']}** ({r['item']['type']}, ref p.{r['item']['reference_printed_pages']}, "
            f"retrieved PDF {ranked_pages(r['answer']['hits'])[:5]}) correctness "
            f"{j['score']:.2f}: {r['item']['question']}\n  - judge: {j['reason']}"
        )
    missed = [r for r in rows if r["retrieval"] and not r["retrieval"]["hit@5"]]

    md = "\n".join([
        f"# RAG evaluation: {run_dir.name}",
        "",
        f"Items evaluated: {len(rows)} · config: {json.dumps(config)} · "
        f"mean answer latency {fmt(latency)} s",
        "",
        "## Retrieval (exact, against reference pages)",
        retrieval_table,
        "",
        "## Generation (DeepEval, local judge; scores 0-1)",
        judge_table,
        "",
        f"## Retrieval misses (reference page not in top 5): {len(missed)}",
        *[f"- {r['item']['id']} ({r['item']['type']}): {r['item']['question']} "
          f"— ref PDF {r['item']['reference_pdf_pages']}, got {ranked_pages(r['answer']['hits'])[:5]}"
          for r in missed[:15]],
        "",
        "## Lowest correctness",
        *worst_lines,
    ])
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "run": run_dir.name, "config": config, "n": len(rows), "latency_s": latency,
        "retrieval": retrieval_summary, "generation": judge_summary,
    }, indent=2), encoding="utf-8")
    print(md)
    print(f"\nSaved: {run_dir / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="baseline", help="name of this configuration")
    parser.add_argument("--stages", default="answers,judge,report")
    parser.add_argument("--limit", type=int, help="only the first N items")
    parser.add_argument("--types", help="comma list, e.g. factual,visual")
    parser.add_argument("--metrics", help="comma list of judge metrics to run")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--image-pages", type=int, default=3)
    parser.add_argument("--no-feedback", action="store_true",
                        help="don't add UI feedback test cases")
    args = parser.parse_args()

    items = load_items(include_feedback=not args.no_feedback)
    if args.types:
        wanted = set(args.types.split(","))
        items = [it for it in items if it["type"] in wanted]
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"No items; build the dataset first: python scripts/build_golden.py")
        return

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = {"k": args.k, "image_pages": args.image_pages}
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            print(f"Run '{args.run}' was made with {saved}; using that (pick a new --run "
                  f"name to change settings).")
            config = saved
    config_path.write_text(json.dumps(config))

    stages = args.stages.split(",")
    selected = set(args.metrics.split(",")) if args.metrics else None
    if "answers" in stages:
        run_answers(items, run_dir, config)
    if "judge" in stages:
        run_judge(items, run_dir, selected)
    if "report" in stages:
        report(items, run_dir, config)


if __name__ == "__main__":
    main()
