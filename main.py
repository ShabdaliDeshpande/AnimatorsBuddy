"""Single entry point for the RAG project.

Usage:
    python main.py ingest                 # full ingestion pipeline, steps 1-5 in order
    python main.py ingest --from chunk    # resume the pipeline from a step
    python main.py ask "how do I animate a sneak?"
    python main.py ui                     # Streamlit app
    python main.py eval --run baseline    # evaluate on the golden dataset

Any step can also be run on its own (extra arguments are passed through):
    python main.py ocr --pages 12,40      # 1. OCR (Tesseract + Gemma)
    python main.py describe               # 2. drawing descriptions
    python main.py crop                   # 3. cut out figures
    python main.py chunk                  # 4. build chunks
    python main.py index                  # 5. embed + index in Chroma
    python main.py query "Ken Harris"     # retrieval only, no answer
    python main.py golden                 # build the golden dataset
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# command -> script; the ingestion pipeline runs these in this order
PIPELINE = [
    ("ocr", "ocr_pdfs.py"),
    ("describe", "describe_pages.py"),
    ("crop", "crop_figures.py"),
    ("chunk", "chunk_pages.py"),
    ("index", "ingest.py"),
]
COMMANDS = {
    **dict(PIPELINE),
    "query": "query.py",
    "ask": "ask.py",
    "golden": "build_golden.py",
    "eval": "eval_rag.py",
}


def run_script(script, args):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def ingest(args):
    """Run the ingestion steps in order, optionally starting from one (--from chunk)."""
    names = [name for name, _ in PIPELINE]
    start = 0
    if "--from" in args:
        step = args[args.index("--from") + 1]
        if step not in names:
            sys.exit(f"Unknown step '{step}'. Steps: {', '.join(names)}")
        start = names.index(step)
    for name, script in PIPELINE[start:]:
        code = run_script(script, [])
        if code:
            sys.exit(f"Step '{name}' failed (exit {code}); fix it and resume with "
                     f"python main.py ingest --from {name}")
    print("\nIngestion complete. Try: python main.py ask \"what is a breakdown?\"")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    command, args = sys.argv[1], sys.argv[2:]
    if command == "ingest":
        ingest(args)
    elif command == "ui":
        sys.exit(subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "ui.py"), *args],
            cwd=ROOT,
        ).returncode)
    elif command in COMMANDS:
        sys.exit(run_script(COMMANDS[command], args))
    else:
        sys.exit(f"Unknown command '{command}'. Run: python main.py --help")


if __name__ == "__main__":
    main()
