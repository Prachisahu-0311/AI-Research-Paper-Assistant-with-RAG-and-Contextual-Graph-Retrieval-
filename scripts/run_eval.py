"""Run the plain RAG eval harness and save results to data/eval/."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import load_questions, print_metrics_table, run_eval
from src.vector_store import VectorStore


def main() -> None:
    questions = load_questions()
    out_path = Path("data/eval/results_plain_rag.json")

    # Resume mode: load existing results and only re-run failed questions
    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            for r in json.loads(out_path.read_text(encoding="utf-8")):
                existing[r["id"]] = r
            failed = [q for q in questions if existing.get(q["id"], {}).get("total_score", -1) < 0]
            if failed:
                print(f"Resuming: {len(failed)} failed questions to re-run (of {len(questions)} total)")
                questions_to_run = failed
            else:
                print("All questions already scored. Nothing to re-run.")
                print_metrics_table(list(existing.values()))
                return
        except Exception:
            questions_to_run = questions
    else:
        questions_to_run = questions
        print(f"Loaded {len(questions)} eval questions")

    vs = VectorStore()
    print(f"Vector store: {vs.count()} chunks indexed")
    judge_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    print(f"Model: {judge_model}\n")

    confirm = input(f"Run eval on {len(questions_to_run)} questions? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    new_results = run_eval(questions_to_run, vs, judge_model=judge_model)

    # Merge with existing results, preserving order from questions.json
    for r in new_results:
        existing[r["id"]] = r
    all_results = [existing[q["id"]] for q in load_questions() if q["id"] in existing]

    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    print_metrics_table(all_results)

    total_rag = sum(r.get("rag_tokens", 0) for r in all_results)
    total_judge = sum(r.get("judge_tokens", 0) for r in all_results)
    print(f"\nToken usage: {total_rag:,} (RAG) + {total_judge:,} (judge) = {total_rag + total_judge:,} total")


if __name__ == "__main__":
    main()
