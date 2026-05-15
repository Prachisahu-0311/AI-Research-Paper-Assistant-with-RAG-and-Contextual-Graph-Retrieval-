"""Run the final hybrid RAG eval with all improvements: HyDE + decomposition + comparison guard."""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import load_questions, judge_answer, compute_retrieval_recall
from src.hybrid_pipeline import answer_question_hybrid
from src.vector_store import VectorStore


def run_final_eval(questions: list[dict], vs: VectorStore, judge_model: str) -> list[dict]:
    api_key = os.environ["GROQ_API_KEY"]
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i:2}/{len(questions)}] {q['id']} — {q['question'][:55]}...")

        try:
            result = answer_question_hybrid(q["question"], vector_store=vs, use_hyde=True)
            recall = compute_retrieval_recall(result["retrieved_chunks"], q.get("expected_papers", []))
            scores = judge_answer(q["question"], q["reference_answer"], result["answer"], api_key, judge_model)
        except Exception as e:
            print(f"          ERROR: {e}")
            results.append({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "answer": "", "retrieval_mode": "error", "graph_papers_used": [],
                "sub_questions": [],
                "retrieval_recall": None, "correctness": -1, "grounding": -1,
                "citation_quality": -1, "total_score": -1,
                "judge_reasoning": f"exception: {e}", "latency_ms": 0,
                "rag_tokens": 0, "judge_tokens": 0,
                "retrieved_papers": [], "expected_papers": q.get("expected_papers", []),
            })
            time.sleep(65)
            continue

        recall_str = f"{recall:.0%}" if recall is not None else "N/A"
        mode = result["retrieval_mode"]
        n_sub = len(result.get("sub_questions", []))
        sub_note = f" ({n_sub} sub-qs)" if n_sub else ""
        print(f"          mode={mode:<11}  recall={recall_str}  score={scores.get('total', -1)}/7{sub_note}")

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": result["answer"],
            "retrieval_mode": result["retrieval_mode"],
            "graph_papers_used": result["graph_papers_used"],
            "sub_questions": result.get("sub_questions", []),
            "retrieval_recall": recall,
            "correctness": scores.get("correctness", -1),
            "grounding": scores.get("grounding", -1),
            "citation_quality": scores.get("citation_quality", -1),
            "total_score": scores.get("total", -1),
            "judge_reasoning": scores.get("reasoning", ""),
            "latency_ms": result["total_time_ms"],
            "rag_tokens": result["tokens_used"],
            "judge_tokens": scores.get("judge_tokens", 0),
            "retrieved_papers": [c["paper_id"] for c in result["retrieved_chunks"]],
            "expected_papers": q.get("expected_papers", []),
        })

        time.sleep(40)

    return results


def main() -> None:
    questions = load_questions()
    out_path = Path("data/eval/results_hybrid_final.json")

    existing: dict[str, dict] = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8")):
            existing[r["id"]] = r
        failed = [q for q in questions if existing.get(q["id"], {}).get("total_score", -1) < 0]
        questions_to_run = failed if failed else []
        if not questions_to_run:
            print("All final eval questions already scored.")
            return
        print(f"Resuming: {len(questions_to_run)} failed questions")
    else:
        questions_to_run = questions

    vs = VectorStore()
    judge_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    print(f"Vector store: {vs.count()} chunks | Model: {judge_model}\n")

    confirm = input(f"Run final eval on {len(questions_to_run)} questions? (y/n): ").strip().lower()
    if confirm != "y":
        return

    new_results = run_final_eval(questions_to_run, vs, judge_model)
    for r in new_results:
        existing[r["id"]] = r
    all_results = [existing[q["id"]] for q in load_questions() if q["id"] in existing]

    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
