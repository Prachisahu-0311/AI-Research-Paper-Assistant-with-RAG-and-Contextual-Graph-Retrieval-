"""Run the hybrid RAG eval and compare against plain RAG baseline."""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import load_questions, judge_answer, compute_retrieval_recall, print_metrics_table
from src.hybrid_pipeline import answer_question_hybrid
from src.vector_store import VectorStore


def run_hybrid_eval(questions: list[dict], vs: VectorStore, judge_model: str) -> list[dict]:
    api_key = os.environ["GROQ_API_KEY"]
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i:2}/{len(questions)}] {q['id']} — {q['question'][:55]}...")

        try:
            result = answer_question_hybrid(q["question"], vector_store=vs)
            recall = compute_retrieval_recall(result["retrieved_chunks"], q.get("expected_papers", []))
            scores = judge_answer(q["question"], q["reference_answer"], result["answer"], api_key, judge_model)
        except Exception as e:
            print(f"          ERROR: {e}")
            results.append({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "answer": "", "retrieval_mode": "error", "graph_papers_used": [],
                "retrieval_recall": None, "correctness": -1, "grounding": -1,
                "citation_quality": -1, "total_score": -1,
                "judge_reasoning": f"exception: {e}", "latency_ms": 0,
                "rag_tokens": 0, "judge_tokens": 0,
                "retrieved_papers": [], "expected_papers": q.get("expected_papers", []),
            })
            time.sleep(65)
            continue

        recall_str = f"{recall:.0%}" if recall is not None else "N/A"
        print(
            f"          mode={result['retrieval_mode']:<11}  "
            f"recall={recall_str}  score={scores.get('total', -1)}/7"
        )

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": result["answer"],
            "retrieval_mode": result["retrieval_mode"],
            "graph_papers_used": result["graph_papers_used"],
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


def compare_tables(plain: list[dict], hybrid: list[dict]) -> None:
    from collections import defaultdict
    import statistics

    def avg_by_type(results):
        by_type = defaultdict(list)
        for r in results:
            if r["total_score"] >= 0:
                by_type[r["type"]].append(r)
        return by_type

    p, h = avg_by_type(plain), avg_by_type(hybrid)
    print("\n" + "=" * 70)
    print("COMPARISON: PLAIN RAG vs HYBRID RAG")
    print("=" * 70)
    print(f"{'Type':<14}  {'Plain Recall':>12}  {'Plain Score':>11}  {'Hybrid Recall':>13}  {'Hybrid Score':>12}")
    print("-" * 70)
    for qtype in ["factual", "connection", "multi_hop", "out_of_scope"]:
        pr = p.get(qtype, [])
        hr = h.get(qtype, [])
        if not pr and not hr:
            continue
        p_recalls = [r["retrieval_recall"] for r in pr if r["retrieval_recall"] is not None]
        h_recalls = [r["retrieval_recall"] for r in hr if r["retrieval_recall"] is not None]
        p_recall = f"{statistics.mean(p_recalls):.0%}" if p_recalls else "N/A"
        h_recall = f"{statistics.mean(h_recalls):.0%}" if h_recalls else "N/A"
        p_score = f"{statistics.mean(r['total_score'] for r in pr):.2f}" if pr else "—"
        h_score = f"{statistics.mean(r['total_score'] for r in hr):.2f}" if hr else "—"
        print(f"{qtype:<14}  {p_recall:>12}  {p_score:>11}  {h_recall:>13}  {h_score:>12}")
    print("=" * 70)

    hybrid_modes = defaultdict(int)
    for r in hybrid:
        hybrid_modes[r.get("retrieval_mode", "unknown")] += 1
    print(f"\nRetrieval mode breakdown: {dict(hybrid_modes)}")


def main() -> None:
    questions = load_questions()
    out_path = Path("data/eval/results_hybrid_v3.json")

    existing: dict[str, dict] = {}
    if out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8")):
            existing[r["id"]] = r
        failed = [q for q in questions if existing.get(q["id"], {}).get("total_score", -1) < 0]
        questions_to_run = failed if failed else []
        if not questions_to_run:
            print("All hybrid questions already scored.")
            hybrid_results = list(existing.values())
        else:
            print(f"Resuming: {len(questions_to_run)} failed questions")
    else:
        questions_to_run = questions

    vs = VectorStore()
    judge_model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
    print(f"Vector store: {vs.count()} chunks | Model: {judge_model}\n")

    confirm = input(f"Run hybrid eval on {len(questions_to_run)} questions? (y/n): ").strip().lower()
    if confirm != "y":
        return

    new_results = run_hybrid_eval(questions_to_run, vs, judge_model)
    for r in new_results:
        existing[r["id"]] = r
    hybrid_results = [existing[q["id"]] for q in load_questions() if q["id"] in existing]

    out_path.write_text(json.dumps(hybrid_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    print("\n--- Hybrid Eval Results ---")
    print_metrics_table(hybrid_results)

    plain_path = Path("data/eval/results_plain_rag.json")
    if plain_path.exists():
        plain_results = json.loads(plain_path.read_text(encoding="utf-8"))
        compare_tables(plain_results, hybrid_results)


if __name__ == "__main__":
    main()
