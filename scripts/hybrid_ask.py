"""Interactive CLI for the hybrid RAG pipeline."""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_pipeline import answer_question_hybrid
from src.vector_store import VectorStore

_BAR = "━" * 44


def print_result(result: dict) -> None:
    mode = result["retrieval_mode"]
    graph_label = f" | graph matched: {result['graph_papers_used']}" if result["graph_papers_used"] else ""
    print(f"\n{_BAR}")
    print(f"Question: {result['question']}\n")
    print(f"Mode: {mode}{graph_label}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {chunk['paper_title']} (chunk {chunk['chunk_index']}) — sim: {chunk['similarity']:.3f}")
    print(
        f"\nTiming: retrieval={result['retrieval_time_ms']}ms  "
        f"generation={result['generation_time_ms']}ms  "
        f"total={result['total_time_ms']}ms"
    )
    print(f"Grounded: {result['grounded']}  |  Tokens: {result['tokens_used']}")
    print(_BAR)


def main() -> None:
    vs = VectorStore()
    print(f"Hybrid RAG ready — {vs.count()} chunks + knowledge graph loaded.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question (or 'quit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if question.lower() == "quit":
            break
        if not question:
            continue
        try:
            result = answer_question_hybrid(question, vector_store=vs)
            print_result(result)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
