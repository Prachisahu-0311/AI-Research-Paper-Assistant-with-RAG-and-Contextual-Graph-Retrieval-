"""Interactive CLI for the full plain-RAG pipeline."""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import answer_question
from src.vector_store import VectorStore

_BAR = "━" * 42


def print_result(result: dict) -> None:
    print(f"\n{_BAR}")
    print(f"Question: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {chunk['paper_title']} (chunk {chunk['chunk_index']}) — similarity: {chunk['similarity']:.3f}")
    print(
        f"\nTiming: retrieval={result['retrieval_time_ms']}ms, "
        f"generation={result['generation_time_ms']}ms, "
        f"total={result['total_time_ms']}ms"
    )
    print(f"Tokens: {result['tokens_used']}")
    print(_BAR)


def main() -> None:
    vs = VectorStore()
    print(f"RAG pipeline ready — {vs.count()} chunks indexed.")
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
            result = answer_question(question, vector_store=vs)
            print_result(result)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
