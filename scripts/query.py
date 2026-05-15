"""Interactive query CLI for semantic search over paper chunks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import embed
from src.vector_store import VectorStore


def main() -> None:
    vs = VectorStore()
    print(f"Vector store loaded: {vs.count()} chunks across 20 papers.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query (or 'quit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() == "quit":
            break
        if not query:
            continue

        results = vs.query(embed(query), top_k=5)
        print()
        for rank, r in enumerate(results, 1):
            print(f"[{rank}] Paper: {r['paper_title']} ({r['paper_id']}, chunk {r['chunk_index']})")
            print(f"Similarity: {r['similarity']:.3f}")
            print(f"Text: {r['text'][:300]}...")
            print("---")
        print()


if __name__ == "__main__":
    main()
