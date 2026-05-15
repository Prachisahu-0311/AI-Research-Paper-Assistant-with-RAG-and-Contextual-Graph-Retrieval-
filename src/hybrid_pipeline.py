"""End-to-end hybrid RAG pipeline: question → graph+vector retrieve → generate."""
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time

from src.embeddings import embed, hyde_embed
from src.graph_store import build_graph, load_triples
from src.query_classifier import classify_query
from src.retrieval import hybrid_retrieve
from src.vector_store import VectorStore
from src.llm import generate_answer

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(load_triples())
    return _graph


def answer_question_hybrid(
    question: str,
    top_k: int = 5,
    vector_store: VectorStore | None = None,
    use_hyde: bool = False,
) -> dict:
    """Hybrid pipeline: graph traversal + scoped vector search + LLM generation.

    Multi-hop questions use query decomposition instead of single-shot retrieval.
    """
    vs = vector_store if vector_store is not None else VectorStore()
    G = _get_graph()

    t0 = time.perf_counter()

    # Classify first so multi-hop can branch to decomposition
    clf = classify_query(question)
    query_type = clf.get("type", "factual")

    # HyDE helps connection/multi-hop questions but hurts factual ones where
    # the correct paper is specific — the hypothesis drifts to wrong chunks.
    effective_hyde = use_hyde and (query_type != "factual")

    from src.decomposition import decomposed_retrieve, is_comparison_question

    if query_type == "multi_hop" and not is_comparison_question(question):
        decomp = decomposed_retrieve(question, vector_store=vs, use_hyde=effective_hyde)
        retrieved = decomp["retrieved_chunks"]
        sub_questions = decomp["sub_questions"]
        graph_papers: list[str] = []
        retrieval_mode = "decomposed"
    else:
        query_embedding = hyde_embed(question) if effective_hyde else embed(question)
        retrieved, graph_papers = hybrid_retrieve(question, query_embedding, G, vs, top_k=top_k)
        sub_questions = []
        retrieval_mode = "hybrid" if graph_papers else "vector_only"

    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not retrieved or max((c["similarity"] for c in retrieved), default=0) < 0.35:
        return {
            "question": question,
            "answer": "The provided sources do not contain enough information to answer this question.",
            "retrieved_chunks": retrieved,
            "graph_papers_used": graph_papers,
            "retrieval_mode": "below_threshold",
            "sub_questions": sub_questions,
            "grounded": False,
            "unsupported_claims": ["No relevant sources retrieved — similarity below threshold"],
            "model": None,
            "tokens_used": 0,
            "retrieval_time_ms": round(retrieval_ms, 1),
            "generation_time_ms": 0.0,
            "total_time_ms": round(retrieval_ms, 1),
        }

    t1 = time.perf_counter()
    llm_result = generate_answer(question, retrieved)
    generation_ms = (time.perf_counter() - t1) * 1000

    return {
        "question": question,
        "answer": llm_result["answer"],
        "retrieved_chunks": retrieved,
        "graph_papers_used": graph_papers,
        "retrieval_mode": retrieval_mode,
        "sub_questions": sub_questions,
        "model": llm_result["model"],
        "tokens_used": llm_result["tokens_used"],
        "grounded": llm_result["grounded"],
        "unsupported_claims": llm_result["unsupported_claims"],
        "retrieval_time_ms": round(retrieval_ms, 1),
        "generation_time_ms": round(generation_ms, 1),
        "total_time_ms": round(retrieval_ms + generation_ms, 1),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    q = "Which papers address quadratic attention complexity?"
    result = answer_question_hybrid(q)
    print(f"Question: {result['question']}")
    print(f"Mode: {result['retrieval_mode']}")
    print(f"Graph papers: {result['graph_papers_used']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\nChunks retrieved:")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {c['paper_title']} (chunk {c['chunk_index']}) sim={c['similarity']:.3f}")
