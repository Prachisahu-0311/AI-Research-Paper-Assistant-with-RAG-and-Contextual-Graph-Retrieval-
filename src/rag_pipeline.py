"""End-to-end plain RAG pipeline: question → retrieve → generate → answer with citations."""
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.embeddings import embed, hyde_embed
from src.vector_store import VectorStore
from src.llm import generate_answer


def answer_question(
    question: str,
    top_k: int = 5,
    vector_store: VectorStore | None = None,
    use_hyde: bool = False,
) -> dict:
    """Retrieve relevant chunks and generate a cited answer."""
    vs = vector_store if vector_store is not None else VectorStore()

    t0 = time.perf_counter()
    # Only use HyDE for non-factual questions — HyDE hurts factual retrieval
    # by generating hypotheses that match wrong papers.
    if use_hyde:
        from src.query_classifier import classify_query
        effective_hyde = classify_query(question).get("type", "factual") != "factual"
    else:
        effective_hyde = False
    query_embedding = hyde_embed(question) if effective_hyde else embed(question)
    retrieved = vs.query(query_embedding, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not retrieved or max(c["similarity"] for c in retrieved) < 0.35:
        return {
            "question": question,
            "answer": "The provided sources do not contain enough information to answer this question.",
            "retrieved_chunks": retrieved,
            "grounded": True,
            "unsupported_claims": [],
            "model": None,
            "tokens_used": 0,
            "retrieval_time_ms": round(retrieval_ms, 1),
            "generation_time_ms": 0.0,
            "total_time_ms": round(retrieval_ms, 1),
        }

    t1 = time.perf_counter()
    llm_result = generate_answer(question, retrieved)
    generation_ms = (time.perf_counter() - t1) * 1000

    import re
    et_al_citations = re.findall(r'[A-Z][a-z]+\s+et al\.', llm_result["answer"])
    if et_al_citations:
        llm_result["grounded"] = False
        llm_result["unsupported_claims"] = llm_result.get("unsupported_claims", []) + [
            f"Answer cites external authors not in corpus: {', '.join(set(et_al_citations))}"
        ]

    return {
        "question": question,
        "answer": llm_result["answer"],
        "retrieved_chunks": retrieved,
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
    load_dotenv()

    result = answer_question("What is multi-head attention?")

    print(f"Question: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources retrieved:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {chunk['paper_title']} (chunk {chunk['chunk_index']}) — sim: {chunk['similarity']:.3f}")
    print(f"\nTiming: retrieval={result['retrieval_time_ms']}ms, "
          f"generation={result['generation_time_ms']}ms, "
          f"total={result['total_time_ms']}ms")
    print(f"Tokens: {result['tokens_used']}")
