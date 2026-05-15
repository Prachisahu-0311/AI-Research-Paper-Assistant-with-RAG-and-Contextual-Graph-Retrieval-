"""
Query decomposition for multi-hop questions.
Breaks complex questions into sub-queries, retrieves each, synthesizes.
Only used when query type is classified as multi_hop.
"""
import json
import os

import httpx

from src.embeddings import embed, hyde_embed
from src.vector_store import VectorStore

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DECOMPOSE_PROMPT = """You are helping a RAG system answer complex questions about AI research papers.

Break this multi-hop question into 2-4 simpler sub-questions. Each sub-question must:
- Ask about ONE specific paper or ONE specific concept
- Be answerable by retrieving from a single paper
- Be a complete, self-contained question

IMPORTANT: You must always return MORE THAN ONE sub-question. Never return the original question unchanged.

Examples of good decomposition:

Question: "How did efficient attention methods evolve from 2019 to 2021?"
Output: ["What problem does the Reformer solve and when was it published?", "What problem does the Linformer solve and when was it published?", "What problem do Performers solve and when were they published?", "What core limitation do Reformer, Linformer, and Performers all address?"]

Question: "What did RoBERTa and ALBERT each discover about BERT training?"
Output: ["What specific changes did RoBERTa make to BERT pretraining?", "What specific changes did ALBERT make to reduce BERT parameters?", "How do RoBERTa and ALBERT differ in their approach to improving BERT?"]

Question: "Which papers modified BERT pretraining objective and what did each change?"
Output: ["What pretraining objective change did RoBERTa make compared to BERT?", "What pretraining objective change did XLNet make compared to BERT?", "What pretraining objective change did ELECTRA make compared to BERT?"]

Now decompose this question. Return ONLY a JSON array of strings. No explanation. No markdown. Minimum 2 sub-questions, maximum 4.

Question: {question}

JSON array:"""


COMPARISON_KEYWORDS = [
    " vs ", " versus ", "differ", "compare", "contrast",
    "how do", "how does", "difference between",
    "what assumption", "each rely on", "each reduce",
    "each use", "each approach",
]


def is_comparison_question(question: str) -> bool:
    """
    Returns True if the question is a comparison between two or more things.
    Comparison questions should NOT be decomposed — they require synthesis,
    not independent sub-retrieval.
    """
    q = question.lower()
    return any(k in q for k in COMPARISON_KEYWORDS)


def decompose_question(question: str) -> list[str]:
    """Break a multi-hop question into sub-questions via LLM."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return [question]

    try:
        with httpx.Client(timeout=15.0, verify=False) as client:
            r = client.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
            )
        if r.status_code == 200:
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            sub_questions = json.loads(raw)
            if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                result = sub_questions[:4]
                # Guard: if LLM returned a single item very similar to original, apply fallback
                def _norm(s: str) -> str:
                    return " ".join(s.lower().split()).rstrip("?.")
                if len(result) == 1 and _norm(result[0]) == _norm(question):
                    print("  Decomposition returned original question — applying fallback split")
                    return [
                        f"What is the core technical approach of each paper mentioned in: {question}",
                        f"What specific problem does each paper address in: {question}",
                        f"How do the approaches compare to each other in: {question}",
                    ]
                return result
    except Exception as e:
        print(f"Decomposition failed, using original question: {e}")
    return [question]


def retrieve_for_subquestion(
    sub_question: str,
    vector_store: VectorStore,
    use_hyde: bool = False,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve chunks for a single sub-question."""
    embedding = hyde_embed(sub_question) if use_hyde else embed(sub_question)
    return vector_store.query(embedding, top_k=top_k)


def decomposed_retrieve(
    question: str,
    vector_store: VectorStore | None = None,
    use_hyde: bool = False,
) -> dict:
    """
    Full decomposed retrieval pipeline for multi-hop questions.
    Returns dict with deduplicated chunks and decomposition metadata.
    """
    if vector_store is None:
        vector_store = VectorStore()

    sub_questions = decompose_question(question)
    print(f"  Decomposed into {len(sub_questions)} sub-questions")

    all_chunks: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for sq in sub_questions:
        chunks = retrieve_for_subquestion(sq, vector_store, use_hyde=use_hyde, top_k=3)
        for chunk in chunks:
            cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
            if cid not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(cid)

    # Cap at 6 chunks to stay within context limits
    all_chunks = all_chunks[:6]

    return {
        "retrieved_chunks": all_chunks,
        "sub_questions": sub_questions,
        "decomposed": True,
    }
