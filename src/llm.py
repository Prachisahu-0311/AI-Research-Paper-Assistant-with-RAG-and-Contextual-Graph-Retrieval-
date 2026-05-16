"""LLM wrapper for answer generation using Groq REST API via httpx."""
import json
import os
import re

import httpx

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _parse_retry_seconds(text: str) -> float:
    """Extract retry-after seconds from Groq 429 body. Returns 20.0 if not found."""
    m = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', text)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    return 20.0

_SYSTEM_PROMPT = """You are a source verification assistant. Your only job is to extract and synthesize information that is explicitly present in the numbered source chunks provided below.

Rules:
1. Every factual claim in your answer MUST be traceable to a specific source chunk. Cite it as [Source N].
2. If the sources do not contain the information needed to answer, your correct response is: "The provided sources do not contain enough information to answer this question." This is a successful answer, not a failure.
3. Never use your training knowledge. Never speculate. Never infer beyond what the sources state.
4. If sources partially answer the question, answer only the part they support and say the rest is not in the sources.
5. When comparing two or more papers, address each paper separately using its labeled section before drawing comparisons."""


def _call_groq(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> dict:
    """Make a Groq REST call. On 429, sleeps the exact retry-after time (capped at 120s)."""
    import time
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(3):
        with httpx.Client(timeout=30.0, verify=False) as client:
            r = client.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 and attempt < 2:
            wait = min(_parse_retry_seconds(r.text) + 2, 120)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq API: max retries exceeded")


def _build_context(context_chunks: list[dict]) -> str:
    """Group chunks by paper and build structured context."""
    source_counter = 1
    parts = []
    current_paper = None
    paper_group: list[dict] = []
    grouped: list[tuple[str, list[dict]]] = []

    for chunk in context_chunks:
        if chunk["paper_title"] != current_paper:
            if paper_group:
                grouped.append((current_paper, paper_group))
            current_paper = chunk["paper_title"]
            paper_group = []
        paper_group.append(chunk)
    if paper_group:
        grouped.append((current_paper, paper_group))

    for paper_title, chunks in grouped:
        paper_id = chunks[0].get("paper_id", "unknown")
        parts.append(f"=== Paper: {paper_title} ({paper_id}) ===")
        for chunk in chunks:
            parts.append(
                f"[Source {source_counter}] Chunk {chunk['chunk_index']}\n"
                f"Text: {chunk['text']}"
            )
            source_counter += 1

    return "\n\n".join(parts)


def _check_grounding(
    answer: str,
    context_chunks: list[dict],
    model: str,
    api_key: str,
) -> dict:
    """Check if every claim in the answer is grounded in the source chunks."""
    similarities = [c["similarity"] for c in context_chunks if "similarity" in c]
    if similarities and all(s < 0.4 for s in similarities):
        return {
            "grounded": False,
            "unsupported_claims": ["No relevant sources were retrieved for this query"],
        }

    sources_summary = "\n\n".join(
        f"[Source {i}] {c['paper_title']}: {c['text'][:600]}{'...' if len(c['text']) > 600 else ''}"
        for i, c in enumerate(context_chunks, 1)
    )
    prompt = (
        f"Sources provided:\n{sources_summary}\n\n"
        f"Answer to verify:\n{answer}\n\n"
        "Task: Does every factual claim in the answer above appear explicitly in at least one "
        "of the source chunks provided? Reply with YES or NO on the first line. "
        "If NO, list the unsupported claims one per line, starting with a dash."
    )
    try:
        resp = _call_groq(
            messages=[
                {"role": "system", "content": "You are a fact-checker. Verify if claims are grounded in provided sources. Be strict."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=200,
            api_key=api_key,
        )
        lines = resp["choices"][0]["message"]["content"].strip().split("\n")
        grounded = "YES" in lines[0].upper()
        unsupported = [
            l.lstrip("-•").strip()
            for l in lines[1:]
            if l.strip() and (l.strip().startswith("-") or l.strip().startswith("•"))
        ] if not grounded else []
        return {"grounded": grounded, "unsupported_claims": unsupported}
    except Exception as e:
        return {"grounded": False, "unsupported_claims": [f"Grounding check failed: {e}"]}


def generate_answer(
    question: str,
    context_chunks: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> dict:
    """Generate a cited answer from retrieved chunks using Groq."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "<paste-your-key-here>":
        raise ValueError("GROQ_API_KEY not set in .env")
    if model is None:
        model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

    context_str = _build_context(context_chunks)
    resp = _call_groq(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context_str}\n\nAnswer:"},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )

    answer_text = resp["choices"][0]["message"]["content"]
    tokens_used = resp.get("usage", {}).get("total_tokens", 0)

    # Refusal answers are grounded by definition — skip the grounding check
    # to avoid false positives and save one API call.
    _REFUSAL = "The provided sources do not contain enough information"
    if _REFUSAL in answer_text:
        return {
            "answer": answer_text,
            "model": model,
            "tokens_used": tokens_used,
            "sources_provided": [
                {"paper_title": c["paper_title"], "chunk_index": c["chunk_index"]}
                for c in context_chunks
            ],
            "grounded": True,
            "unsupported_claims": [],
        }

    import time; time.sleep(8)  # brief pause so grounding check doesn't hit the same TPM window
    grounding = _check_grounding(answer_text, context_chunks, model=model, api_key=api_key)

    return {
        "answer": answer_text,
        "model": model,
        "tokens_used": tokens_used,
        "sources_provided": [
            {"paper_title": c["paper_title"], "chunk_index": c["chunk_index"]}
            for c in context_chunks
        ],
        "grounded": grounding["grounded"],
        "unsupported_claims": grounding["unsupported_claims"],
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    dummy_chunks = [
        {
            "paper_title": "Attention Is All You Need",
            "paper_id": "1706.03762",
            "chunk_index": 5,
            "similarity": 0.82,
            "text": (
                "Multi-Head Attention: Instead of performing a single attention function "
                "with dmodel-dimensional keys, values and queries, we found it beneficial "
                "to linearly project the queries, keys and values h times with different, "
                "learned linear projections to dk, dk and dv dimensions, respectively."
            ),
        },
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "paper_id": "1810.04805",
            "chunk_index": 3,
            "similarity": 0.71,
            "text": (
                "BERT uses multi-head self-attention in each of its layers. "
                "The attention mechanism allows the model to jointly attend to information "
                "from different representation subspaces at different positions."
            ),
        },
    ]

    result = generate_answer("What is multi-head attention?", dummy_chunks)
    print(f"Answer:\n{result['answer']}\n")
    print(f"Model: {result['model']}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Grounded: {result['grounded']}")
    if result["unsupported_claims"]:
        print(f"Unsupported claims: {result['unsupported_claims']}")
    print(f"Sources provided: {result['sources_provided']}")
