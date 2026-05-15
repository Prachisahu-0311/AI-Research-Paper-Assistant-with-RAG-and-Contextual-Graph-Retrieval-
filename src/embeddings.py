"""Embedding generation using sentence-transformers."""
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed(text: str) -> list[float]:
    """Embed a single text string into a 384-dim vector."""
    return get_model().encode(text).tolist()


def hyde_embed(question: str) -> list[float]:
    """
    HyDE: generate a hypothetical answer, embed it instead of the question.
    Falls back to regular embedding if LLM call fails.
    """
    import os
    import httpx
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return embed(question)

    hyde_prompt = (
        "Write a single technical sentence that directly answers this question "
        "about AI research papers. Be specific and factual. "
        "Do not say 'I' or 'the answer is'. Just state the fact directly.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        with httpx.Client(timeout=15.0, verify=False) as client:
            r = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": hyde_prompt}],
                    "max_tokens": 80,
                    "temperature": 0.1,
                },
            )
        if r.status_code == 200:
            hypothesis = r.json()["choices"][0]["message"]["content"].strip()
            return embed(hypothesis)
        print(f"HyDE fallback to standard embed: HTTP {r.status_code}")
    except Exception as e:
        print(f"HyDE fallback to standard embed: {e}")
    return embed(question)


if __name__ == "__main__":
    sample = "Attention is all you need."
    vector = embed(sample)
    print(f"Embedded text into vector of length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
