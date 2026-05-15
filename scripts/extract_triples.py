"""
Extract knowledge graph triples from all 20 papers using the Groq LLM.
Reads SCHEMA.md extraction rules and applies them to each paper's text.
Saves raw triples to data/graph/raw_triples.json
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_store import parse_triples, validate_triple, save_triples, triples_summary

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PAPER_NAMES = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT",
    "1907.11692": "RoBERTa",
    "1909.11942": "ALBERT",
    "1910.01108": "DistilBERT",
    "1906.08237": "XLNet",
    "2005.14165": "GPT-3",
    "1910.10683": "T5",
    "1901.02860": "Transformer-XL",
    "2001.04451": "Reformer",
    "2005.12872": "Linformer",
    "2005.00743": "Longformer",
    "2002.12327": "Big Bird",
    "2006.16236": "Performers",
    "1807.03819": "Universal Transformers",
    "2010.11929": "ViT",
    "2103.00020": "CLIP",
    "2101.03961": "Switch Transformers",
    "2003.05997": "ELECTRA",
    "2003.10555": "PET",
}

EXTRACTION_PROMPT = """Extract knowledge graph triples from this AI paper text. Paper: {paper_name}

Output format: (Subject, relation, Object) — one per line, nothing else.
Allowed relations (only these 5):
- addresses_problem: object is snake_case problem (e.g. quadratic_attention_complexity)
- builds_on: object is another paper name (e.g. BERT, Attention Is All You Need)
- applies_to_domain: object is one word (vision, language, multimodal, genomics)
- uses_technique: object is snake_case technique (e.g. masked_language_modeling)
- authored_by: FIRST author last name only — extract at most ONE authored_by triple per paper

Only extract facts explicitly stated. No inference.

TEXT:
{text_chunk}"""


def _call_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    """Make a single Groq REST call via httpx. Retries on 429/413 rate limits."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    for attempt in range(3):
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code in (429, 413) and attempt < 2:
            wait = 60 if attempt == 1 else 20
            print(f"    Rate limited ({r.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Groq API: max retries exceeded")


def chunk_for_extraction(text: str, max_chars: int = 800) -> list[str]:
    """Split text into small chunks. Hard-truncates to keep total request under 400 tokens."""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    chunks = []
    current = ""
    for para in paragraphs:
        para = para[:max_chars]  # hard-truncate any single long paragraph
        if len(current) + len(para) < max_chars:
            current += " " + para
        else:
            if current:
                chunks.append(current.strip()[:max_chars])
            current = para
    if current:
        chunks.append(current.strip()[:max_chars])
    return chunks[:4]


def extract_from_paper(api_key: str, paper_id: str, paper_name: str, text: str) -> list[dict]:
    """Extract triples from a single paper. Returns validated triples."""
    chunks = chunk_for_extraction(text)
    all_triples = []
    seen: set[tuple] = set()

    for i, chunk in enumerate(chunks):
        prompt = EXTRACTION_PROMPT.format(paper_name=paper_name, text_chunk=chunk)
        try:
            raw = _call_groq(prompt, api_key)
            triples = parse_triples(raw)
            for t in triples:
                key = (t["subject"], t["relation"], t["object"])
                if key not in seen and validate_triple(t):
                    t["source_paper_id"] = paper_id
                    all_triples.append(t)
                    seen.add(key)
            time.sleep(15)  # 6000 TPM / ~300 tokens per request → safe at 15s gap
        except Exception as e:
            print(f"    Error on chunk {i+1}: {e}")
            time.sleep(30)

    return all_triples


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    all_triples: list[dict] = []
    papers = sorted(Path("data/processed").glob("*.txt"))
    print(f"Found {len(papers)} papers to process\n")

    for idx, paper_path in enumerate(papers):
        paper_id = paper_path.stem
        paper_name = PAPER_NAMES.get(paper_id, paper_id)
        text = paper_path.read_text(encoding="utf-8")

        print(f"[{idx+1}/{len(papers)}] {paper_name} ({paper_id})")
        triples = extract_from_paper(api_key, paper_id, paper_name, text)
        all_triples.extend(triples)
        print(f"    → {len(triples)} triples extracted")
        if idx < len(papers) - 1:
            print(f"    (sleeping 65s for TPM reset...)")
            time.sleep(65)  # ensure fresh 6000-TPM window for each paper

    print("\nDeduplicating across all papers...")
    seen: set[tuple] = set()
    unique_triples: list[dict] = []
    for t in all_triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            unique_triples.append(t)
            seen.add(key)

    save_triples(unique_triples)
    triples_summary(unique_triples)
    print("\nDone. Review data/graph/raw_triples.json before Day 2.")


if __name__ == "__main__":
    main()
