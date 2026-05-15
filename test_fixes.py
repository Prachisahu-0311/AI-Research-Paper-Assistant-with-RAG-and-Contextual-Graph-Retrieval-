"""Manual tests for f007 (leakage) and m003 (grouping) fixes."""
import os
from dotenv import load_dotenv
from src.llm import generate_answer

load_dotenv()

print("=" * 80)
print("TEST 2: f007 - ELECTRA Leakage Detection")
print("=" * 80)
print("\nScenario: Question about ELECTRA, but only PET chunks retrieved (0% recall)")
print("Expected: grounded=False OR answer says 'sources do not contain'\n")

electra_chunks = [
    {
        "paper_title": "PET",
        "paper_id": "2003.10555",
        "chunk_index": 1,
        "text": "Pattern-exploiting training uses cloze questions and patterns for few-shot learning."
    },
    {
        "paper_title": "PET",
        "paper_id": "2003.10555",
        "chunk_index": 2,
        "text": "PET achieves good results with minimal labeled data by using verbalizers and task-specific patterns."
    },
]

result = generate_answer(
    "What is the replaced token detection objective used in ELECTRA?",
    electra_chunks
)

print(f"✓ Grounded: {result['grounded']}")
print(f"✓ Answer (first 200 chars): {result['answer'][:200]}")
print(f"✓ Unsupported claims: {result['unsupported_claims']}")

print("\n" + "=" * 80)
print("TEST 3: m003 - Reformer vs Linformer Grouping")
print("=" * 80)
print("\nScenario: Comparing Reformer and Linformer mechanisms")
print("Expected: grounded=True, separate discussion of each paper, no hallucinations\n")

reform_linform_chunks = [
    {
        "paper_title": "Reformer",
        "paper_id": "2001.04451",
        "chunk_index": 1,
        "text": "Reformer uses locality-sensitive hashing (LSH) to reduce attention complexity from O(n^2) to O(n log n) by hashing queries and keys."
    },
    {
        "paper_title": "Reformer",
        "paper_id": "2001.04451",
        "chunk_index": 2,
        "text": "Reformer also uses reversible layers to reduce memory consumption during training."
    },
    {
        "paper_title": "Linformer",
        "paper_id": "2005.12872",
        "chunk_index": 1,
        "text": "Linformer approximates the attention matrix as a product of low-rank matrices, reducing complexity from O(n^2) to O(n)."
    },
    {
        "paper_title": "Linformer",
        "paper_id": "2005.12872",
        "chunk_index": 2,
        "text": "Linformer uses linear projections to approximate full attention without hashing or pruning."
    },
]

result = generate_answer(
    "Compare the attention mechanisms used in the Reformer and Linformer papers",
    reform_linform_chunks
)

print(f"✓ Grounded: {result['grounded']}")
print(f"✓ Answer (first 250 chars): {result['answer'][:250]}")
print(f"✓ Unsupported claims: {result['unsupported_claims']}")

print("\n" + "=" * 80)
print("TESTS COMPLETE")
print("=" * 80)
