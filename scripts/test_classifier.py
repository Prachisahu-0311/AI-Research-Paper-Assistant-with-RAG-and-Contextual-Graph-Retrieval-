"""Test the LLM classifier against all eval questions and print accuracy."""
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import load_questions
from src.query_classifier import classify_query

# Expected type by question ID prefix
_EXPECTED_BROAD = {
    "f": "factual",
    "c": "connection",  # builds_on / connection_problem / connection_technique / connection_domain
    "m": "multi_hop",
    "o": "out_of_scope",
}

_CONNECTION_TYPES = {"builds_on", "connection_problem", "connection_technique", "connection_domain", "cites"}


def broad_type(clf_type: str) -> str:
    if clf_type in _CONNECTION_TYPES:
        return "connection"
    return clf_type


questions = load_questions()
print(f"Testing classifier on {len(questions)} questions (5s delay between calls)\n")
print(f"{'ID':<6} {'Expected':<12} {'Got':<25} {'Target':<35} {'OK'}")
print("-" * 90)

correct = 0
for q in questions:
    clf = classify_query(q["question"])
    expected_broad = _EXPECTED_BROAD.get(q["id"][0], "?")
    got_broad = broad_type(clf["type"])
    ok = "✓" if got_broad == expected_broad else "✗"
    if got_broad == expected_broad:
        correct += 1
    print(f"{q['id']:<6} {expected_broad:<12} {clf['type']:<25} {str(clf['target']):<35} {ok}")
    time.sleep(5)  # gentle TPM pacing for classifier calls

print(f"\nAccuracy: {correct}/{len(questions)} = {correct/len(questions):.0%}")
print("\nKey routing checks:")
for qid in ["c002", "m002", "c011", "c015", "c016"]:
    match = next((q for q in questions if q["id"] == qid), None)
    if match:
        clf = classify_query(match["question"])
        print(f"  {qid}: '{match['question'][:55]}' → {clf}")
        time.sleep(5)
