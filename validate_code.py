"""Quick validation test - no API calls needed."""
import sys
sys.path.insert(0, '/c/Users/prach/medmatch-ai/graphrag-project')

from src.llm import _build_context

print("=" * 80)
print("VALIDATION TEST: Context Builder (Grouping by Paper)")
print("=" * 80)

# Test data: mixed chunks from two papers
mixed_chunks = [
    {"paper_title": "Reformer", "paper_id": "2001.04451", "chunk_index": 1, "text": "Reformer chunk 1"},
    {"paper_title": "Reformer", "paper_id": "2001.04451", "chunk_index": 2, "text": "Reformer chunk 2"},
    {"paper_title": "Linformer", "paper_id": "2005.12872", "chunk_index": 1, "text": "Linformer chunk 1"},
    {"paper_title": "Linformer", "paper_id": "2005.12872", "chunk_index": 2, "text": "Linformer chunk 2"},
]

context = _build_context(mixed_chunks)
print("\nContext output:")
print(context)

# Verify structure
checks = [
    ("Has Reformer header", "=== Paper: Reformer (2001.04451) ===" in context),
    ("Has Linformer header", "=== Paper: Linformer (2005.12872) ===" in context),
    ("Reformer chunks grouped", context.index("Reformer") < context.index("Linformer")),
    ("Has [Source 1] label", "[Source 1]" in context),
    ("Has [Source 4] label", "[Source 4]" in context),
]

print("\n✓ Validation Checks:")
for check_name, result in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {check_name}")

all_pass = all(result for _, result in checks)
print(f"\nOverall: {'✓ PASS' if all_pass else '✗ FAIL'}")
