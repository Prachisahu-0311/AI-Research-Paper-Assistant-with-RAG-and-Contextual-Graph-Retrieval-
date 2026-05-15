"""Test the 7 graph queries corresponding to questions plain RAG failed on."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_store import (
    load_triples, build_graph, graph_summary,
    query_papers_by_problem, query_papers_by_technique,
    query_papers_building_on, query_papers_by_domain,
)

triples = load_triples()
G = build_graph(triples)

print("=== Graph Summary ===")
graph_summary(G)

print("\n=== Query Tests (these are the 7 questions plain RAG failed) ===\n")

print("Q: Which papers address quadratic attention complexity? (c001, c004, c005)")
print("A:", query_papers_by_problem(G, "quadratic_attention"))
print()

print("Q: Which papers build on BERT? (c002)")
print("A:", query_papers_building_on(G, "BERT"))
print()

print("Q: Which papers apply to vision? (c003)")
print("A:", query_papers_by_domain(G, "vision"))
print()

print("Q: Which papers apply to multimodal? (c008)")
print("A:", query_papers_by_domain(G, "multimodal"))
print()

print("Q: Which papers use knowledge distillation? (c010)")
print("A:", query_papers_by_technique(G, "knowledge_distillation"))
print()

print("Q: Which papers use mixture of experts? (c007)")
print("A:", query_papers_by_technique(G, "mixture_of_experts"))
print()

print("Q: Which papers address large model size? (c002 related)")
print("A:", query_papers_by_problem(G, "large_model_size"))
print()
