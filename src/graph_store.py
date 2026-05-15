"""Knowledge graph wrapper using NetworkX."""

import json
import re
from pathlib import Path
from collections import defaultdict

VALID_RELATIONSHIPS = {
    "addresses_problem",
    "builds_on",
    "applies_to_domain",
    "uses_technique",
    "authored_by",
    "cites",
}

CANONICAL_NAMES = {
    "transformer": "Attention Is All You Need",
    "original transformer": "Attention Is All You Need",
    "vaswani et al.": "Attention Is All You Need",
    "vaswani et al": "Attention Is All You Need",
    "bert-base": "BERT",
    "bert-large": "BERT",
    "devlin et al.": "BERT",
    "devlin et al": "BERT",
    "roberta-base": "RoBERTa",
    "liu et al.": "RoBERTa",
    "liu et al": "RoBERTa",
    "vit-b": "ViT",
    "vit-l": "ViT",
    "vision transformer": "ViT",
    "dosovitskiy et al.": "ViT",
    "dosovitskiy et al": "ViT",
    "switch transformer": "Switch Transformers",
    "gpt3": "GPT-3",
    "brown et al.": "GPT-3",
    "brown et al": "GPT-3",
}

KNOWN_PAPERS = {
    "Attention Is All You Need", "BERT", "RoBERTa", "ALBERT", "DistilBERT",
    "XLNet", "GPT-3", "T5", "Transformer-XL", "Reformer", "Linformer",
    "Longformer", "Big Bird", "Performers", "Universal Transformers",
    "ViT", "CLIP", "Switch Transformers", "ELECTRA", "PET"
}


def normalize_entity(name: str) -> str:
    """Normalize a paper name to its canonical form."""
    cleaned = name.strip().strip("()")
    lower = cleaned.lower()
    if lower in CANONICAL_NAMES:
        return CANONICAL_NAMES[lower]
    for alias, canonical in CANONICAL_NAMES.items():
        if alias in lower:
            return canonical
    return cleaned


def parse_triples(raw_text: str) -> list[dict]:
    """
    Parse LLM output into a list of triple dicts.
    Handles format: (Subject, relationship_type, Object)
    Returns list of {"subject": ..., "relation": ..., "object": ...}
    """
    triples = []
    pattern = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
    for match in pattern.finditer(raw_text):
        subject = normalize_entity(match.group(1).strip())
        relation = match.group(2).strip().lower()
        obj = match.group(3).strip()
        if relation not in VALID_RELATIONSHIPS:
            continue
        if relation == "authored_by":
            obj = obj.strip().split()[0].rstrip(".,")
        if relation in ("addresses_problem", "uses_technique"):
            obj = obj.lower().replace(" ", "_").replace("-", "_")
        if relation == "applies_to_domain":
            obj = obj.lower().split()[0]
        triples.append({
            "subject": subject,
            "relation": relation,
            "object": obj
        })
    return triples


def validate_triple(triple: dict) -> bool:
    """Return True if the triple passes basic quality checks."""
    if triple["relation"] not in VALID_RELATIONSHIPS:
        return False
    if triple["relation"] == "builds_on" and triple["object"] not in KNOWN_PAPERS:
        return False
    if not triple["subject"] or not triple["object"]:
        return False
    return True


def save_triples(triples: list[dict], output_path: str = "data/graph/raw_triples.json") -> None:
    """Save triples to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2)
    print(f"Saved {len(triples)} triples to {output_path}")


def load_triples(path: str = "data/graph/raw_triples.json") -> list[dict]:
    """Load triples from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def triples_summary(triples: list[dict]) -> None:
    """Print a summary of extracted triples by relationship type."""
    by_relation = defaultdict(list)
    for t in triples:
        by_relation[t["relation"]].append(t)
    print(f"\nTotal triples: {len(triples)}")
    for rel, items in sorted(by_relation.items()):
        print(f"  {rel}: {len(items)}")


import networkx as nx


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    """Build a NetworkX graph from triples list."""
    G = nx.MultiDiGraph()
    for t in triples:
        G.add_edge(
            t["subject"],
            t["object"],
            relation=t["relation"],
            source_paper_id=t.get("source_paper_id", "")
        )
    return G


def query_papers_by_problem(G: nx.MultiDiGraph, problem: str) -> list[str]:
    """Find all papers that address a given problem."""
    papers = []
    for u, v, data in G.edges(data=True):
        if data["relation"] == "addresses_problem" and problem.lower() in v.lower():
            papers.append(u)
    return list(set(papers))


def query_papers_by_technique(G: nx.MultiDiGraph, technique: str) -> list[str]:
    """Find all papers that use a given technique."""
    papers = []
    for u, v, data in G.edges(data=True):
        if data["relation"] == "uses_technique" and technique.lower() in v.lower():
            papers.append(u)
    return list(set(papers))


def query_papers_building_on(G: nx.MultiDiGraph, base_paper: str) -> list[str]:
    """Find all papers that build on a given paper."""
    papers = []
    for u, v, data in G.edges(data=True):
        if data["relation"] == "builds_on" and v.lower() == base_paper.lower():
            papers.append(u)
    return list(set(papers))


def query_papers_citing(G: nx.MultiDiGraph, cited_paper: str) -> list[str]:
    """Find all papers that explicitly cite a given paper."""
    papers = []
    for u, v, data in G.edges(data=True):
        if data["relation"] == "cites" and v.lower() == cited_paper.lower():
            papers.append(u)
    return list(set(papers))


def query_papers_by_domain(G: nx.MultiDiGraph, domain: str) -> list[str]:
    """Find all papers that apply to a given domain."""
    papers = []
    for u, v, data in G.edges(data=True):
        if data["relation"] == "applies_to_domain" and domain.lower() in v.lower():
            papers.append(u)
    return list(set(papers))


def graph_summary(G: nx.MultiDiGraph) -> None:
    """Print graph statistics."""
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    by_relation: dict[str, int] = defaultdict(int)
    for _, _, data in G.edges(data=True):
        by_relation[data["relation"]] += 1
    for rel, count in sorted(by_relation.items()):
        print(f"  {rel}: {count}")
