"""Retrieval logic: vector search, graph traversal, hybrid routing."""
import networkx as nx

from src.graph_store import (
    query_papers_by_problem,
    query_papers_by_technique,
    query_papers_building_on,
    query_papers_by_domain,
    query_papers_citing,
)
from src.query_classifier import classify_query

# Map paper names to arXiv IDs for scoped vector search
_PAPER_NAME_TO_ID: dict[str, str] = {
    "Attention Is All You Need": "1706.03762",
    "BERT": "1810.04805",
    "RoBERTa": "1907.11692",
    "ALBERT": "1909.11942",
    "DistilBERT": "1910.01108",
    "XLNet": "1906.08237",
    "GPT-3": "2005.14165",
    "T5": "1910.10683",
    "Transformer-XL": "1901.02860",
    "Reformer": "2001.04451",
    "Linformer": "2005.12872",
    "Longformer": "2005.00743",
    "Big Bird": "2002.12327",
    "Performers": "2006.16236",
    "Universal Transformers": "1807.03819",
    "ViT": "2010.11929",
    "CLIP": "2103.00020",
    "Switch Transformers": "2101.03961",
    "ELECTRA": "2003.05997",
    "PET": "2003.10555",
}

# Pattern-based routing rules — evaluated in order, first match wins
_PROBLEM_PATTERNS: list[tuple[str, list[str]]] = [
    ("quadratic_attention_complexity", [
        "quadratic", "efficient attention", "long sequence", "o(n^2)",
        "attention complexity", "linear attention", "sparse attention",
        "scalab", "efficient transform",
    ]),
    ("limited_context_length", [
        "context length", "long context", "long document", "context window",
        "extended context", "memory span",
    ]),
    ("large_model_size", [
        "large model", "model size", "compression", "smaller model",
        "distill", "parameter count", "parameter-efficient",
    ]),
    ("training_efficiency", [
        "training efficiency", "trillion parameter", "scale model",
        "mixture of expert",
    ]),
    ("pretrain_finetune_discrepancy", [
        "pretrain finetune", "pretrain-finetune", "[mask] token",
        "mask token discrepancy",
    ]),
    ("few_shot_learning", [
        "few-shot", "few shot", "zero-shot", "zero shot", "in-context learning",
    ]),
]

_TECHNIQUE_PATTERNS: list[tuple[str, list[str]]] = [
    ("knowledge_distillation", ["distillation", "distilled", "knowledge transfer", "teacher"]),
    ("mixture_of_experts", ["mixture of experts", "moe", " expert routing", "sparse expert"]),
    ("locality_sensitive_hashing", ["lsh", "locality sensitive hashing", "locality-sensitive"]),
    ("contrastive_learning", ["contrastive learning", "contrastive loss", "clip training"]),
    ("masked_language_modeling", ["masked language modeling", " mlm ", "masked lm",
                                   "bert pretraining", "bert's pretraining"]),
    ("permutation_language_modeling", ["permutation language", "xlnet", "autoregressive pretraining"]),
    ("replaced_token_detection", ["replaced token", "discriminator pretraining", "electra"]),
    ("permutation_language_modeling", ["permutation language", "autoregressive pretraining"]),
    ("random_feature_approximation", ["random feature", "performer", "favor+"]),
    ("low_rank_approximation", ["low-rank", "low rank approximation", "linformer"]),
    ("factorized_embedding_parameterization", ["factorized embedding", "parameter sharing"]),
]

_BUILDS_ON_PATTERNS: list[tuple[str, list[str]]] = [
    ("BERT", [
        "build on bert", "build upon bert", "build upon", "improve upon",
        "extend bert", "based on bert", "improve bert", "bert variant",
        "bert model", "bert architecture", "improve or build upon bert",
        "directly improve", "successor to", "which papers improve",
        "which papers extend", "pretraining approach",
        "discover about bert", "about bert", "original training",
    ]),
    ("Attention Is All You Need", [
        "build on transformer", "extend transformer", "based on attention is all",
        "based on the transformer",
    ]),
]

_DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("vision", ["vision", " image ", "visual ", "image classification", "object detection"]),
    ("multimodal", ["multimodal", "multi-modal", "vision-language", "image-text"]),
    ("genomics", ["genomic", "dna ", "genome", "biological sequence"]),
]


def _keyword_graph_query(G: nx.MultiDiGraph, question: str) -> list[str]:
    """Fallback keyword-based router used when LLM classifier is unavailable."""
    q = question.lower()
    for problem, keywords in _PROBLEM_PATTERNS:
        if any(kw in q for kw in keywords):
            results = query_papers_by_problem(G, problem)
            if results:
                return results
    for base_paper, keywords in _BUILDS_ON_PATTERNS:
        if any(kw in q for kw in keywords):
            results = query_papers_building_on(G, base_paper)
            if results:
                return results
    for domain, keywords in _DOMAIN_PATTERNS:
        if any(kw in q for kw in keywords):
            results = query_papers_by_domain(G, domain)
            if results:
                return results
    for technique, keywords in _TECHNIQUE_PATTERNS:
        if any(kw in q for kw in keywords):
            results = query_papers_by_technique(G, technique)
            if results:
                return results
    return []


def graph_query(G: nx.MultiDiGraph, question: str, use_llm: bool = True) -> list[str]:
    """
    Route a question to the appropriate graph query using the LLM classifier
    (with keyword fallback). Returns a list of paper names, or [] for vector-only.
    """
    import os

    if use_llm and os.environ.get("GROQ_API_KEY"):
        clf = classify_query(question)
        qtype, target = clf["type"], clf["target"]

        if qtype == "out_of_scope" or qtype == "multi_hop":
            return []

        if qtype == "factual":
            # Classifier may return "factual" as a rate-limit fallback; try
            # keyword matching as a safety net before giving up.
            return _keyword_graph_query(G, question)

        if qtype == "builds_on" and target:
            results = query_papers_building_on(G, target)
            if results:
                return results

        if qtype == "connection_problem" and target:
            results = query_papers_by_problem(G, target)
            if results:
                return results

        if qtype == "connection_technique" and target:
            results = query_papers_by_technique(G, target)
            if results:
                return results

        if qtype == "connection_domain" and target:
            results = query_papers_by_domain(G, target)
            if results:
                return results

        if qtype == "cites" and target:
            results = query_papers_citing(G, target)
            if results:
                return results

        return []

    # Fallback: keyword patterns
    return _keyword_graph_query(G, question)


def hybrid_retrieve(
    question: str,
    query_embedding: list[float],
    G: nx.MultiDiGraph,
    vs,
    top_k: int = 5,
) -> tuple[list[dict], list[str]]:
    """
    Hybrid retrieval: graph traversal narrows candidate papers,
    then vector search retrieves relevant chunks from those papers.

    Returns (chunks, graph_papers_found).
    graph_papers_found is [] when falling back to pure vector search.
    """
    graph_papers = graph_query(G, question)

    if graph_papers:
        paper_ids = [_PAPER_NAME_TO_ID[p] for p in graph_papers if p in _PAPER_NAME_TO_ID]

        # Get top-1 chunk per graph-identified paper so every paper the graph
        # found is represented in context — a global top-k scoped search would
        # return all chunks from the 2-3 highest-similarity papers and miss the rest.
        per_paper: list[dict] = []
        seen_ids: set[str] = set()
        for pid in paper_ids:
            chunks = vs.query_scoped(query_embedding, [pid], top_k=1)
            for c in chunks:
                if c["chunk_id"] not in seen_ids:
                    per_paper.append(c)
                    seen_ids.add(c["chunk_id"])

        # Sort by similarity; keep all graph-paper chunks (covers every identified paper)
        per_paper.sort(key=lambda c: c["similarity"], reverse=True)
        return per_paper, graph_papers

    # No graph match — pure vector search
    return vs.query(query_embedding, top_k=top_k), []
