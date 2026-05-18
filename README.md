# GraphRAG vs Plain RAG: A Comparative Study

**Live demo:** https://web-production-bb694.up.railway.app

Hybrid retrieval system that combines vector search with a knowledge graph, benchmarked against plain RAG on 20 seminal AI/NLP papers (Transformers, BERT, GPT-3, ViT, CLIP, and more).

---

## The Problem

Plain RAG retrieves chunks by embedding similarity alone. For questions that require multi-paper reasoning ("Which papers build on BERT?", "How did efficient attention evolve from 2019 to 2021?"), the vector index silently misses relevant papers that aren't the closest embedding match to the query.

---

## Approach

```
User query
    │
    ├─► Query Classifier (Groq LLM)
    │       factual / builds_on / multi_hop / out_of_scope
    │
    ├─► [builds_on / multi_hop] ──► Knowledge Graph traversal (NetworkX)
    │                                    128 triples, 6 relation types
    │                                    → paper names resolved to IDs
    │                                    │
    │                               Scoped vector search
    │                               (1 chunk per graph paper, ChromaDB)
    │
    ├─► [multi_hop, non-comparison] ──► Query Decomposition
    │                                       break into sub-questions
    │                                       retrieve 1 chunk each, merge
    │
    └─► [factual / fallback] ──► Plain vector search (top-k cosine similarity)

Retrieved chunks → Groq LLM (llama-3.1-8b-instant) → Cited answer
                                                      → Grounding check
```

**Knowledge graph relations:** `builds_on`, `addresses_problem`, `uses_technique`, `applies_to_domain`, `authored_by`, `cites`

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

**Chunks:** 1,077 chunks across 20 papers · 512 tokens · 50-token overlap · year prefix injected for temporal reasoning

**Optional HyDE:** embed a hypothetical answer instead of the raw question (enabled for non-factual queries only)

---

## Evaluation Results

Scored by an LLM judge (0–7 scale) on a 50-question benchmark covering factual, relational, multi-hop, and comparison question types.

| System | Score (mean) | Questions |
|--------|-------------|-----------|
| Plain RAG (vector only) | 5.97 / 7 | 30 |
| **Hybrid RAG (graph + vector)** | **6.16 / 7** | 50 |

**Where hybrid wins:** connection questions ("Which papers build on BERT?") — hybrid retrieves 7 papers vs plain's 3, finding ALBERT, RoBERTa, ELECTRA, and XLNet that vector search misses.

**Where scores are similar:** factual questions with a clear best-match chunk — graph adds no signal, both pipelines retrieve the same chunk.

---

## Project Structure

```
graphrag-project/
├── data/
│   ├── chroma_db/          # Persistent ChromaDB vector store (committed)
│   └── graph/
│       └── raw_triples.json  # 128 knowledge graph triples
├── src/
│   ├── api.py              # FastAPI app — serves frontend + /query endpoint
│   ├── chunking.py         # Token-based chunking with year prefix injection
│   ├── decomposition.py    # Query decomposition for multi-hop questions
│   ├── embeddings.py       # Sentence-transformers embeddings + HyDE
│   ├── graph_store.py      # NetworkX knowledge graph loader
│   ├── hybrid_pipeline.py  # Hybrid retrieval pipeline
│   ├── llm.py              # Groq API wrapper with retry + grounding check
│   ├── query_classifier.py # LLM-based query type classifier with cache
│   ├── rag_pipeline.py     # Plain RAG baseline pipeline
│   ├── retrieval.py        # Hybrid retrieval logic (graph → scoped vector)
│   └── vector_store.py     # ChromaDB wrapper
├── scripts/
│   └── ingest_all.py       # One-time ingestion: extract → chunk → embed → store
├── frontend/
│   └── index.html          # Single-file demo UI (no build tools)
├── railway.toml            # Railway deployment config
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Prachisahu-0311/AI-Research-Paper-Assistant-with-RAG-and-Contextual-Graph-Retrieval-.git
cd graphrag-project
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.1-8b-instant
```

Run locally:
```bash
uvicorn src.api:app --reload
# → http://localhost:8000
```

The `data/chroma_db/` is committed — no re-ingestion needed to run the demo.

---

## Key Design Decisions

- **Graph-scoped retrieval over graph-only retrieval:** the graph identifies *which papers* are relevant; the vector index finds *which chunk* within those papers. Pure graph traversal would return paper-level context, not the specific passage the LLM needs.
- **1 chunk per graph paper:** prevents any single paper from dominating the context window when the graph returns 7+ papers.
- **Comparison guard on decomposition:** questions like "How do Reformer and Linformer each reduce attention complexity?" look multi-hop but need both papers simultaneously — decomposition would split them and hallucinate comparisons. The guard routes these to standard hybrid retrieval instead.
- **HyDE disabled for factual queries:** generating a hypothetical answer for "What is the learning rate used in BERT?" embeds into the wrong region of the space. HyDE is only activated for connection and multi-hop questions.
