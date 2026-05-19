# AI Research Paper Assistant — Hybrid RAG with Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)](https://trychroma.com)
[![Deployed on Railway](https://img.shields.io/badge/Railway-deployed-blueviolet?logo=railway)](https://web-production-bb694.up.railway.app/)

**Live Demo →** [https://web-production-bb694.up.railway.app/](https://web-production-bb694.up.railway.app/)

A hybrid retrieval system that combines a **knowledge graph** with **dense vector search** to answer questions over 20 seminal AI/NLP papers. Benchmarked against plain RAG to measure where graph-guided retrieval wins and where it doesn't.

---

## The Problem

Plain RAG retrieves chunks by embedding similarity alone. For questions that cross paper boundaries — *"Which papers build on BERT?"*, *"How did efficient attention evolve from 2019 to 2021?"* — the vector index silently returns the closest single-paper match and misses the relational structure entirely.

**Example failure:**
> Query: *"Which papers build on BERT?"*
> - Plain RAG returns: 3 papers (closest embedding match)
> - Hybrid RAG returns: 7 papers (ALBERT, RoBERTa, ELECTRA, XLNet + 3 more via graph traversal)

The graph knows the `builds_on` relationship. The vector index does not.

---

## Live Demo

**Try it here:** [https://web-production-bb694.up.railway.app/](https://web-production-bb694.up.railway.app/)

The demo runs both pipelines in parallel and shows results side-by-side:

| Query type | Example |
|------------|---------|
| Connection (graph wins) | *"Which papers build on BERT?"* |
| Multi-hop | *"How did efficient attention evolve from 2019 to 2021?"* |
| Factual | *"What is multi-head attention?"* |
| Comparison | *"How do Reformer and Linformer each reduce attention complexity?"* |

When the knowledge graph fires, a green banner shows which papers it found that plain RAG missed.

---

## Architecture

```
User query
    │
    ├─► Query Classifier (Groq LLM + keyword cache)
    │       factual / builds_on / multi_hop / out_of_scope
    │
    ├─► [builds_on] ──────────────► Knowledge Graph traversal (NetworkX)
    │                                    128 triples · 6 relation types
    │                                    ↓
    │                               Scoped vector search
    │                               (top-1 chunk per graph paper, ChromaDB)
    │
    ├─► [multi_hop, non-comparison] ► Query Decomposition
    │                                    break into sub-questions
    │                                    retrieve 1 chunk per sub-question
    │                                    merge → cap at 6 chunks
    │
    └─► [factual / fallback] ────────► Plain vector search
                                           top-k cosine similarity

Retrieved chunks → Groq LLM (llama-3.1-8b-instant) → Cited answer
                                                      → Grounding check
```

### Knowledge graph relations
`builds_on` · `addresses_problem` · `uses_technique` · `applies_to_domain` · `authored_by` · `cites`

### Key components

| Component | Detail |
|-----------|--------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB with cosine similarity |
| Chunks | 1,077 chunks · 512 tokens · 50-token overlap |
| Chunk prefix | `[Paper: {title}, Year: {year}]` injected for temporal reasoning |
| LLM | Groq `llama-3.1-8b-instant` via REST API |
| Graph | NetworkX MultiDiGraph · 128 triples · LLM-extracted |
| HyDE | Hypothetical Document Embeddings — enabled for non-factual queries only |
| API | FastAPI with `asyncio.gather` parallel execution of both pipelines |

---

## Evaluation Results

LLM-as-judge scoring (0–7 scale) across 50 benchmark questions covering 4 reasoning types: factual, relational, multi-hop, and comparison.

| System | Mean Score | Questions |
|--------|-----------|-----------|
| Plain RAG (vector only) | 5.97 / 7 | 30 |
| **Hybrid RAG (graph + vector)** | **6.16 / 7** | **50** |

**Where hybrid wins:** relational and connection queries — the graph surfaces citation paths that embedding similarity misses entirely.

**Where scores are similar:** single-paper factual questions — graph adds no signal when the answer lives in one chunk.

**Honest limitation:** 0.19 score gain is modest. At this corpus size (20 papers), the graph's advantage shows clearly on connection queries but the aggregate number is small. The architecture is designed to scale — the signal grows with corpus size.

---

## Design Decisions

**Graph-scoped retrieval, not graph-only**
The graph identifies *which papers* are relevant; the vector index finds *which chunk* within those papers. Pure graph traversal returns paper-level context — not the specific passage the LLM needs to generate a cited answer.

**1 chunk per graph paper**
When a traversal returns 7 papers, taking top-5 chunks total would let 2 papers dominate. Taking 1 chunk per paper ensures every graph-identified paper gets representation in the context window.

**Comparison guard on decomposition**
*"How do Reformer and Linformer each reduce attention complexity?"* looks multi-hop but needs both papers simultaneously. Decomposing it into sub-questions retrieves each paper separately and produces hallucinated comparisons. The guard routes comparison questions to standard hybrid retrieval instead.

**HyDE disabled for factual queries**
Generating a hypothetical answer for *"What is the learning rate used in BERT?"* embeds into the wrong region of the space — the hypothesis talks about fine-tuning, not the original BERT hyperparameters. HyDE is only activated for connection and multi-hop questions where semantic expansion helps.

**In-process classifier cache**
The query classifier is called by both the pipeline orchestrator and the retrieval module. A module-level dict cache eliminates the duplicate Groq API call on every request.

---

## Project Structure

```
graphrag-project/
├── data/
│   ├── chroma_db/            # Persistent ChromaDB vector store (committed)
│   └── graph/
│       └── raw_triples.json  # 128 knowledge graph triples
├── src/
│   ├── api.py                # FastAPI app — /query endpoint + frontend serving
│   ├── chunking.py           # Token-based chunking with year prefix injection
│   ├── decomposition.py      # Query decomposition for multi-hop questions
│   ├── embeddings.py         # Sentence-transformers embeddings + HyDE
│   ├── graph_store.py        # NetworkX knowledge graph loader
│   ├── hybrid_pipeline.py    # Hybrid retrieval pipeline
│   ├── llm.py                # Groq API wrapper with retry + grounding check
│   ├── query_classifier.py   # LLM-based query type classifier with cache
│   ├── rag_pipeline.py       # Plain RAG baseline pipeline
│   ├── retrieval.py          # Graph → scoped vector retrieval logic
│   └── vector_store.py       # ChromaDB wrapper
├── scripts/
│   └── ingest_all.py         # One-time ingestion: extract → chunk → embed → store
├── frontend/
│   └── index.html            # Single-file demo UI (no build tools)
├── railway.toml              # Railway deployment config
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Prachisahu-0311/AI-Research-Paper-Assistant-with-RAG-and-Contextual-Graph-Retrieval-.git
cd graphrag-project
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Create `.env` in the project root:
```
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.1-8b-instant
```

Run locally:
```bash
uvicorn src.api:app --reload
# → http://localhost:8000
```

> `data/chroma_db/` is committed to the repo — no re-ingestion needed to run the demo.

---

## Papers Indexed

20 seminal Transformer-era papers (2017–2021):

| Year | Paper |
|------|-------|
| 2017 | Attention Is All You Need |
| 2018 | BERT · Transformer-XL |
| 2019 | RoBERTa · XLNet · ALBERT · T5 · GPT-2 · Transformer-XL |
| 2020 | GPT-3 · Reformer · Longformer · Linformer · Performers · ELECTRA · PET · ViT |
| 2021 | Switch Transformers · CLIP |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Knowledge graph | NetworkX |
| LLM inference | Groq API (llama-3.1-8b-instant) |
| Tokenisation | tiktoken (cl100k_base) |
| Deployment | Railway |
| Frontend | Vanilla HTML/JS (no build tools) |
