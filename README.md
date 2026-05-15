# GraphRAG vs Plain RAG: A Comparative Study

Hybrid retrieval system using vector search and knowledge graphs, benchmarked against plain RAG on arXiv AI papers.

## Problem

## Approach

## Results

## Setup

## Usage

## Project Structure

```
graphrag-project/
├── data/
│   ├── papers/         # Raw PDF papers (not tracked in git)
│   └── processed/      # Extracted text files (not tracked in git)
├── src/
│   ├── __init__.py
│   ├── ingestion.py    # PDF text extraction
│   ├── chunking.py     # Text chunking
│   ├── embeddings.py   # Embedding generation
│   ├── vector_store.py # Vector database wrapper
│   ├── graph_store.py  # Knowledge graph wrapper
│   ├── retrieval.py    # Retrieval logic (vector, graph, hybrid)
│   └── evaluation.py   # Benchmarking harness
├── notebooks/          # Exploratory notebooks
├── tests/              # Test suite
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```
