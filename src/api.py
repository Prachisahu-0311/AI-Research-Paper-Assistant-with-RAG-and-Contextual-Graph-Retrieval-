"""
FastAPI backend for GraphRAG vs Plain RAG demo.
Serves both plain RAG and hybrid RAG for side-by-side comparison.
Also serves the frontend HTML at GET /.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="GraphRAG Demo", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/")
def serve_frontend():
    """Serve the demo UI."""
    if _FRONTEND.exists():
        return FileResponse(_FRONTEND, media_type="text/html")
    return {"message": "GraphRAG API is running. POST /query to use it."}

_vs = None
_graph_loaded = False


@app.on_event("startup")
async def startup():
    global _vs, _graph_loaded
    from src.vector_store import VectorStore
    from src.graph_store import load_triples, build_graph
    from src.hybrid_pipeline import _get_graph
    _vs = VectorStore()
    _get_graph()  # warms the graph cache
    _graph_loaded = True
    print(f"Pipeline loaded — {_vs.count()} chunks, graph ready")


class QueryRequest(BaseModel):
    question: str
    use_hyde: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph_loaded, "chunks": _vs.count() if _vs else 0}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(req.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars)")

    from src.rag_pipeline import answer_question
    from src.hybrid_pipeline import answer_question_hybrid

    def fmt_chunks(chunks):
        return [
            {
                "paper_title": c.get("paper_title", ""),
                "chunk_index": c.get("chunk_index", 0),
                "similarity": round(c.get("similarity", 0), 3),
                "text": c.get("text", "")[:300],
            }
            for c in chunks
        ]

    try:
        t0 = time.perf_counter()
        plain = answer_question(req.question, top_k=5, vector_store=_vs)
        plain_ms = round((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        hybrid = answer_question_hybrid(req.question, top_k=5, vector_store=_vs, use_hyde=req.use_hyde)
        hybrid_ms = round((time.perf_counter() - t1) * 1000)

        plain_papers = list({c.get("paper_title", "") for c in plain["retrieved_chunks"]})
        hybrid_papers = list({c.get("paper_title", "") for c in hybrid["retrieved_chunks"]})
        graph_papers = hybrid.get("graph_papers_used", [])

        return {
            "question": req.question,
            "plain_rag": {
                "answer": plain["answer"],
                "retrieved_chunks": fmt_chunks(plain["retrieved_chunks"]),
                "latency_ms": plain_ms,
                "tokens_used": plain.get("tokens_used", 0),
                "grounded": plain.get("grounded", True),
            },
            "hybrid_rag": {
                "answer": hybrid["answer"],
                "retrieved_chunks": fmt_chunks(hybrid["retrieved_chunks"]),
                "graph_papers": graph_papers,
                "retrieval_mode": hybrid.get("retrieval_mode", ""),
                "sub_questions": hybrid.get("sub_questions", []),
                "latency_ms": hybrid_ms,
                "tokens_used": hybrid.get("tokens_used", 0),
                "grounded": hybrid.get("grounded", True),
            },
            "comparison": {
                "plain_papers": plain_papers,
                "hybrid_papers": hybrid_papers,
                "graph_found": graph_papers,
                "hybrid_only_papers": [p for p in hybrid_papers if p not in plain_papers],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/example-questions")
def example_questions():
    return {
        "factual": [
            "What is multi-head attention?",
            "How many parameters does GPT-3 have?",
            "How does DistilBERT use knowledge distillation?",
        ],
        "connection": [
            "Which papers build on BERT?",
            "Which papers address the quadratic attention complexity problem?",
            "Which papers use knowledge distillation?",
            "Which papers apply Transformers to vision tasks?",
        ],
        "multi_hop": [
            "How do Reformer and Linformer each reduce attention complexity?",
            "What did RoBERTa and ALBERT each discover about BERT training?",
            "How did efficient attention methods evolve from 2019 to 2021?",
        ],
    }
