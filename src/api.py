"""
FastAPI backend for GraphRAG vs Plain RAG demo.
Serves both plain RAG and hybrid RAG for side-by-side comparison.
Also serves the frontend HTML at GET /.
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

_executor = ThreadPoolExecutor(max_workers=2)

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
    from src.hybrid_pipeline import _get_graph
    _vs = VectorStore()
    _get_graph()
    _graph_loaded = True
    print(f"Pipeline loaded — {_vs.count()} chunks, graph ready")


class QueryRequest(BaseModel):
    question: str
    use_hyde: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph_loaded, "chunks": _vs.count() if _vs else 0}


@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(req.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars)")

    from src.rag_pipeline import answer_question
    from src.hybrid_pipeline import answer_question_hybrid

    loop = asyncio.get_event_loop()

    try:
        t_start = time.perf_counter()

        # top_k=3 keeps each pipeline under 2500 tokens so both fit in the
        # 6000 TPM free-tier window when running in parallel.
        plain_future = loop.run_in_executor(
            _executor,
            lambda: answer_question(req.question, top_k=3, vector_store=_vs)
        )
        hybrid_future = loop.run_in_executor(
            _executor,
            lambda: answer_question_hybrid(
                req.question, top_k=3, vector_store=_vs, use_hyde=req.use_hyde
            )
        )

        plain_result, hybrid_result = await asyncio.gather(
            plain_future, hybrid_future
        )

        total_ms = round((time.perf_counter() - t_start) * 1000)

        def format_chunks(chunks):
            return [
                {
                    "paper_title": c.get("paper_title", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "similarity": round(c.get("similarity", 0), 3),
                    "text": c.get("text", "")[:300],
                }
                for c in chunks
            ]

        plain_paper_titles = set(
            c.get("paper_title", "") for c in plain_result["retrieved_chunks"]
        )
        hybrid_paper_titles = set(
            c.get("paper_title", "") for c in hybrid_result["retrieved_chunks"]
        )
        graph_papers = hybrid_result.get("graph_papers_used", [])

        return {
            "question": req.question,
            "total_latency_ms": total_ms,
            "plain_rag": {
                "answer": plain_result["answer"],
                "retrieved_chunks": format_chunks(plain_result["retrieved_chunks"]),
                "latency_ms": plain_result.get("total_time_ms", 0),
                "tokens_used": plain_result.get("tokens_used", 0),
                "grounded": plain_result.get("grounded", True),
                "retrieval_mode": "vector_only",
            },
            "hybrid_rag": {
                "answer": hybrid_result["answer"],
                "retrieved_chunks": format_chunks(hybrid_result["retrieved_chunks"]),
                "graph_papers": graph_papers,
                "retrieval_mode": hybrid_result.get("retrieval_mode", ""),
                "sub_questions": hybrid_result.get("sub_questions", []),
                "latency_ms": hybrid_result.get("total_time_ms", 0),
                "tokens_used": hybrid_result.get("tokens_used", 0),
                "grounded": hybrid_result.get("grounded", True),
            },
            "comparison": {
                "plain_papers": list(plain_paper_titles),
                "hybrid_papers": list(hybrid_paper_titles),
                "graph_found": graph_papers,
                # Full-title papers hybrid retrieved that plain did NOT retrieve
                "graph_only_papers": list(hybrid_paper_titles - plain_paper_titles),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/plain")
async def query_plain(req: QueryRequest):
    """Lightweight endpoint — plain RAG only, faster for testing."""
    from src.rag_pipeline import answer_question
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: answer_question(req.question, top_k=5, vector_store=_vs)
    )
    return {
        "question": req.question,
        "answer": result["answer"],
        "retrieved_chunks": [
            {
                "paper_title": c.get("paper_title", ""),
                "chunk_index": c.get("chunk_index", 0),
                "similarity": round(c.get("similarity", 0), 3),
                "text": c.get("text", "")[:300],
            }
            for c in result["retrieved_chunks"]
        ],
        "latency_ms": result.get("total_time_ms", 0),
        "grounded": result.get("grounded", True),
    }


@app.get("/example-questions")
def example_questions():
    return {
        "connection": [
            "Which papers build on BERT?",
            "Which papers address the quadratic attention complexity problem?",
            "Which papers apply Transformers to vision tasks?",
        ],
        "multi_hop": [
            "How do Reformer and Linformer each reduce attention complexity?",
            "What did RoBERTa and ALBERT each discover about BERT training?",
            "How did efficient attention methods evolve from 2019 to 2021?",
        ],
        "factual": [
            "What is multi-head attention?",
            "How many parameters does GPT-3 have?",
            "How does DistilBERT use knowledge distillation?",
        ],
    }
