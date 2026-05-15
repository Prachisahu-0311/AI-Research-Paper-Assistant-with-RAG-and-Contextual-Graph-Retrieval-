"""Evaluation harness: runs benchmarks comparing RAG systems."""
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq(messages: list[dict], model: str, temperature: float, max_tokens: int, api_key: str, response_json: bool = False) -> dict:
    payload: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        with httpx.Client(timeout=30.0, verify=False) as client:
            r = client.post(_GROQ_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 and attempt < 2:
            m = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', r.text)
            wait = (float(m.group(1) or 0) * 60 + float(m.group(2)) + 2) if m else 20.0
            time.sleep(min(wait, 120))
            continue
        raise RuntimeError(f"Groq API {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Groq API: max retries exceeded")

_JUDGE_PROMPT = """You are an expert evaluator of AI research question-answering systems.

Given a question, a reference answer, and a system's generated answer, score the system answer.

correctness (0-3):
  3 = Correct and complete
  2 = Mostly correct with minor errors or omissions
  1 = Partially correct with significant errors
  0 = Wrong, irrelevant, or refuses when it should answer
  Out-of-scope special case: correctness=3 if system says sources lack information; correctness=0 if it answers using outside knowledge.

grounding (0-2):
  2 = All claims supported by cited [Source N] references
  1 = Minor unsupported claims present
  0 = Hallucinations or outside knowledge used
  Out-of-scope special case: grounding=2 if system correctly refuses (no claims to ground).

citation_quality (0-2):
  2 = All claims properly cited with [Source N]
  1 = Citations present but incomplete or misattributed
  0 = No citations or citations are wrong
  Out-of-scope special case: citation_quality=2 if system correctly refuses.

Return ONLY valid JSON, no other text:
{"correctness": <int 0-3>, "grounding": <int 0-2>, "citation_quality": <int 0-2>, "total": <int 0-7>, "reasoning": "<one sentence>"}"""


def load_questions(path: str = "data/eval/questions.json") -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_retrieval_recall(
    retrieved_chunks: list[dict],
    expected_papers: list[str],
) -> float | None:
    """Fraction of expected papers that appear in retrieved chunks. None for out-of-scope."""
    if not expected_papers:
        return None
    retrieved_ids = {c["paper_id"] for c in retrieved_chunks}
    return sum(1 for p in expected_papers if p in retrieved_ids) / len(expected_papers)


def judge_answer(
    question: str,
    reference_answer: str,
    system_answer: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """Score a system answer using LLM-as-judge. Returns scores dict."""
    try:
        resp = _call_groq(
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nReference answer: {reference_answer}\n\nSystem answer: {system_answer}"},
            ],
            model=model,
            temperature=0.0,
            max_tokens=200,
            api_key=api_key,
            response_json=True,
        )
        scores = json.loads(resp["choices"][0]["message"]["content"])
        scores["judge_tokens"] = resp.get("usage", {}).get("total_tokens", 0)
        return scores
    except (json.JSONDecodeError, KeyError, RuntimeError) as e:
        return {"correctness": -1, "grounding": -1, "citation_quality": -1, "total": -1, "reasoning": f"error: {e}", "judge_tokens": 0}


def run_eval(
    questions: list[dict],
    vs,
    judge_model: str = "llama-3.1-8b-instant",
    top_k: int = 5,
) -> list[dict]:
    """Run the full eval loop: retrieve → generate → judge for each question."""
    from src.rag_pipeline import answer_question

    api_key = os.environ["GROQ_API_KEY"]
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i:2}/{len(questions)}] {q['id']} — {q['question'][:55]}...")

        rag_result = answer_question(q["question"], top_k=top_k, vector_store=vs)
        recall = compute_retrieval_recall(
            rag_result["retrieved_chunks"], q.get("expected_papers", [])
        )
        scores = judge_answer(
            q["question"],
            q["reference_answer"],
            rag_result["answer"],
            api_key,
            judge_model,
        )

        recall_str = f"{recall:.0%}" if recall is not None else "N/A"
        print(
            f"          recall={recall_str}  score={scores.get('total', -1)}/7  "
            f"lat={rag_result['total_time_ms']:.0f}ms"
        )

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": rag_result["answer"],
            "retrieval_recall": recall,
            "correctness": scores.get("correctness", -1),
            "grounding": scores.get("grounding", -1),
            "citation_quality": scores.get("citation_quality", -1),
            "total_score": scores.get("total", -1),
            "judge_reasoning": scores.get("reasoning", ""),
            "latency_ms": rag_result["total_time_ms"],
            "rag_tokens": rag_result["tokens_used"],
            "judge_tokens": scores.get("judge_tokens", 0),
            "retrieved_papers": [c["paper_id"] for c in rag_result["retrieved_chunks"]],
            "expected_papers": q.get("expected_papers", []),
        })

        time.sleep(65)  # classifier+generation+grounding+judge = ~4400 tokens; 65s ensures fresh window

    return results


def print_metrics_table(results: list[dict]) -> None:
    """Print a formatted metrics table grouped by question type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    print("\n" + "=" * 74)
    print("PLAIN RAG BASELINE — EVALUATION RESULTS")
    print("=" * 74)
    print(
        f"{'Type':<14} {'N':>3}  {'Recall':>7}  {'Correct/3':>9}  "
        f"{'Ground/2':>8}  {'Cite/2':>6}  {'Score/7':>7}  {'Lat(ms)':>8}"
    )
    print("-" * 74)

    all_recalls: list[float] = []
    all_scores: list[float] = []

    for qtype in ["factual", "connection", "multi_hop", "out_of_scope"]:
        rows = [r for r in by_type.get(qtype, []) if r["total_score"] >= 0]
        if not rows:
            continue
        recalls = [r["retrieval_recall"] for r in rows if r["retrieval_recall"] is not None]
        avg_recall = statistics.mean(recalls) if recalls else None
        avg_correct = statistics.mean(r["correctness"] for r in rows)
        avg_ground = statistics.mean(r["grounding"] for r in rows)
        avg_cite = statistics.mean(r["citation_quality"] for r in rows)
        avg_score = statistics.mean(r["total_score"] for r in rows)
        avg_lat = statistics.mean(r["latency_ms"] for r in rows)

        recall_str = f"{avg_recall:.0%}" if avg_recall is not None else "  N/A"
        print(
            f"{qtype:<14} {len(rows):>3}  {recall_str:>7}  {avg_correct:>9.2f}  "
            f"{avg_ground:>8.2f}  {avg_cite:>6.2f}  {avg_score:>7.2f}  {avg_lat:>8.0f}"
        )
        all_recalls.extend(r for r in recalls)
        all_scores.extend(r["total_score"] for r in rows)

    if all_scores:
        print("-" * 74)
        print(
            f"{'OVERALL':<14} {len(all_scores):>3}  "
            f"{statistics.mean(all_recalls):.0%}  "
            f"{'':>9}  {'':>8}  {'':>6}  "
            f"{statistics.mean(all_scores):>7.2f}"
        )
    print("=" * 74)

    failures = [r for r in results if 0 <= r["total_score"] < 4]
    if failures:
        print(f"\nLow-scoring answers ({len(failures)} with score < 4/7):")
        for r in sorted(failures, key=lambda x: x["total_score"]):
            print(f"  [{r['id']}] {r['type']} score={r['total_score']}/7 — {r['question'][:52]}")
            print(f"          {r['judge_reasoning']}")
    else:
        print("\nNo low-scoring answers (all scored 4+/7).")
