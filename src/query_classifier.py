"""LLM-based query classifier for hybrid retrieval routing."""
import json
import os
import time

import httpx

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# In-process cache so the second classify_query call per query (hybrid_pipeline
# calls it, then graph_query calls it again) is free — no extra API call.
_CLASSIFY_CACHE: dict[str, dict] = {}

# Keyword shortcuts for high-confidence connection queries — avoids LLM call
# entirely for obvious patterns (faster, no rate-limit hit).
_BUILDS_ON_SHORTCUTS = [
    ("build on bert", "BERT"),
    ("build upon bert", "BERT"),
    ("builds on bert", "BERT"),
    ("improve bert", "BERT"),
    ("extend bert", "BERT"),
    ("based on bert", "BERT"),
    ("build on the transformer", "Attention Is All You Need"),
    ("build on attention is all", "Attention Is All You Need"),
    ("build upon the original transformer", "Attention Is All You Need"),
]

_SYSTEM = """You are a query router for a knowledge graph about AI research papers (Transformers/Attention).
Classify questions into one of these types and extract the search target.

Types:
- factual: Specific technical detail about one paper (what is X, how does Y work, how many params). Target: null
- builds_on: Which papers improve/extend/build upon/succeed a specific base paper. Target: "BERT" or "Attention Is All You Need"
- connection_problem: Which papers address a technical problem. Target: snake_case problem name
- connection_technique: Which papers use a specific technique. Target: snake_case technique name
- connection_domain: Which papers apply to a domain. Target: vision|language|multimodal|genomics
- multi_hop: Cross-paper comparisons, evolutions, lineage questions. Target: null
- out_of_scope: Not answerable from AI papers corpus. Target: null

Return ONLY valid JSON: {"type": "<type>", "target": "<target or null>"}"""

_EXAMPLES = """Examples (learn from these):
"What is multi-head attention?" → {"type":"factual","target":null}
"How does BERT use masked language modeling?" → {"type":"factual","target":null}
"How many parameters does GPT-3 have?" → {"type":"factual","target":null}
"What is the patch size used by ViT?" → {"type":"factual","target":null}
"Which papers build on BERT?" → {"type":"builds_on","target":"BERT"}
"Which papers directly improve or build upon BERT's pretraining?" → {"type":"builds_on","target":"BERT"}
"What did RoBERTa and ALBERT each discover about BERT's original training?" → {"type":"builds_on","target":"BERT"}
"Which papers are successors to BERT?" → {"type":"builds_on","target":"BERT"}
"Which papers extend or improve upon BERT?" → {"type":"builds_on","target":"BERT"}
"Which papers modified BERT's pretraining objective?" → {"type":"builds_on","target":"BERT"}
"Which papers build on the original Transformer?" → {"type":"builds_on","target":"Attention Is All You Need"}
"Which papers cite or extend Attention Is All You Need?" → {"type":"builds_on","target":"Attention Is All You Need"}
"Which papers address quadratic attention complexity?" → {"type":"connection_problem","target":"quadratic_attention_complexity"}
"Which papers propose efficient alternatives to standard self-attention?" → {"type":"connection_problem","target":"quadratic_attention_complexity"}
"Which papers address large model size?" → {"type":"connection_problem","target":"large_model_size"}
"Which papers address limited context length?" → {"type":"connection_problem","target":"limited_context_length"}
"Which papers address training efficiency?" → {"type":"connection_problem","target":"training_efficiency"}
"Which papers address the pretrain-finetune discrepancy?" → {"type":"connection_problem","target":"pretrain_finetune_discrepancy"}
"Which papers address sample inefficiency in pretraining?" → {"type":"connection_problem","target":"sample_inefficiency"}
"Which papers use knowledge distillation?" → {"type":"connection_technique","target":"knowledge_distillation"}
"Which papers use mixture of experts?" → {"type":"connection_technique","target":"mixture_of_experts"}
"Which papers use sparse attention patterns?" → {"type":"connection_technique","target":"sparse_attention"}
"Which papers use locality-sensitive hashing for attention?" → {"type":"connection_technique","target":"locality_sensitive_hashing"}
"Which papers use contrastive learning?" → {"type":"connection_technique","target":"contrastive_learning"}
"Which papers use permutation language modeling?" → {"type":"connection_technique","target":"permutation_language_modeling"}
"Which papers use random feature approximation?" → {"type":"connection_technique","target":"random_feature_approximation"}
"Which papers use factorized embedding parameterization?" → {"type":"connection_technique","target":"factorized_embedding_parameterization"}
"Which papers use distillation or compression?" → {"type":"connection_technique","target":"knowledge_distillation"}
"Which papers propose alternatives to the softmax computation?" → {"type":"connection_technique","target":"random_feature_approximation"}
"Which papers apply transformers to vision?" → {"type":"connection_domain","target":"vision"}
"Which papers use multimodal or contrastive pretraining?" → {"type":"connection_domain","target":"multimodal"}
"Which papers apply transformers to image classification?" → {"type":"connection_domain","target":"vision"}
"Which papers apply transformers to genomics?" → {"type":"connection_domain","target":"genomics"}
"Which papers study or demonstrate extreme scaling of language models?" → {"type":"connection_problem","target":"training_efficiency"}
"How did efficient attention methods evolve from 2019 to 2021?" → {"type":"multi_hop","target":null}
"Compare BERT and ELECTRA pretraining objectives" → {"type":"multi_hop","target":null}
"How do Reformer and Linformer differ?" → {"type":"multi_hop","target":null}
"What is the capital of France?" → {"type":"out_of_scope","target":null}
"Who won the 2020 election?" → {"type":"out_of_scope","target":null}"""


def classify_query(
    question: str,
    api_key: str | None = None,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """Classify a query and extract its search target.

    Returns {"type": str, "target": str | None}.
    Falls back to {"type": "factual", "target": None} on any error.
    """
    # Return cached result immediately (avoids duplicate LLM call when
    # hybrid_pipeline.py and graph_query() both classify the same question).
    if question in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[question]

    q_lower = question.lower()

    # Instant keyword shortcuts for builds_on — no LLM call needed.
    for pattern, target in _BUILDS_ON_SHORTCUTS:
        if pattern in q_lower:
            result = {"type": "builds_on", "target": target}
            _CLASSIFY_CACHE[question] = result
            return result

    # Keyword pre-filter: multi_hop patterns must be caught before builds_on patterns
    # because "modified BERT" would otherwise match builds_on.
    _MULTI_HOP_KEYWORDS = [
        "how did", "evolve", "chronologically", "from 2019", "from 20",
        "each make", "each change", "each discover", "what specific change",
        "what did each", "differ in how", "how do they differ",
        "compare", "comparison between", "contrast",
    ]
    if any(k in q_lower for k in _MULTI_HOP_KEYWORDS):
        result = {"type": "multi_hop", "target": None}
        _CLASSIFY_CACHE[question] = result
        return result

    if api_key is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"type": "factual", "target": None}

    user_msg = f'{_EXAMPLES}\n\nNow classify this query:\n"{question}"'

    for attempt in range(2):
        try:
            with httpx.Client(timeout=15.0, verify=False) as client:
                r = client.post(
                    _GROQ_URL,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": 60,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                qtype = parsed.get("type", "factual")
                target = parsed.get("target")
                if target in ("null", "", "none", "None"):
                    target = None
                result = {"type": qtype, "target": target}
                _CLASSIFY_CACHE[question] = result
                return result
            if r.status_code in (429, 413) and attempt == 0:
                import re
                m = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', r.text)
                wait = (float(m.group(1) or 0) * 60 + float(m.group(2)) + 2) if m else 15
                time.sleep(min(wait, 120))
                continue
        except Exception:
            pass

    return {"type": "factual", "target": None}
