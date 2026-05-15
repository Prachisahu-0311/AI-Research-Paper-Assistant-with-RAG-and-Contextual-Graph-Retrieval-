# Knowledge Graph Schema

This file defines the 5 allowed relationship types for the knowledge graph.
The entity extraction prompt will use this schema strictly — no other relationship
types will be extracted. Every triple must follow the format:
(subject_entity, relationship_type, object_entity)

---

## Relationship Types

### 1. addresses_problem
**What it captures:** A paper tackles a specific technical limitation or challenge.
**Subject:** A paper
**Object:** A problem or limitation (use snake_case noun phrases)

Real examples from our 20 papers:
- (Reformer, addresses_problem, quadratic_attention_complexity)
- (Linformer, addresses_problem, quadratic_attention_complexity)
- (Longformer, addresses_problem, quadratic_attention_complexity)
- (Big Bird, addresses_problem, quadratic_attention_complexity)
- (Performers, addresses_problem, quadratic_attention_complexity)
- (Transformer-XL, addresses_problem, limited_context_length)
- (DistilBERT, addresses_problem, large_model_size)
- (ALBERT, addresses_problem, large_model_size)
- (Switch Transformers, addresses_problem, training_efficiency)

**Why this matters for retrieval:**
Query "which papers address long sequence problems" → graph traversal finds all papers
with addresses_problem → quadratic_attention_complexity or limited_context_length.
Plain RAG misses 4 out of 5 of these papers (c004 recall was 20%).

---

### 2. builds_on
**What it captures:** A paper directly extends, improves, or is a successor to another paper.
**Subject:** The newer/extending paper
**Object:** The paper being extended

Real examples from our 20 papers:
- (RoBERTa, builds_on, BERT)
- (ALBERT, builds_on, BERT)
- (DistilBERT, builds_on, BERT)
- (XLNet, builds_on, BERT)
- (ELECTRA, builds_on, BERT)
- (Big Bird, builds_on, BERT)
- (Transformer-XL, builds_on, Attention Is All You Need)
- (Reformer, builds_on, Attention Is All You Need)
- (Universal Transformers, builds_on, Attention Is All You Need)

**Why this matters for retrieval:**
Query "which papers build on BERT" → graph traversal finds all papers with
builds_on → BERT. Plain RAG for c002 got 20% recall, missing ALBERT, ELECTRA, XLNet.

---

### 3. applies_to_domain
**What it captures:** A paper applies the Transformer architecture to a specific domain or task.
**Subject:** A paper
**Object:** A domain (use simple lowercase nouns: vision, language, speech, multimodal)

Real examples from our 20 papers:
- (ViT, applies_to_domain, vision)
- (CLIP, applies_to_domain, vision)
- (CLIP, applies_to_domain, multimodal)
- (BERT, applies_to_domain, language)
- (GPT-3, applies_to_domain, language)
- (T5, applies_to_domain, language)
- (Big Bird, applies_to_domain, language)
- (Big Bird, applies_to_domain, genomics)

**Why this matters for retrieval:**
Query "which papers apply transformers to vision" → graph finds ViT and CLIP.
Plain RAG for c003 and c008 missed CLIP entirely (0% recall).

---

### 4. uses_technique
**What it captures:** A paper uses a specific technical method as a core part of its approach.
**Subject:** A paper
**Object:** A technique (use snake_case noun phrases)

Real examples from our 20 papers:
- (DistilBERT, uses_technique, knowledge_distillation)
- (Switch Transformers, uses_technique, knowledge_distillation)
- (Switch Transformers, uses_technique, mixture_of_experts)
- (T5, uses_technique, mixture_of_experts)
- (Reformer, uses_technique, locality_sensitive_hashing)
- (BERT, uses_technique, masked_language_modeling)
- (ELECTRA, uses_technique, replaced_token_detection)
- (XLNet, uses_technique, permutation_language_modeling)
- (RoBERTa, uses_technique, masked_language_modeling)
- (ALBERT, uses_technique, factorized_embedding_parameterization)
- (CLIP, uses_technique, contrastive_learning)
- (Performers, uses_technique, random_feature_approximation)
- (Linformer, uses_technique, low_rank_approximation)

**Why this matters for retrieval:**
Query "which papers use distillation" → graph finds DistilBERT and Switch Transformers.
Plain RAG for c010 had 0% recall — DistilBERT not retrieved at all.

---

### 5. authored_by
**What it captures:** The first or most recognizable author of a paper.
**Subject:** A paper
**Object:** Author last name only (for normalization — no initials, no "et al.")

Real examples from our 20 papers:
- (Attention Is All You Need, authored_by, Vaswani)
- (BERT, authored_by, Devlin)
- (RoBERTa, authored_by, Liu)
- (ALBERT, authored_by, Lan)
- (DistilBERT, authored_by, Sanh)
- (XLNet, authored_by, Yang)
- (GPT-3, authored_by, Brown)
- (T5, authored_by, Raffel)
- (Transformer-XL, authored_by, Dai)
- (Reformer, authored_by, Kitaev)
- (Linformer, authored_by, Wang)
- (Longformer, authored_by, Beltagy)
- (Big Bird, authored_by, Zaheer)
- (Performers, authored_by, Choromanski)
- (Universal Transformers, authored_by, Dehghani)
- (ViT, authored_by, Dosovitskiy)
- (CLIP, authored_by, Radford)
- (Switch Transformers, authored_by, Fedus)
- (ELECTRA, authored_by, Clark)
- (PET, authored_by, Schick)

**Why this matters for retrieval:**
Enables author-based graph queries. Also helps the normalization step — we can
verify extracted paper entities against this known list.

---

## Extraction Rules (for the LLM extraction prompt — Week 2 Day 1)

These rules will be included verbatim in the extraction prompt:

1. Only extract triples using the 5 relationship types above. No other relationship types.
2. Subject must always be a paper name from the known paper list. Never use arXiv IDs as subjects.
3. Object must match the controlled vocabulary shown in the examples above. Do not invent new object values — map to the closest existing one.
4. One triple per line. Format: (Subject, relationship_type, Object)
5. If a relationship is not explicitly stated in the text, do not extract it. No inference.
6. authored_by: use last name only, no initials, no "et al."
7. addresses_problem and uses_technique objects must be snake_case (underscores, no spaces, all lowercase).
8. applies_to_domain objects must be single lowercase words only: vision, language, multimodal, speech, genomics.

---

## Known Paper Name Aliases (for normalization — Week 2 Day 2)

The LLM will sometimes refer to papers by different names. Map all of these to the canonical name:

| Alias seen in text | Canonical name |
|---|---|
| Transformer | Attention Is All You Need |
| original Transformer | Attention Is All You Need |
| Vaswani et al. | Attention Is All You Need |
| BERT-base | BERT |
| BERT-large | BERT |
| Devlin et al. | BERT |
| RoBERTa-base | RoBERTa |
| Liu et al. | RoBERTa |
| ViT-B | ViT |
| ViT-L | ViT |
| Vision Transformer | ViT |
| Dosovitskiy et al. | ViT |
| Switch Transformer | Switch Transformers |
| GPT3 | GPT-3 |
| Brown et al. | GPT-3 |

---

## What This Schema Does NOT Allow

Do not extract triples for:
- Paper-to-paper citation relationships (too noisy, not useful for our queries)
- Dataset names as subjects or objects
- Benchmark names (GLUE, SQuAD) as subjects
- Vague relationships like "related_to", "similar_to", "inspired_by"
- Any relationship type not in the list of 5 above
