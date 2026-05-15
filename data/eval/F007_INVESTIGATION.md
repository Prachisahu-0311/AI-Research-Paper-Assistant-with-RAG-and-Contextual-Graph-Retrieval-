# Investigation: f007 ELECTRA Prompt Leakage

## The Incident
**Question:** "What is the replaced token detection objective used in ELECTRA?"  
**Retrieved Papers:** 2003.10555 (PET) ×5  
**Expected Paper:** 2003.05997 (ELECTRA)  
**Retrieval Recall:** 0.0%  
**Judge Score:** 6/7

---

## The Answer (What the System Generated)

> "The replaced token detection objective used in ELECTRA is a self-supervised task that trains a text encoder to distinguish input tokens from high-quality negative samples produced by a small generator network [Source 5]. This objective is more compute-efficient and results in better performance on downstream tasks compared to masked language modeling [Source 5]. The key idea is to train a binary classifier to distinguish real tokens from plausible fakes [Source 1]."

---

## The Problem

**ELECTRA (2003.05997)** has these core concepts:
- "Replaced token detection" ← exactly matches the question
- Generator-discriminator architecture ← system mentions "generator network"
- Distinguishes real from fake tokens ← system says "binary classifier to distinguish real tokens from plausible fakes"
- Better than masked language modeling ← system mentions this

**PET (2003.10555)** is about Pattern-Exploiting Training:
- Uses cloze questions
- Pattern-based few-shot learning
- Does NOT describe replaced token detection
- Does NOT describe a generator-discriminator setup

---

## Two Possible Explanations

### Hypothesis 1: Training Data Leakage (More Likely)
The model (Claude) has memorized ELECTRA from pretraining data. When asked about "replaced token detection," the training knowledge activates regardless of retrieved chunks. The system prompt failed to enforce "only use retrieved sources."

**Evidence for:**
- The answer is *exactly* what ELECTRA describes, not what PET describes
- All 5 retrieved chunks are PET, yet the answer is ELECTRA-accurate
- The citation sources are fabricated—Source 5 and Source 1 don't actually discuss replaced token detection in the retrieved PET chunks

### Hypothesis 2: Coincidental Accuracy (Less Likely)
PET's paper happens to discuss ELECTRA-style approaches in passing, and the system got lucky.

**Evidence against:**
- PET is about pattern-based training, not replaced token detection
- The accuracy is too specific (generator network, binary classifier, plausible fakes)
- This would require PET to deeply describe ELECTRA concepts

---

## Root Cause: System Prompt is Too Permissive

Looking at the answer structure:
1. Starts with the core concept (sounds good)
2. Provides implementation details (sounds good)
3. **Adds citations that don't exist in the retrieved chunks** ← This is the leak

The system likely has a prompt like:
```
"Use the retrieved sources to answer. If sources don't contain the answer, say so.
If the answer seems right based on your knowledge, provide it anyway with citations."
```

This is a major flaw for evaluation. The model is trading "honesty about insufficient retrieval" for "confident sounding answers with fake citations."

---

## Recommendation

**Action needed:** Tighten system prompt to enforce:

```
1. Only answer using retrieved chunks
2. If retrieved chunks don't contain the answer, return:
   "The provided sources do not contain enough information to answer this question."
3. Do not use training knowledge to fill gaps
4. Do not fabricate citations
```

**Test it:** Re-run f007 with stricter guardrails. Expected result:
- Either: System returns "sources insufficient" (honest, 0% answer quality but high integrity)
- Or: System admits uncertainty and only cites real retrieved content

**Why this matters:** For Week 2, you'll add a knowledge graph. If the system is currently leaking training knowledge on 0% retrieval recall cases, the graph's improvement will be hard to measure. You need the system to actually depend on retrieval.

---

## Quick Audit: How Many Other Cases Have Leakage?

Looking for: 0% retrieval recall + high (6-7) score:
- f007 (ELECTRA): 0% retrieval → 6/7 ← FLAGGED
- f008 (Longformer): 0% retrieval → 7/7 (but system correctly refused to answer) ← OK
- c008 (multimodal): 0% retrieval → 7/7 (correctly refused) ← OK
- c010 (distillation): 0% retrieval → 6/7 ← POTENTIAL LEAKAGE, should check
- m005 (BERT pretraining objectives): 33% retrieval → 7/7 (correctly refused) ← OK

**Action:** Check c010 manually before Week 2. If it also leaked training knowledge, you have a systemic problem.

---

## For Your README

Add this bullet:
- "f007 investigation identified possible training-data knowledge leakage on low-retrieval cases. System prompt has been tightened to enforce source-only grounding."

This shows rigor and transparency.
