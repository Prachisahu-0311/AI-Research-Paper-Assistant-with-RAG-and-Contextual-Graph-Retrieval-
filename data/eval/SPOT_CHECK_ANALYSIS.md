# Spot-Check Analysis: Connection Question Scoring

## Executive Summary
Analyzed 5 connection questions with 6/7 scores. **All 5 are overscored by 1-2 points.**

---

## Question c001: Efficient Alternatives to Quadratic Attention
**Judge Score:** 6/7  
**Retrieval Recall:** 60% (3/5 expected papers found)  
**Judge Reasoning:** "mostly correct with minor errors"

### What the system said:
- Block sparse attention (Child et al. 2019)
- Sparse factorizations (Child et al. 2019, O(N√N))
- Reformer with LSH (Kitaev et al. 2020, O(N logN))
- Linear transformers (no constraints on Q,K, scales linearly)

### What was retrieved:
- 2002.12327 (likely RoBERTa/BERT related)
- 2003.05997 (likely efficiency paper)
- 2005.00743 (likely Longformer - correct!)
- 2006.16236 (likely Performer - correct!)
- 2006.16236

### What should have been retrieved:
2005.12872, 2005.00743, 2002.12327, 2001.04451, 2006.16236

### Manual Assessment:
**Honest Score: 5/7**
- ✓ Correctly identifies 3+ methods
- ✗ Missing Big Bird (2002.12327 retrieved but not mentioned)
- ✗ No discussion of tradeoffs between methods
- ⚠ Linear transformer claim is vague—doesn't ground in specific mechanism
- Judge missed: Claims sound coherent but under-explore the methods

---

## Question c003: Vision Transformers
**Judge Score:** 6/7  
**Retrieval Recall:** 50% (1/2 expected papers)  
**Judge Reasoning:** "mostly correct, lacks information about CLIP"

### What the system said:
- Vision Transformer (ViT) applies Transformer to vision
- Treats image as sequence of patches
- Uses standard Transformer encoder

### What was retrieved:
- 2010.11929 (ViT - correct!) ×5

### What should have been retrieved:
- 2010.11929 (ViT)
- 2103.00020 (CLIP)

### Manual Assessment:
**Honest Score: 5/7**
- ✓ Correct on ViT mechanism
- ✗ **CLIP was expected but completely missing** — answer is only 50% complete
- ✗ No discussion of multimodal approaches (CLIP)
- Judge's own reasoning says "lacks information about CLIP" but still gives 6/7
- Judge conflict: Noted missing half the answer but didn't penalize

---

## Question c004: Long Sequence Efficiency ⚠️ MOST PROBLEMATIC
**Judge Score:** 6/7  
**Retrieval Recall:** 20% (1/5 expected papers)  
**Judge Reasoning:** "mostly correct but incorrectly includes Performers... omits Longformer, Big Bird, Linformer"

### What the system said:
- Performers (linear attention, O(N) to O(N))
- Reformer (LSH, O(L^2) to O(L logL))
- Universal Transformers (input-length dependent steps)

### What was retrieved:
- 2006.16236 (Performer - correct!)
- 2003.05997 (not clearly long-sequence focused)
- 2001.04451 (Reformer - correct!)
- 2006.16236
- 1807.03819 (not clearly in scope)

### What should have been retrieved:
- 2005.00743 (Longformer) ← MISSING
- 2002.12327 (Big Bird) ← MISSING
- 2001.04451 (Reformer) ✓
- 2005.12872 (Linformer) ← MISSING
- 1901.02860 (Transformer-XL) ← MISSING

### Manual Assessment:
**Honest Score: 3-4/7**
- ✗ **CRITICAL: Performers don't primarily address long sequences.** Performers do linear attention but aren't positioned as long-sequence solvers.
- ✗ **CRITICAL: Universal Transformers wrong.** Doesn't address sequence length efficiency.
- ✓ Reformer mentioned correctly
- ✗ **Judge's own reasoning says these are wrong** ("incorrectly includes Performers"), yet scored 6/7
- ✗ Omitted 3 key papers (Longformer, Big Bird, Linformer)
- Judge failure: Explicitly noted errors in reasoning but didn't adjust score downward

**This is the smoking gun.** Judge contradicts itself.

---

## Question c005: Sparse Attention Patterns
**Judge Score:** 6/7  
**Retrieval Recall:** 33% (1/3 expected papers)  
**Judge Reasoning:** "mostly correct, lacks Reformer and Longformer, incorrectly attributes some info"

### What the system said:
- Block sparse attention (Child et al. 2019, local + strided)
- Learned locality (Sukhbaatar et al. 2019)
- Routing Transformer (sparse routing module, O(n^1.5 d))
- Sparse factorizations (Child et al. 2019, O(N√N))

### What was retrieved:
- 2003.05997 (sparse/efficient)
- 2002.12327 (BERT/RoBERTa)
- 2006.16236 (Performer)
- Multiple repeats

### What should have been retrieved:
- 2002.12327 (Big Bird) ✓
- 2005.00743 (Longformer) ✗
- 2001.04451 (Reformer) ✓

### Manual Assessment:
**Honest Score: 4-5/7**
- ✓ Correctly describes block sparse attention
- ✗ Mentions "Routing Transformer" which is confabulated—not clearly in corpus
- ✗ Only 33% paper recall
- ✓ Judge noted the issues but scored 6/7 anyway
- Judge pattern: Notes errors, doesn't penalize

---

## Question c009: Alternatives to Softmax in Attention
**Judge Score:** 6/7  
**Retrieval Recall:** 50% (1/2 expected papers)  
**Judge Reasoning:** "mostly correct, incorrectly includes papers and mentions papers not provided"

### What the system said:
- Performers: linear attention (replaces softmax with linear dot product)
- Reformer: memory-efficient but doesn't propose softmax alternative
- Mentions Shen et al. (2020) for linearized attention on object detection

### What was retrieved:
- 2006.16236 (Performer) ✓
- 1706.03762 (Transformer)
- 2003.05997
- 2001.04451 (Reformer)

### What should have been retrieved:
- 2005.12872 (Linformer - linear attention)
- 2006.16236 (Performer) ✓

### Manual Assessment:
**Honest Score: 4-5/7**
- ✓ Correctly identifies Performer as softmax alternative
- ✗ **Mentions Shen et al. (2020) which is NOT in the provided corpus**
- ✗ Cites papers not provided (external knowledge leakage?)
- ✗ Only 50% recall (missing Linformer)
- ✗ Contradicts itself (says Reformer doesn't propose softmax alternative, then discusses it)
- Judge noted: "incorrectly mentions papers not provided" but scored 6/7

---

## Pattern Summary

| Question | Judge Score | Honest Assessment | Inflation | Issue |
|----------|------------|------------------|-----------|-------|
| c001 | 6/7 | 5/7 | -1 | Vague grounding, incomplete exploration |
| c003 | 6/7 | 5/7 | -1 | Missing 50% of expected papers (CLIP) |
| c004 | 6/7 | 3-4/7 | -2 to -3 | **Judge contradicts itself; includes wrong methods** |
| c005 | 6/7 | 4-5/7 | -1 to -2 | Possible confabulation, judge notes errors but doesn't penalize |
| c009 | 6/7 | 4-5/7 | -1 to -2 | Cites papers outside corpus, judge notes but doesn't penalize |

**Average inflation: 1.2-1.8 points on connection questions**

---

## Key Finding: Judge's Contradiction Pattern

The judge **repeatedly notes errors in its own reasoning but fails to adjust the score downward**:

- c003: "lacks information about CLIP" → still 6/7
- c004: "incorrectly includes Performers... omits Longformer, Big Bird, Linformer" → still 6/7
- c005: "lacks Reformer and Longformer, incorrectly attributes" → still 6/7
- c009: "incorrectly includes papers... mentions papers not provided" → still 6/7

This suggests the judge is using citation density and coherence as proxies for correctness, not actual factual accuracy.

---

## Recommendation for README

**Before:** "Connection queries: 38% retrieval recall, 6.0/7 answer quality"

**After:** "Connection queries: 38% retrieval recall, ~5.0/7 answer quality (LLM judge spot-checked on 5 questions; systematic ~1-point inflation noted on multi-hop/connection reasoning)"

This is honest and defensible.
