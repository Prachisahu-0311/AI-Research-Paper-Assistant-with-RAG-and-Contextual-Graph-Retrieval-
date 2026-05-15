# Prompt for Claude Code: Apply Spot-Check Corrections

Copy and paste this into Claude Code:

---

## TASK: Apply Evaluation Spot-Check Corrections

I need you to fix my Graph RAG evaluation based on a manual spot-check audit. Here's what to do:

### Step 1: Update Results Scores
Open the file: `c:\Users\prach\medmatch-ai\graphrag-project\data\eval\results_plain_rag.json`

Find these 5 questions and LOWER their `total_score`:
- **c001** (efficient alternatives to attention): Change from 6 → 5
- **c003** (vision transformers): Change from 6 → 5  
- **c004** (long sequences): Change from 6 → 3
- **c005** (sparse attention): Change from 6 → 4
- **c009** (softmax alternatives): Change from 6 → 4

**Why:** Manual review showed these were overscored. The judge gave too many points for incomplete/partially wrong answers.

### Step 2: Audit c010 for Leakage
Look at the answer for **c010** (distillation and compression):
- It has retrieval_recall: 0.0 but scored 6/7
- Read the answer text and tell me: Does it sound like it came from the sources, or like the system made it up from knowledge it already knows?

Report: Is this answer likely hallucinated/leaked training knowledge? (yes/no + brief reason)

### Step 3: Create Corrected Metrics Report
Calculate and report these NEW numbers:

1. **Average connection question score (type: "connection"):**
   - OLD: 6.0/7
   - NEW: (take the corrected scores and average)

2. **Overall RAG answer quality:**
   - OLD: 5.97/7
   - NEW: (recalculate with all corrected scores)

### Step 4: Generate Updated README Section
Write a new section for the README that says:

```
## Evaluation Methodology (Honest Version)

### Baseline Plain RAG Results
- **Connection query retrieval recall:** 38% (manually verified)
- **Connection query answer quality:** ~5.0/7 (LLM judge used, but spot-checked for reliability)
  - Note: Initial judge scoring showed ~1 point inflation on connection questions. Manual review of 5 representative questions (c001, c003, c004, c005, c009) identified systematic over-scoring. Scores corrected to reflect honest assessment.
- **Factual query accuracy:** 80% (judge reliable on straightforward questions)
- **Out-of-scope refusal:** 100% (perfect)
- **Overall answer quality:** ~5.2/7

### Evaluation Concern Identified & Fixed
During quality audit, discovered LLM judge was:
1. Marking citations as correct even when facts were wrong (judge didn't verify claims)
2. Noting errors in reasoning but not penalizing them in final score

**Action taken:** Corrected scores downward by 1-3 points on connection questions to reflect honest assessment.

### Week 2 Goal
Improve connection query retrieval recall from 38% → 65%+ using knowledge graph augmentation.
```

### Step 5: Summary Output
Create a file `c:\Users\prach\medmatch-ai\graphrag-project\data\eval\CORRECTIONS_APPLIED.md` that includes:
- List of corrected scores (before/after)
- New average scores
- c010 audit result
- Updated README section

---

**When done, show me:**
1. The updated `results_plain_rag.json` with new scores
2. The `CORRECTIONS_APPLIED.md` file
3. Your assessment of c010 (is it hallucinated?)

---

## Context Files (for your reference)
- Analysis: `c:\Users\prach\medmatch-ai\graphrag-project\data\eval\SPOT_CHECK_ANALYSIS.md`
- f007 investigation: `c:\Users\prach\medmatch-ai\graphrag-project\data\eval\F007_INVESTIGATION.md`
- Full plan: `c:\Users\prach\medmatch-ai\graphrag-project\data\eval\NEXT_STEPS_AND_TARGETS.md`

These have all the details if you need them.
