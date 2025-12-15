# docs/

Documentation, research plans, and analysis.

## Files

### plan_lora.md
**Purpose**: Complete LoRA training plan and methodology.

**Contents**:
- Theoretical foundation for graph-conditioned FLUX training
- Loss function design (L_gen, L_rel_rank, L_gen_rel)
- LoRA configuration (blocks 7-12, attention projections)
- Negative graph construction strategies
- Hyperparameter recommendations
- Evaluation metrics and success criteria

**Key sections**:
- Section 1: Training without full sampling (teacher-forced)
- Section 2: Trainable vs frozen components
- Section 3: Three-term loss objective
- Section 4: Negative graph design (directional swap + hard negatives)
- Section 7: Evaluation protocol (accuracy, margin, win rate)

**Status**: Reference document for implementation

**Current deviation**: 20-epoch training used pure generative loss (no L_rel_rank), resulted in failed semantic alignment (49% CLIP accuracy)

---

### plan_crit.md
**Purpose**: Critical analysis and alternative approaches (if exists).

**Contents**: TBD

---

### plan.md
**Purpose**: General project plan and milestones (if exists).

**Contents**: TBD

---

### proj.md
**Purpose**: Project overview and high-level description.

**Contents**: Likely includes:
- Research goals and motivation
- Technical approach summary
- Dataset description
- Expected outcomes

---

### hf_token.md
**Purpose**: Hugging Face authentication notes.

**Contents**:
- HF token for downloading models
- Instructions for setting up authentication
- Model access notes (FLUX.1-schnell, etc.)

**Security**: Should NOT contain actual token in plaintext (use environment variables)

---

### DATASET_README.md
**Purpose**: Dataset documentation and usage.

**Contents**:
- VG dataset structure
- Annotation format
- Predicate vocabulary
- Data statistics
- Usage examples

---

### results/
**Purpose**: Analysis documents and result summaries.

**Expected contents**:
- Evaluation result summaries
- Comparison tables (10-epoch vs 20-epoch)
- Metric interpretation guides
- Visual result galleries
- Failure analysis

**Example document**: `EVAL_RESULTS_20EPOCHS.md`
```markdown
# 20-Epoch Evaluation Results

## Metrics
- CLIP alignment: 49.4% (FAILED)
- MSE: 8613 ± 5267 (graph sensitivity detected)

## Interpretation
Pure generative loss insufficient for semantic relationship learning.
Need contrastive ranking loss (L_rel_rank) as per plan_lora.md.

## Next Steps
1. Implement L_rel_rank with frozen classifier
2. Retrain with contrastive supervision
3. Target: >80% CLIP alignment
```

---

## Document Workflow

**Research planning**:
1. Literature review → methodology notes
2. Design decisions → `plan_*.md` documents
3. Implementation based on plans

**Experiment tracking**:
1. Run training/evaluation
2. Document results in `results/`
3. Analysis and iteration

**Publication preparation**:
1. Consolidate key results
2. Generate figures and tables
3. Write paper sections from docs

---

## Writing Guidelines

**Plans** (plan_*.md):
- Detailed technical specifications
- Mathematical formulations
- Implementation guidance
- Reference citations

**Results** (results/):
- Quantitative metrics
- Qualitative observations
- Comparison analysis
- Lessons learned

**README files**:
- Quick reference
- Usage examples
- File descriptions
- Directory navigation

---

## Status

**Complete**:
- `plan_lora.md`: Full LoRA training methodology
- `DATASET_README.md`: VG dataset documentation

**In progress**:
- `results/`: Need to document 20-epoch failure analysis
- Comparison 10-epoch vs 20-epoch

**TODO**:
- Architecture diagrams
- Loss curve plots
- Visual result galleries
- Final paper sections
