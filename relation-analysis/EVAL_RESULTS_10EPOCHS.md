# Evaluation Results Summary (10-Epoch Checkpoint)

**Test Set**: 10 samples (quick test)  
**Checkpoint**: `run_noranking_10epochs/best_graph_lora_token.pt`  
**Date**: December 15, 2025, 03:20 AM

---

## Key Findings

### ✅ Graph Conditioning IS Working
- **MSE between g+ and g-**: 6460 (very high)
- **Images are perceptually different** when using different graphs
- Model responds to scene graph inputs

### ⚠️ But Alignment is Weak
- **CLIP alignment accuracy**: 60% (should be >80% for strong results)
- **Positive margin**: 1.01 (marginal preference for correct prompt)
- **Negative margin**: 0.51 (confused)

---

## What This Means

**Good News**:
- Graph conditioning mechanism works
- Model generates different outputs for different relationships
- Not ignoring the graph input

**Challenge**:
- Weak semantic alignment with text descriptions
- May not be capturing correct spatial relationships
- Could be changing style/details rather than relationships

---

## Next Steps

1. **Wait for 20-epoch training** (~1.5 hours remaining)
   - More training may improve alignment
   
2. **Run full evaluation** (160 samples, 3 seeds)
   - More robust statistics
   - Better confidence in metrics
   
3. **Visual inspection** (critical!)
   - Look at generated images manually
   - See if "riding" vs "next to" actually differ spatially
   - CLIP might be wrong if visual relationships are subtle

4. **Compare checkpoints**:
   - Baseline (no graph)
   - 10 epochs
   - 20 epochs
   - See if longer training helps

---

## Detailed Metrics

```json
{
  "mse_mean": 6459.9667,
  "mse_std": 3323.1442,
  "clip_pos_correct": 25.6160,
  "clip_pos_wrong": 24.6056,
  "clip_pos_margin": 1.0104,
  "clip_neg_correct": 25.6219,
  "clip_neg_wrong": 25.1154,
  "clip_neg_margin": 0.5066,
  "clip_alignment_acc": 0.6000
}
```

---

## Interpretation Guide

| Metric | Value | Ideal | Status |
|--------|-------|-------|--------|
| MSE mean | 6460 | >1000 | ✅ GOOD |
| CLIP pos margin | 1.01 | >3.0 | ⚠️ WEAK |
| CLIP alignment acc | 60% | >80% | ⚠️ WEAK |

**Verdict**: Model is graph-aware but needs either:
- More training (try 20 epochs)
- Higher resolution (512x512)
- Stronger conditioning (more LoRA blocks)
- Better negative sampling

---

## Current Training Status

- **Epoch**: 5/20 (25% complete)
- **ETA**: ~1.5 hours
- **Next checkpoint**: Will have 2× more training

Let's see if doubling training time improves CLIP alignment from 60% → 80%+
