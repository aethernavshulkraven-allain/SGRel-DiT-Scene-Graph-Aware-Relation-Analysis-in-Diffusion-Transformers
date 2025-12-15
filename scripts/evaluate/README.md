# scripts/evaluate/

Evaluation scripts for assessing graph-conditioned FLUX models.

## Files

### evaluate_graph_lora_comprehensive.py
**Purpose**: Comprehensive multi-metric evaluation of graph-conditioned FLUX LoRA checkpoints.

**What it does**:
- Generates images with positive (g+) and negative (g-) scene graphs for fair comparison
- Computes multiple evaluation metrics to assess relationship understanding
- Saves all generated images and quantitative results
- Handles unsupported predicates gracefully (skips samples not in SGDiff vocab)

**Metrics computed**:
1. **CLIP Text Alignment**: 
   - Measures semantic consistency between generated images and text descriptions
   - Computes `CLIP(img_pos, prompt_pos)` vs `CLIP(img_pos, prompt_neg)`
   - Reports alignment accuracy (% samples where correct prompt scores higher)
   - **Target**: >80% accuracy indicates strong semantic understanding

2. **MSE (Mean Squared Error)**:
   - Pixel-level difference between g+ and g- generated images
   - High MSE → model responds to graph changes
   - High std dev → model differentiates between relationship types
   - **Target**: High variance shows graph sensitivity

3. **LPIPS (Learned Perceptual Image Patch Similarity)**:
   - Perceptual distance using AlexNet features
   - Captures semantic differences beyond pixel changes
   - **Target**: High LPIPS indicates perceptually distinct generations

4. **Spatial Reasoning (optional, --no-spatial to disable)**:
   - Uses GroundingDINO to detect objects in generated images
   - Computes spatial relationships from bounding boxes
   - Compares predicted vs ground truth spatial predicates
   - **Target**: >70% spatial accuracy for geometric relationships

**Usage**:
```bash
# Full evaluation with all metrics
python evaluate_graph_lora_comprehensive.py \
  --checkpoint ../../outputs/checkpoints/run_20epochs/final_graph_lora_token.pt \
  --device cuda:2 \
  --max-samples 160 \
  --seeds 0,1,2 \
  --output-dir ../../outputs/evaluations/eval_20epochs_full

# Quick test (10 samples, no spatial)
python evaluate_graph_lora_comprehensive.py \
  --checkpoint ../../outputs/checkpoints/run_10epochs/best_graph_lora_token.pt \
  --max-samples 10 \
  --seeds 0 \
  --no-spatial \
  --output-dir ../../outputs/evaluations/eval_test
```

**Outputs**:
- `evaluation_results.json`: All quantitative metrics
- `images/`: Generated images organized by sample and seed
- Log file: Progress and diagnostics

**Latest results** (20-epoch model, 160 samples × 3 seeds):
- **CLIP alignment**: 49.4% ⚠️ (random performance, failed)
- **MSE**: 8613 ± 5267 ✅ (high variance = graph sensitivity)
- **Interpretation**: Model responds to graphs but doesn't learn semantically correct relationships

---

### eval_graph_lora_full_sampling.py
**Purpose**: Legacy evaluation script using full multi-step sampling.

**What it does**:
- Runs complete FLUX inference pipeline (4-step sampling)
- Generates images with both g+ and g- conditioning
- Simpler evaluation focused on generation quality

**Difference from comprehensive script**:
- Fewer metrics (no CLIP, LPIPS, spatial reasoning)
- Used for earlier experiments
- Kept for reference and comparison

**Status**: Superseded by `evaluate_graph_lora_comprehensive.py`

---

## Evaluation Workflow

1. **Train model**: Use `scripts/train/train_flux_graph_lora_diffusion.py`
2. **Quick test**: Run evaluation on 10 samples to verify checkpoint loads
3. **Full evaluation**: Run comprehensive evaluation (160 samples × 3 seeds)
4. **Analyze results**: Check JSON metrics and inspect generated images
5. **Visual inspection**: Compare g+ vs g- images for relationship accuracy

## Interpretation Guide

**Good results**:
- CLIP alignment >80%
- High MSE variance (different relationships look different)
- Spatial accuracy >70% for geometric predicates

**Bad results** (current 20-epoch model):
- CLIP alignment ~50% (random guess)
- Negative margin (wrong graphs score better)
- Indicates pure generative loss insufficient

**Next steps**: Need contrastive ranking loss (L_rel_rank) as specified in `docs/plan_lora.md`
