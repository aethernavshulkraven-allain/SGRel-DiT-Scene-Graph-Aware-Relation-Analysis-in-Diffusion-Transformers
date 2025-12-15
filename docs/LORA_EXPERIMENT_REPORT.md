# Graph-Conditioned FLUX LoRA Experiment Report

**Date**: December 2024  
**Model**: FLUX.1-schnell (12B parameters)  
**Task**: Scene graph-conditioned image generation  
**Method**: LoRA fine-tuning with graph conditioning

---

## Executive Summary

**Goal**: Fine-tune FLUX.1-schnell to generate images conditioned on scene graph triples (subject-predicate-object) using LoRA adapters.

**Approach**: Teacher-forced training with graph encoder injecting embeddings into transformer blocks 7-12.

**Result**: ⚠️ **Training converged but failed semantic evaluation**
- Training loss decreased successfully (0.98 → 0.46)
- Model learned to respond to graph changes (high MSE variance)
- **But**: CLIP alignment = 49.4% (random performance)
- **Conclusion**: Pure generative loss insufficient for relationship understanding

---

## 1. Methodology

### 1.1 Architecture

**Base Model**: FLUX.1-schnell
- 12B parameter diffusion transformer
- Rectified flow formulation (flow matching)
- Dual-stream architecture (19 double-stream blocks, 38 single-stream blocks)
- 4-step inference (schnell = fast variant)

**Graph Conditioning**:
- **Graph Encoder**: SGDiff CGIP (Conditional Graph Injection Pipeline)
  - Pretrained on Visual Genome
  - Converts (subject, predicate, object) → embeddings
  - Outputs: Local tokens (per-concept) + Global token (scene-level)
- **Injection mechanism**: Prepend graph tokens to transformer context
- **Target blocks**: Double-stream blocks 7-12 (mid-to-late layers)

**LoRA Configuration**:
- **Trainable**: LoRA adapters on attention projections (Q, K, V, output)
- **Rank**: 16
- **Alpha**: 16
- **Target blocks**: 7-12 only (6 out of 19 double-stream blocks)
- **Trainable parameters**: ~10-12M (0.1% of base model)
- **Frozen**: Base FLUX weights, text encoders, VAE, graph encoder

### 1.2 Training Setup

**Loss Function**: Pure generative loss (teacher-forced)
```
L = MSE(predicted_flow, target_flow)
```

Where:
- Sample timestep t ∈ [0.5, 1.0] (focus on low-noise region)
- Build noisy latent: z_t = (1-t) * z_0 + t * noise
- Predict flow/velocity: v_θ(z_t, t, prompt, graph)
- Target: noise - z_0

**Negative graph training**: Disabled (λ_rel_rank = 0.0)
- Initially planned contrastive loss L_rel_rank using frozen classifier
- Classifier found to be fundamentally flawed (trained on attention maps, not VAE latents)
- Decided to proceed with pure generative loss only

**No saliency weighting**: L_gen_rel disabled (α = 0.0)

### 1.3 Data

**Dataset**: Visual Genome (VG)
- **Train**: 800 samples (balanced across 16 predicates)
- **Val**: 160 samples (balanced across 16 predicates)
- **Image resolution**: 256×256
- **Predicates**: 16 SGDiff-supported (out of 24 canonical)
  ```
  Supported: above, around/near, behind, below, carrying, eating,
             hanging from, holding, in, in front of, looking at,
             on, riding, sitting on, standing on, wearing
  ```

**Caching**:
- VAE latents cached in memory (first epoch)
- Text embeddings cached per unique prompt
- Speeds up training ~2x after first epoch

---

## 2. Hyperparameters

### 2.1 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 20 | Sufficient for convergence |
| **Batch size** | 2 | Memory constraint (A6000 48GB) |
| **Learning rate** | 1e-4 | Standard LoRA LR |
| **Optimizer** | AdamW | Default for fine-tuning |
| **Weight decay** | 0.01 | Regularization |
| **Gradient clipping** | 1.0 | Stability |
| **Mixed precision** | bfloat16 | Speed + stability |

### 2.2 LoRA Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Rank** | 16 | Balance capacity/overfitting |
| **Alpha** | 16 | Scaling factor = rank |
| **Target modules** | attn.{to_q, to_k, to_v, to_out.0} | Attention only |
| **Target blocks** | 7-12 | Mid-to-late layers |
| **Dropout** | 0.0 | Not used |

### 2.3 Loss Weights

| Loss Term | Weight | Status |
|-----------|--------|--------|
| L_gen (generative) | 1.0 | ✅ Active |
| L_rel_rank (contrastive) | 0.0 | ❌ Disabled |
| L_gen_rel (saliency) | 0.0 | ❌ Disabled |

### 2.4 Inference Configuration

| Parameter | Value |
|-----------|-------|
| **Steps** | 4 (schnell default) |
| **Guidance scale** | 0.0 (no CFG) |
| **Resolution** | 256×256 |
| **Seeds** | 0, 1, 2 (for evaluation) |

---

## 3. Training Results

### 3.1 Loss Curves

**Validation Loss Progression**:
```
Epoch   | Val Loss | Notes
--------|----------|------------------
1       | 0.98     | Initial
5       | 0.72     | Rapid descent
10      | 0.52     | Plateau begins
15      | 0.48     | Slow improvement
20      | 0.46     | Final (converged)
```

**Training time**: ~17 minutes/epoch on A6000
- Total: ~5.7 hours for 20 epochs
- 8000 total steps (400 steps/epoch)

### 3.2 Observations

✅ **Training convergence**:
- Loss decreased smoothly (0.98 → 0.46)
- No instabilities or divergence
- Validation loss tracked training loss
- Model checkpoints saved successfully

✅ **Graph sensitivity**:
- MSE between g+ and g- images: 8613 ± 5267
- High variance indicates model responds to different graphs
- Not ignoring graph conditioning

❌ **Semantic alignment failure**:
- CLIP accuracy: 49.4% (random guess is 50%)
- Negative margin: -0.022 (wrong graphs score better!)
- Model makes different images but not correct relationships

---

## 4. Evaluation Results

### 4.1 Evaluation Protocol

**Samples**: 160 validation triples × 3 seeds = 480 total images

**For each triple (subject, predicate, object)**:
1. Create positive graph g+: ground truth relationship
2. Create negative graph g-: corrupted relationship
   - Directional swap: "dog riding horse" → "horse riding dog"
   - Predicate replacement: "dog holding ball" → "dog wearing ball"
3. Generate images with same seed: img_pos (g+), img_neg (g-)
4. Compute metrics comparing img_pos vs img_neg

### 4.2 Quantitative Results

**CLIP Text Alignment** (primary metric):
```json
{
  "clip_alignment_acc": 0.494,        // 49.4% - FAILED
  "clip_pos_margin": -0.022,          // Negative = wrong graph better
  "clip_neg_margin": 0.059,           // Inconsistent
  "clip_pos_correct": 12.97,          // img_pos with correct prompt
  "clip_pos_wrong": 12.99             // img_pos with wrong prompt (HIGHER!)
}
```

**Interpretation**:
- 49.4% accuracy = **random guessing**
- Negative margin = images generated with g+ actually align better with g- descriptions
- Model doesn't understand which relationship is correct

**Graph Sensitivity**:
```json
{
  "mse_mean": 8613.70,
  "mse_std": 5267.92
}
```

**Interpretation**:
- High MSE = g+ and g- produce different images ✅
- High std dev = strong variation across relationships ✅
- Model responds to graph changes (not ignoring conditioning)

**LPIPS** (not computed in final run):
- Expected: High perceptual distance between g+ and g-
- Would confirm visual differences beyond pixel-level

**Spatial Reasoning** (attempted but predicates unsupported):
- GroundingDINO object detection + bbox spatial reasoning
- Failed due to SGDiff vocab limitations ("hanging from" not in vocab)

### 4.3 Failure Analysis

**Why did CLIP alignment fail?**

1. **No contrastive supervision**:
   - Pure generative loss optimizes: "fit any z_0 given (z_t, t, prompt, graph)"
   - No signal for: "g+ should match better than g-"
   - Model learns to vary outputs but not which variation is correct

2. **Missing L_rel_rank**:
   - Original plan specified contrastive loss with frozen classifier
   - Would encourage: margin(ℓ_g+ - ℓ_g-) > 0
   - Dropped due to flawed classifier (attention maps vs VAE latents mismatch)

3. **Graph encoder limitations**:
   - SGDiff encoder frozen (pretrained on VG)
   - May not provide sufficiently discriminative embeddings for subtle relationship differences
   - No gradient signal to improve graph representations

4. **Prompt ambiguity**:
   - Text prompt includes relationship: "dog riding horse"
   - Graph conditioning may be redundant with text
   - Model relies on text, ignores graph embeddings
   - Need evaluation with neutral prompts: "dog and horse"

---

## 5. Key Findings

### 5.1 What Worked ✅

1. **Training stability**:
   - LoRA adapters trained smoothly without divergence
   - Loss converged as expected
   - No gradient explosions or NaN issues

2. **Graph injection mechanism**:
   - Graph tokens successfully injected into transformer
   - Model processes graph embeddings (evidenced by MSE variance)
   - No architectural issues

3. **Computational efficiency**:
   - ~17 min/epoch on A6000 (batch_size=2)
   - Teacher-forced training (single timestep) much faster than full sampling
   - Caching reduced memory and time

4. **Graph sensitivity**:
   - High MSE variance confirms model responds to graph changes
   - Different graphs → different images
   - Conditioning mechanism functional

### 5.2 What Failed ❌

1. **Semantic relationship learning**:
   - 49.4% CLIP accuracy = random performance
   - Model doesn't learn correct relationships
   - Pure generative loss insufficient

2. **Contrastive supervision absence**:
   - No L_rel_rank due to flawed classifier
   - No alternative contrastive loss implemented
   - Critical component from plan_lora.md missing

3. **Evaluation limitations**:
   - Spatial reasoning blocked by vocab limitations
   - No human evaluation of image quality
   - Limited diversity in test predicates

### 5.3 Lessons Learned 📚

1. **Generative loss alone is insufficient** for learning relationships:
   - Need explicit contrastive/ranking supervision
   - Model needs signal: "g+ better than g-"
   - Loss must encode preference, not just reconstruction

2. **CLIP evaluation reveals semantic failure** that loss curves hide:
   - Training loss decreased → assumed learning
   - CLIP evaluation exposed: model learned to vary, not to be correct
   - Need semantic metrics during training

3. **Frozen classifier approach has fundamental issues**:
   - Classifier trained on different data distribution (attention maps)
   - Evaluation on different modality (VAE latents)
   - Need end-to-end trainable alternatives

4. **Graph encoder quality matters**:
   - Frozen SGDiff encoder may not provide optimal representations
   - Fine-tuning graph encoder could improve results
   - Or train graph encoder jointly with LoRA

---

## 6. Comparison: 10-Epoch vs 20-Epoch

### Quick Test (10 epochs, 10 samples):
```
MSE: 6460
CLIP: 60% (weak)
Margin: 1.01
```

### Full Training (20 epochs, 160 samples):
```
MSE: 8613 (higher variance - more sensitivity)
CLIP: 49% (worse!)
Margin: -0.022 (negative - failed)
```

**Conclusion**: Doubling training time **degraded** performance
- More epochs without proper supervision → overfitting to text, ignoring graphs
- Need correct loss formulation, not more training

---

## 7. Next Steps & Recommendations

### 7.1 Immediate Fixes

1. **Implement proper contrastive loss**:
   - Option A: CLIP-based ranking
     ```python
     L_contrast = max(0, margin - CLIP(img_g+, prompt_g+) + CLIP(img_g+, prompt_g-))
     ```
   - Option B: Train lightweight classifier on VAE latents (not attention maps)
   - Option C: Pairwise ranking loss on generated images

2. **Add CLIP-based monitoring during training**:
   - Compute CLIP alignment every N steps
   - Early stopping based on semantic metrics
   - Detect failure before full 20 epochs

3. **Test with neutral prompts**:
   - Remove relationship from text: "a photo of dog and horse"
   - Force model to rely on graph conditioning only
   - Better isolate graph vs text contributions

### 7.2 Architecture Improvements

1. **Fine-tune graph encoder**:
   - Unfreeze SGDiff encoder
   - Train jointly with LoRA adapters
   - Learn better discriminative graph representations

2. **Stronger graph injection**:
   - Cross-attention between image tokens and graph tokens
   - Current: simple concatenation may be too weak
   - Add learnable fusion module

3. **More LoRA capacity**:
   - Increase rank: 16 → 32 or 64
   - Target more blocks: 7-12 → 4-15
   - Add LoRA to MLP layers, not just attention

### 7.3 Training Strategy

1. **Multi-stage training**:
   - Stage 1: Pure generative (5 epochs) - learn basic generation
   - Stage 2: Add contrastive loss (15 epochs) - learn relationships
   - Gradual curriculum

2. **Better negative sampling**:
   - Current: simple swap or random replacement
   - Improved: Hard negatives based on confusion matrix
   - Adversarial: Generate worst-case negatives

3. **Data augmentation**:
   - More training data (800 → 2000+ samples)
   - Better predicate balance
   - Include harder relationship examples

### 7.4 Evaluation Enhancements

1. **Human evaluation**:
   - MTurk study: "Which image matches the description?"
   - Gold standard for semantic correctness
   - CLIP may have biases

2. **Fine-grained metrics**:
   - Per-predicate accuracy (riding vs sitting on vs wearing)
   - Directional accuracy (above vs below)
   - Object identity preservation

3. **Qualitative analysis**:
   - Visual inspection of failure cases
   - Identify systematic errors
   - Understand what model learned instead

---

## 8. Conclusion

**Summary**: Successfully implemented and trained graph-conditioned FLUX LoRA, but **failed to achieve semantic relationship understanding** due to insufficient supervision signal.

**Root cause**: Pure generative loss (MSE on flow predictions) does not encode preference for correct vs incorrect relationships. Model learned to generate different images for different graphs but not semantically correct relationships.

**Validation of hypothesis**: Experiment confirms the necessity of contrastive/ranking losses (L_rel_rank) as originally specified in plan_lora.md. The decision to proceed without this component led to predictable failure.

**Path forward**: Implement CLIP-based or alternative contrastive supervision, retrain with proper loss formulation, and re-evaluate. The training infrastructure, graph injection mechanism, and LoRA configuration are sound - only the loss function needs correction.

**Positive outcomes**:
- Stable training pipeline established
- Graph conditioning mechanism validated (responds to graphs)
- Evaluation framework developed (CLIP, MSE, LPIPS ready)
- Clear understanding of failure mode and solution

**Research contribution**: Demonstrates that graph-conditioned fine-tuning requires explicit contrastive supervision beyond standard diffusion training objectives. Validates CLIP-based evaluation as critical metric for detecting semantic failures invisible in training loss curves.

---

## Appendix A: File Locations

**Training script**: `scripts/train/train_flux_graph_lora_diffusion.py`

**Evaluation script**: `scripts/evaluate/evaluate_graph_lora_comprehensive.py`

**Checkpoints**:
- 10-epoch: `outputs/checkpoints/graph_flux_lora_diffusion/run_noranking_10epochs/`
- 20-epoch: `outputs/checkpoints/graph_flux_lora_diffusion/run_20epochs/`

**Results**:
- Test (10 samples): `outputs/evaluations/eval_test_10samples/`
- Full (160 samples): `outputs/evaluations/eval_20epochs_full_with_spatial/`

**Training log**: `outputs/checkpoints/graph_flux_lora_diffusion/run_20epochs/training.log`

**Dataset**: `data/splits/vg_quickwin_train.jsonl`, `vg_quickwin_test.jsonl`

---

## Appendix B: Commands Reference

**Training**:
```bash
python scripts/train/train_flux_graph_lora_diffusion.py \
  --train-jsonl data/splits/vg_quickwin_train.jsonl \
  --val-jsonl data/splits/vg_quickwin_test.jsonl \
  --output-dir outputs/checkpoints/my_run \
  --epochs 20 \
  --batch-size 2 \
  --lora-rank 16 \
  --device cuda:3
```

**Evaluation**:
```bash
python scripts/evaluate/evaluate_graph_lora_comprehensive.py \
  --checkpoint outputs/checkpoints/run_20epochs/final_graph_lora_token.pt \
  --device cuda:2 \
  --max-samples 160 \
  --seeds 0,1,2 \
  --output-dir outputs/evaluations/my_eval
```

**Monitoring**:
```bash
# Training progress
tail -f outputs/checkpoints/my_run/training.log

# GPU usage
watch -n 1 nvidia-smi

# Results
cat outputs/evaluations/my_eval/evaluation_results.json | jq
```

---

**Last updated**: December 15, 2024  
**Authors**: Research Team  
**Status**: Experiment concluded, next iteration planned
