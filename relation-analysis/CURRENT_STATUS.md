# Current Training & Evaluation Status

**Date**: December 15, 2025, 03:15 AM

---

## Active Processes

### 1. Training (GPU 3)
- **Status**: Running ✅
- **Checkpoint**: `run_20epochs`
- **Progress**: Epoch 4/20 (~20% complete)
- **Batch Size**: 2 (you changed it)
- **Config**: 
  - Height/Width: 256×256
  - LoRA rank: 16, blocks 7-13
  - No ranking loss (lambda=0.0)
  - No saliency weighting (alpha=0.0)
- **GPU Memory**: ~35 GB
- **Speed**: ~2.5 it/s
- **ETA**: ~1.5-2 hours total (started ~03:13 AM, finish ~05:00 AM)
- **Log**: `../outputs/graph_flux_lora_diffusion/run_20epochs/training.log`

### 2. Evaluation (GPU 2)
- **Status**: Running ✅  
- **Checkpoint**: `run_noranking_10epochs/best_graph_lora_token.pt`
- **Test Run**: 10 samples, 1 seed (quick test)
- **Metrics**: CLIP + LPIPS + MSE
- **Output**: `../outputs/eval_test_10samples/`
- **Log**: `eval_test.log`

---

## Evaluation Script Created

**Location**: `evaluate_graph_lora_comprehensive.py`

**Features**:
1. **Image Generation**
   - Generates with positive graph (g+) and negative graph (g-)
   - Uses same seeds for fair comparison
   - Saves all generated images

2. **CLIP Text Alignment** (Option 1)
   - Scores: positive image vs positive/negative prompts
   - Scores: negative image vs positive/negative prompts  
   - Metric: Does g+ match correct prompt better than wrong prompt?
   
3. **Perceptual Similarity** (Graph Sensitivity)
   - LPIPS: Perceptual distance between g+ and g- images
   - MSE: Pixel-level difference
   - Higher distance = model responds more to graphs
   
4. **Saved Outputs**:
   - `evaluation_results.json`: All metrics
   - `images/sample_XXXX_pos.png`: Positive graph images
   - `images/sample_XXXX_neg.png`: Negative graph images

---

## Usage

### Full Evaluation on 160 Validation Samples:
```bash
python evaluate_graph_lora_comprehensive.py \
  --checkpoint ../outputs/graph_flux_lora_diffusion/run_20epochs/best_graph_lora_token.pt \
  --device cuda:1 \
  --dtype bfloat16 \
  --max-samples 160 \
  --seeds 0,1,2 \
  --height 256 \
  --width 256 \
  --num-steps 4 \
  --output-dir ../outputs/eval_20epochs_full \
  --no-spatial  # Skip spatial detection for now
```

### Monitor Both:
```bash
./monitor_both.sh  # Shows training + evaluation status + GPU usage
```

---

## Key Metrics to Report

From the evaluation results:

### 1. CLIP Alignment
- `clip_pos_margin`: How much better g+ matches correct vs wrong prompt
- `clip_alignment_acc`: % of samples where g+ prefers correct prompt
- **Claim**: "Images generated with correct graphs achieve X% higher CLIP similarity with relationship descriptions"

### 2. Graph Sensitivity  
- `lpips_mean`: Perceptual distance between g+ and g-
- `mse_mean`: Pixel-level difference
- **Claim**: "Positive and negative graphs produce perceptually distinct images (LPIPS=X, significantly above threshold of Y)"

### 3. Win Rate
- Manual: Count how many times g+ looks better than g-
- **Claim**: "In X% of cases, positive graph produces more accurate spatial relationships"

---

## Next Steps (After Training Completes)

1. **Run full evaluation** on 20-epoch checkpoint (160 samples, 3 seeds)
   - ETA: ~10-15 minutes
   
2. **Compare checkpoints**:
   - Baseline (no graph conditioning)
   - 10 epochs
   - 20 epochs
   
3. **Generate paper figures**:
   - Side-by-side comparisons (g+ vs g-)
   - Qualitative examples showing spatial relationships
   - Metrics table

4. **Optional**: Train 10 more epochs if results are weak

---

## Files Created/Modified

1. ✅ `evaluate_graph_lora_comprehensive.py` - Main evaluation script
2. ✅ `monitor_both.sh` - Status monitoring
3. 🔄 `run_20epochs/training.log` - Training in progress
4. 🔄 `eval_test_10samples/` - Test evaluation running

---

## Timeline Summary

| Time | Event |
|------|-------|
| 01:54-02:11 | 10-epoch training (completed) |
| 02:36-03:09 | Failed ranking loss attempts |
| 03:13 | Started 20-epoch training (batch_size=2) |
| 03:15 | Started test evaluation (10 samples) |
| ~05:00 | **Expected**: 20-epoch training complete |
| ~05:15 | **Plan**: Run full evaluation (160 samples) |
| ~05:30 | **Deliverable**: Complete metrics + comparison |

---

## Monitoring Commands

```bash
# Training progress
tail -f ../outputs/graph_flux_lora_diffusion/run_20epochs/training.log

# Evaluation progress  
tail -f eval_test.log

# Both + GPU
watch -n 10 './monitor_both.sh'

# Check if done
ls -lh ../outputs/graph_flux_lora_diffusion/run_20epochs/
ls -lh ../outputs/eval_test_10samples/
```
