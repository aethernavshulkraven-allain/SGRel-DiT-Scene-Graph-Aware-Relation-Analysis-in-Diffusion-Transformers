# Graph-Conditioned LoRA Evaluation Guide

## Quick Start

### 1. Find Your Checkpoint
```bash
cd /home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers/relation-analysis

# List checkpoints
ls -lh checkpoints/graph_lora_diffusion_token/

# Should see: checkpoint_epoch_1.pt, checkpoint_epoch_2.pt, etc.
```

### 2. Run Inference
```bash
cd scripts

# Use the latest checkpoint (replace epoch number)
python test_graph_lora_inference.py \
    --checkpoint ../checkpoints/graph_lora_diffusion_token/checkpoint_epoch_5.pt \
    --device cuda:3 \
    --height 512 \
    --width 512 \
    --output-dir ../inference_results

# For faster generation (smaller images):
python test_graph_lora_inference.py \
    --checkpoint ../checkpoints/graph_lora_diffusion_token/checkpoint_epoch_5.pt \
    --device cuda:3 \
    --height 256 \
    --width 256 \
    --num-steps 4 \
    --output-dir ../inference_results
```

### 3. View Results

**Option A: Using Jupyter Notebook**
```bash
cd /home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers/relation-analysis

# Open the evaluation notebook
jupyter notebook evaluate_graph_lora.ipynb

# Run all cells to see visual comparisons
```

**Option B: View Images Directly**
```bash
cd inference_results
ls *.png

# Images generated:
# - person_riding_bike.png
# - person_on_bike.png
# - person_next_to_bike.png
# - dog_wearing_hat.png
# - dog_next_to_hat.png
# - person_holding_cup.png
# - person_drinking_from_cup.png
```

## What to Look For

### Key Evaluation Criteria:

1. **Spatial Differences**
   - **riding** vs **on** vs **next to**: Different object positions?
   - **wearing**: Object on/attached to subject?
   - **holding**: Object in hand/grasp?

2. **Action vs Static**
   - **riding**: Dynamic, motion implied?
   - **drinking from**: Active consumption?
   - **on**: Static placement?

3. **Graph Conditioning Evidence**
   - Same prompt + different graphs → different images?
   - Consistent rendering of same relationship across objects?

### Expected Results (if training worked):

✓ **Success**: 
- "person riding bike" shows person actively on bike (dynamic)
- "person next to bike" shows separation between person and bike
- "dog wearing hat" shows hat on dog's head
- "dog next to hat" shows hat beside dog

✗ **Failure**:
- All images look the same regardless of graph
- Spatial arrangements don't match relationships
- Graphs ignored, only text prompt matters

## Troubleshooting

### If images look identical:
```bash
# Try training longer or with more samples
python train_flux_graph_lora_diffusion.py \
    --graph-mode token \
    --epochs 10 \
    --max-train-samples -1 \
    --device cuda:3 \
    --no-cpu-offload
```

### If OOM during inference:
```bash
# Use smaller images
python test_graph_lora_inference.py \
    --checkpoint ../checkpoints/graph_lora_diffusion_token/checkpoint_epoch_5.pt \
    --height 256 \
    --width 256 \
    --device cuda:3
```

### Compare with baseline (no graph):
```bash
# Generate without graph conditioning to see difference
# (Would need to modify script to skip graph encoding)
```

## Next Steps

1. **Quantitative Evaluation**: Measure spatial layout differences
2. **User Study**: Human judgment on relationship accuracy
3. **Fine-tuning**: More epochs, more data, different relationships
4. **Ablation**: Try different LoRA ranks, block ranges, learning rates

## Files Created

- `test_graph_lora_inference.py`: Inference script
- `evaluate_graph_lora.ipynb`: Visual comparison notebook
- `inference_results/`: Generated images
- `checkpoints/graph_lora_diffusion_token/`: Trained LoRA weights
