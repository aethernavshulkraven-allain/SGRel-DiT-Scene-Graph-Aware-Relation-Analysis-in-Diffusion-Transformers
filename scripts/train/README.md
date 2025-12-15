# scripts/train/

Training scripts for graph-conditioned FLUX LoRA models.

## Files

### train_flux_graph_lora_diffusion.py
**Purpose**: Main training script for graph-conditioned FLUX.1-schnell LoRA fine-tuning.

**What it does**:
- Implements teacher-forced (single-timestep) training using rectified flow / flow matching
- Trains LoRA adapters on FLUX transformer blocks 7-12 (attention projections only)
- Integrates SGDiff graph encoder to condition image generation on scene graph triples
- Supports pure generative loss or optional contrastive negative-graph training
- Uses VG (Visual Genome) dataset with SGDiff-supported 16 canonical predicates
- Caches VAE latents and text embeddings for efficiency
- Saves checkpoints every N steps and best model based on validation loss

**Key features**:
- **Graph conditioning**: Injects graph embeddings (local + global) into FLUX transformer blocks
- **LoRA training**: Low-rank adaptation on attention weights (Q, K, V, output projections)
- **Configurable losses**: 
  - `L_gen`: Generative flow-matching loss
  - `L_rel_rank` (optional): Contrastive loss encouraging g+ > g- via frozen classifier
  - `L_gen_rel` (optional): Saliency-weighted generative loss
- **Predicate filtering**: Automatically filters to 16 SGDiff-supported predicates from 24 canonical
- **Training efficiency**: ~17 min/epoch on A6000, batch_size=2, 256x256 images

**Usage**:
```bash
python train_flux_graph_lora_diffusion.py \
  --train-jsonl ../data/splits/vg_quickwin_train.jsonl \
  --val-jsonl ../data/splits/vg_quickwin_test.jsonl \
  --output-dir ../outputs/checkpoints/my_run \
  --epochs 20 \
  --batch-size 2 \
  --lora-rank 16
```

**Outputs**:
- Checkpoints: `{output_dir}/step{N}.pt`, `final_graph_lora_token.pt`
- Training log: `{output_dir}/training.log`
- Config snapshot: `{output_dir}/config.json`

**Last run**:
- 20 epochs, pure generative loss (λ_rel_rank=0.0)
- Final val_loss: 0.4605 (improved from ~0.98)
- Checkpoint: `outputs/checkpoints/graph_flux_lora_diffusion/run_20epochs/`
