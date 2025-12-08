# Early-layer saliency relation classification (quick summary)

- **Data**: `saliency_datasets/early_layers` — 24 predicate classes, each with `sample_*.pt` containing `saliency_maps` (3×32×32), `cross_attention_maps` (3×32×32), `concepts`, and `class_id`/`predicate` metadata.
- **Model**: WideResNet-28-8 (variants of first conv for different channel counts); AdamW (lr=1e-3, wd=1e-4), batch size 128, epochs 5, dropout 0.3, no normalization/smoothing/augmentation. Late fusion uses two WRN branches (saliency and cross) with averaged logits.
- **Fusion modes**: `saliency`, `cross`, `concat` (saliency+cross), `triple` (saliency+cross+product), `diff` (saliency+(cross−saliency)), `late` (two-branch).
- **Outputs**: checkpoints and logs in `relation-analysis/scripts/runs/`; aggregated metrics in `runs/experiment_results.json`.

## Results (val acc after 5 epochs)

| Mode    | Val acc |
|---------|---------|
| saliency | 0.553 |
| cross    | 0.340 |
| concat   | 0.437 |
| triple   | 0.480 |
| diff     | 0.443 |
| late     | **0.593** |

Checkpoint files:
- `runs/wrn_mode-saliency_norm-none_smooth-0_abl-none.pt`
- `runs/wrn_mode-cross_norm-none_smooth-0_abl-none.pt`
- `runs/wrn_mode-concat_norm-none_smooth-0_abl-none.pt`
- `runs/wrn_mode-triple_norm-none_smooth-0_abl-none.pt`
- `runs/wrn_mode-diff_norm-none_smooth-0_abl-none.pt`
- `runs/wrn_mode-late_norm-none_smooth-0_abl-none.pt`

Run command (from `relation-analysis/scripts`):
```
nohup python run_classifier_experiments.py > runs/experiments_full.log 2>&1 &
```
