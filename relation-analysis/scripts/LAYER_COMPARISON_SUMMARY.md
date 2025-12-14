# Early vs Middle layer WideResNet experiments

- **Data**: Scene-graph relation saliency dumps. Early-layer set at `saliency_datasets/early_layers`; middle-layer set introduced by Sujal Suri (paths referenced in configs inside each run folder).
- **Model/protocol**: WideResNet-28-10, dropout 0.3, 50 epochs, batch 64, SGD (lr=0.01, momentum=0.9, wd=5e-4), splits 70/15/15. Fusion variants: saliency-only, attention-only, concat, add, multiply, weighted, max.
- **Artifacts**:
  - Early runs: `relation-analysis/scripts/wideresnet_experiments/*/results.json` (+ logs/plots).
  - Middle runs: `relation-analysis/scripts/wideresnet_experiments_middle_layers/*/results.json` (+ logs/plots).
  - Aggregated results (below) derived from the `results.json` files.

## Validation accuracy (latest sweeps, 100 epochs, batch 128)

Early layers (runs/experiment_results.json):
| Arch      | Mode     | Val acc |
|-----------|----------|---------|
| wrn       | saliency | 0.704 |
| wrn       | cross    | 0.689 |
| wrn       | concat   | 0.798 |
| wrn       | triple   | 0.799 |
| wrn       | diff     | 0.797 |
| wrn       | late     | 0.827 |
| tiny_cnn  | saliency | 0.680 |
| tiny_cnn  | cross    | 0.688 |
| tiny_cnn  | concat   | 0.830 |
| tiny_cnn  | triple   | **0.837** |
| tiny_cnn  | diff     | 0.829 |
| mlp       | saliency | 0.604 |
| mlp       | cross    | 0.564 |
| mlp       | concat   | 0.733 |
| mlp       | triple   | 0.727 |
| mlp       | diff     | 0.749 |

Middle layers (runs/experiment_results_middle_full.json):
| Arch      | Mode     | Val acc |
|-----------|----------|---------|
| wrn       | saliency | 0.641 |
| wrn       | cross    | 0.761 |
| wrn       | concat   | 0.794 |
| wrn       | triple   | **0.800** |
| wrn       | diff     | 0.798 |
| wrn       | late     | 0.796 |
| tiny_cnn  | saliency | 0.578 |
| tiny_cnn  | cross    | 0.729 |
| tiny_cnn  | concat   | 0.787 |
| tiny_cnn  | triple   | 0.798 |
| tiny_cnn  | diff     | 0.795 |
| mlp       | saliency | 0.551 |
| mlp       | cross    | 0.636 |
| mlp       | concat   | 0.698 |
| mlp       | triple   | 0.705 |
| mlp       | diff     | 0.700 |

Legacy WRN-28x10 (50 epochs) with more fusions (add/multiply/weighted/max/attention) remain in `wideresnet_experiments/` and `wideresnet_experiments_middle_layers/` (see individual `results.json` for val/test splits).

## Takeaways
- Latest 100-epoch sweeps show tiny CNN triple best on early (0.837), WRN triple best on middle (0.800), WRN late strong on early (0.827), WRN concat competitive on both (≥0.794).
- MLP baselines are substantially weaker than conv models but provide a lower-bound reference.
- Older WRN-28x10 50-epoch runs (with add/multiply/weighted/max/attention) remain for comparison; concat still leads there. Check `wideresnet_experiments*/results.json` for test accuracies.
