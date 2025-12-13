# Early vs Middle layer WideResNet experiments

- **Data**: Scene-graph relation saliency dumps. Early-layer set at `saliency_datasets/early_layers`; middle-layer set introduced by Sujal Suri (paths referenced in configs inside each run folder).
- **Model/protocol**: WideResNet-28-10, dropout 0.3, 50 epochs, batch 64, SGD (lr=0.01, momentum=0.9, wd=5e-4), splits 70/15/15. Fusion variants: saliency-only, attention-only, concat, add, multiply, weighted, max.
- **Artifacts**:
  - Early runs: `relation-analysis/scripts/wideresnet_experiments/*/results.json` (+ logs/plots).
  - Middle runs: `relation-analysis/scripts/wideresnet_experiments_middle_layers/*/results.json` (+ logs/plots).
  - Aggregated results (below) derived from the `results.json` files.

## Validation vs Test accuracy

| Layer set | Fusion mode      | Val acc | Test acc | Run dir |
|-----------|------------------|---------|----------|---------|
| early  | concat          | 0.8091 | 0.8119 | wideresnet_experiments/concat_28x10_20251212_050534 |
| early  | max             | 0.7286 | 0.7321 | wideresnet_experiments/max_28x10_20251212_061010 |
| early  | add             | 0.7103 | 0.7206 | wideresnet_experiments/add_28x10_20251212_054429 |
| early  | multiply        | 0.7151 | 0.7099 | wideresnet_experiments/multiply_28x10_20251212_055718 |
| early  | saliency_only   | 0.7135 | 0.7067 | wideresnet_experiments/saliency_only_28x10_20251212_051828 |
| early  | weighted        | 0.7099 | 0.7206 | wideresnet_experiments/weighted_28x10_20251212_062304 |
| early  | attention_only  | 0.6663 | 0.6730 | wideresnet_experiments/attention_only_28x10_20251212_053130 |
| middle | concat          | 0.7909 | 0.7988 | wideresnet_experiments_middle_layers/concat_28x10_20251212_053640 |
| middle | attention_only  | 0.7480 | 0.7575 | wideresnet_experiments_middle_layers/attention_only_28x10_20251212_060219 |
| middle | max             | 0.7381 | 0.7381 | wideresnet_experiments_middle_layers/max_28x10_20251212_064035 |
| middle | add             | 0.7369 | 0.7476 | wideresnet_experiments_middle_layers/add_28x10_20251212_061504 |
| middle | weighted        | 0.7222 | 0.7131 | wideresnet_experiments_middle_layers/weighted_28x10_20251212_065326 |
| middle | multiply        | 0.7234 | 0.7329 | wideresnet_experiments_middle_layers/multiply_28x10_20251212_062751 |
| middle | saliency_only   | 0.6448 | 0.6425 | wideresnet_experiments_middle_layers/saliency_only_28x10_20251212_054930 |

## Takeaways
- **Best overall**: Early-layer concat slightly leads (test 0.8119). Middle-layer concat is close (0.7988) and best within the middle set.
- **Saliency-only vs attention-only**: Saliency-only outperforms attention-only on early layers; reversed on middle layers.
- **Simple fusions (add/multiply/weighted/max)** trail concat for both layer sets but remain competitive; middle layers generally narrow the gap.
- All runs used identical hyperparameters/splits; checkpoints, training logs, confusion matrices, and curves are stored in the respective run directories.
