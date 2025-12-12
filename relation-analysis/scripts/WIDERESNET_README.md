# WideResNet Training for Saliency-Based Relation Classification

This repository contains scripts for training WideResNet models on saliency and cross-attention maps for relation classification.

## Overview

- **Dataset**: 24 relation classes, 700 samples per class (16,800 total)
- **Input**: Saliency maps and cross-attention maps (3x32x32 each)
- **Model**: WideResNet-28-10 (customizable)
- **Splits**: Stratified train/val/test splits (70%/15%/15% by default)

## Features

### Multiple Fusion Strategies

1. **concat** - Concatenate saliency and attention maps (6 channels)
2. **saliency_only** - Use only saliency maps (3 channels)
3. **attention_only** - Use only cross-attention maps (3 channels)
4. **add** - Element-wise addition of maps (3 channels)
5. **multiply** - Element-wise multiplication of maps (3 channels)
6. **max** - Element-wise maximum of maps (3 channels)
7. **weighted** - Weighted sum (0.6*saliency + 0.4*attention, 3 channels)

### Output for Each Run

- Training logs with timestamps
- Train/Val loss vs epochs curve
- Train/Val accuracy vs epochs curve
- Confusion matrix on test set
- Classification report
- Best model checkpoint
- Final model checkpoint
- Training history JSON
- Results summary JSON

## Quick Start

### Single Experiment

Run training with a specific fusion mode:

```bash
python3 train_wideresnet_saliency.py \
    --data_dir saliency_datasets/early_layers \
    --fusion_mode concat \
    --epochs 100 \
    --batch_size 64
```

### All Fusion Modes

Run experiments with all fusion strategies:

```bash
chmod +x run_all_fusion_experiments.sh
./run_all_fusion_experiments.sh
```

This will train 7 models sequentially, one for each fusion mode.

### Compare Results

After training, compare all experiments:

```bash
python3 compare_fusion_results.py --output_dir wideresnet_experiments
```

## Command-Line Arguments

### Data Arguments
- `--data_dir`: Path to dataset directory (default: `saliency_datasets/early_layers`)
- `--output_dir`: Output directory for results (default: `wideresnet_experiments`)
- `--test_size`: Proportion for test set (default: 0.15)
- `--val_size`: Proportion for validation set (default: 0.15)

### Fusion Arguments
- `--fusion_mode`: Fusion strategy (choices: concat, saliency_only, attention_only, add, multiply, max, weighted)

### Model Arguments
- `--depth`: WideResNet depth (default: 28)
- `--widen_factor`: WideResNet widen factor (default: 10)
- `--dropout`: Dropout rate (default: 0.3)

### Training Arguments
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Initial learning rate (default: 0.1)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--seed`: Random seed (default: 42)

## Examples

### Train with different architectures:

```bash
# WideResNet-28-10 (default, ~36M parameters)
python3 train_wideresnet_saliency.py --depth 28 --widen_factor 10

# WideResNet-40-10 (larger model)
python3 train_wideresnet_saliency.py --depth 40 --widen_factor 10

# WideResNet-28-2 (smaller model)
python3 train_wideresnet_saliency.py --depth 28 --widen_factor 2
```

### Train with custom hyperparameters:

```bash
python3 train_wideresnet_saliency.py \
    --fusion_mode weighted \
    --epochs 150 \
    --batch_size 128 \
    --lr 0.05 \
    --dropout 0.4
```

### Train on different layer datasets:

```bash
# Early layers
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/early_layers

# Middle layers
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/middle_layers

# Late layers
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/late_layers
```

## Output Structure

```
wideresnet_experiments/
├── concat_28x10_20251212_143022/
│   ├── training.log                    # Complete training log
│   ├── training_curves.png             # Loss and accuracy curves
│   ├── confusion_matrix.png            # Test set confusion matrix
│   ├── classification_report.txt       # Per-class metrics
│   ├── best_model.pth                  # Best model checkpoint
│   ├── final_model.pth                 # Final epoch checkpoint
│   ├── checkpoint_epoch_10.pth         # Periodic checkpoints
│   ├── checkpoint_epoch_20.pth
│   ├── history.json                    # Training history
│   ├── results.json                    # Summary results
│   └── class_names.json                # Class label mapping
├── saliency_only_28x10_20251212_145033/
│   └── ...
├── comparison_table.csv                # Compare all experiments
├── fusion_comparison.png               # Comparison visualizations
└── all_training_curves.png             # All curves overlaid
```

## Monitoring Training

During training, you'll see:
- Real-time progress bars for each epoch
- Training loss per batch
- Epoch-level train/val metrics
- Best model updates
- Final test set evaluation

Example output:
```
Epoch 15/100
Learning rate: 0.095000
Training: 100%|████████| 184/184 [00:45<00:00,  4.05it/s, loss=0.523]
Validation: 100%|████████| 33/33 [00:03<00:00,  9.12it/s]
Train Loss: 0.4892, Train Acc: 0.8421
Val Loss: 0.5234, Val Acc: 0.8156
Saved best model with val_acc: 0.8156
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm pandas
```

## Dataset Format

Expected structure:
```
saliency_datasets/early_layers/
├── class_00_above/
│   ├── sample_0000.pt
│   ├── sample_0001.pt
│   └── ...
├── class_01_around_near/
│   └── ...
└── ...
```

Each .pt file contains:
- `saliency_maps`: torch.Tensor [3, 32, 32]
- `cross_attention_maps`: torch.Tensor [3, 32, 32]
- `class_id`: int
- Other metadata...

## Tips

1. **GPU Memory**: If you run out of memory, reduce `--batch_size`
2. **Training Time**: Each epoch takes ~1-2 minutes on a modern GPU
3. **Best Fusion**: Try all modes - performance varies by task
4. **Overfitting**: If train acc >> val acc, increase `--dropout` or `--weight_decay`
5. **Convergence**: Most models converge within 50-100 epochs

## Citation

If you use this code, please cite WideResNet:
```
@inproceedings{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={BMVC},
  year={2016}
}
```
