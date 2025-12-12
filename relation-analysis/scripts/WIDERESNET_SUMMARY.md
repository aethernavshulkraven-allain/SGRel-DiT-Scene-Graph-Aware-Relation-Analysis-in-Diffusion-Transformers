# WideResNet Training Suite - File Summary

## Created Files

### 1. Main Training Script
**File:** `train_wideresnet_saliency.py`
- Complete PyTorch training script for WideResNet
- Implements 7 different fusion strategies for saliency and cross-attention maps
- Stratified train/val/test splits (70%/15%/15%)
- Full logging, checkpointing, and visualization
- Saves training curves, confusion matrix, classification report

**Key Features:**
- Multiple fusion modes: concat, saliency_only, attention_only, add, multiply, max, weighted
- WideResNet architecture (customizable depth and widen factor)
- Cosine annealing learning rate schedule
- Best model checkpointing based on validation accuracy
- Comprehensive metrics and visualizations

### 2. Batch Experiment Script
**File:** `run_all_fusion_experiments.sh`
- Bash script to run experiments with all 7 fusion modes
- Sequential execution of training runs
- Uses consistent hyperparameters across all runs

**Usage:**
```bash
./run_all_fusion_experiments.sh
```

### 3. Results Comparison Script
**File:** `compare_fusion_results.py`
- Compares results across all experiments
- Generates comparison tables and plots
- Identifies best performing fusion mode
- Creates overlay plots of all training curves

**Usage:**
```bash
python3 compare_fusion_results.py --output_dir wideresnet_experiments
```

**Outputs:**
- `comparison_table.csv` - Tabular comparison of all experiments
- `fusion_comparison.png` - Multi-panel comparison visualization
- `all_training_curves.png` - All training curves overlaid

### 4. System Test Script
**File:** `test_wideresnet_setup.py`
- Verifies dataset loading
- Tests all fusion modes
- Tests model creation with different configurations
- Tests DataLoader functionality

**Usage:**
```bash
python3 test_wideresnet_setup.py
```

### 5. Quick Demo Script
**File:** `quick_demo.sh`
- Quick 5-epoch training run for testing
- Verifies entire pipeline works correctly

**Usage:**
```bash
./quick_demo.sh
```

### 6. Documentation
**File:** `WIDERESNET_README.md`
- Comprehensive usage guide
- Command-line argument reference
- Examples for different use cases
- Tips and best practices

## Quick Start Guide

### Step 1: Verify Setup
```bash
python3 test_wideresnet_setup.py
```

Expected output: All tests pass ✓

### Step 2: Quick Demo (Optional)
```bash
./quick_demo.sh
```

Runs 5 epochs to verify everything works (~5-10 minutes)

### Step 3: Full Training - Single Fusion Mode
```bash
python3 train_wideresnet_saliency.py \
    --data_dir saliency_datasets/early_layers \
    --fusion_mode concat \
    --epochs 100 \
    --batch_size 64
```

Expected time: ~2-3 hours on modern GPU

### Step 4: Full Training - All Fusion Modes
```bash
./run_all_fusion_experiments.sh
```

Expected time: ~14-21 hours (7 models × 2-3 hours each)

### Step 5: Compare Results
```bash
python3 compare_fusion_results.py
```

## Output Structure

After running experiments, you'll have:

```
wideresnet_experiments/
├── concat_28x10_TIMESTAMP/
│   ├── training.log                 # Complete training log with timestamps
│   ├── training_curves.png          # Loss and accuracy vs epochs
│   ├── confusion_matrix.png         # 24x24 confusion matrix
│   ├── classification_report.txt    # Per-class precision, recall, F1
│   ├── best_model.pth              # Model with best validation accuracy
│   ├── final_model.pth             # Final epoch model
│   ├── checkpoint_epoch_10.pth     # Periodic checkpoints
│   ├── history.json                # Epoch-by-epoch metrics
│   ├── results.json                # Summary: test_acc, val_acc, etc.
│   └── class_names.json            # Label to class name mapping
├── saliency_only_28x10_TIMESTAMP/
│   └── [same structure]
├── attention_only_28x10_TIMESTAMP/
│   └── [same structure]
├── add_28x10_TIMESTAMP/
│   └── [same structure]
├── multiply_28x10_TIMESTAMP/
│   └── [same structure]
├── max_28x10_TIMESTAMP/
│   └── [same structure]
├── weighted_28x10_TIMESTAMP/
│   └── [same structure]
├── comparison_table.csv            # All experiments compared
├── fusion_comparison.png           # Visual comparison
└── all_training_curves.png         # All curves overlaid
```

## Key Features

### Fusion Strategies Implemented

1. **concat** (6 channels)
   - Concatenates saliency and attention along channel dimension
   - Model input: [batch, 6, 32, 32]
   
2. **saliency_only** (3 channels)
   - Uses only saliency maps
   - Model input: [batch, 3, 32, 32]
   
3. **attention_only** (3 channels)
   - Uses only cross-attention maps
   - Model input: [batch, 3, 32, 32]
   
4. **add** (3 channels)
   - Element-wise addition: saliency + attention
   - Model input: [batch, 3, 32, 32]
   
5. **multiply** (3 channels)
   - Element-wise multiplication: saliency * attention
   - Model input: [batch, 3, 32, 32]
   
6. **max** (3 channels)
   - Element-wise maximum: max(saliency, attention)
   - Model input: [batch, 3, 32, 32]
   
7. **weighted** (3 channels)
   - Weighted sum: 0.6*saliency + 0.4*attention
   - Model input: [batch, 3, 32, 32]

### Model Architecture

**WideResNet-28-10** (default)
- Depth: 28 layers
- Widen factor: 10
- Parameters: ~36.5M
- Dropout: 0.3
- Input: 32×32 (3 or 6 channels depending on fusion mode)
- Output: 24 classes (relation types)

### Training Configuration

- **Optimizer:** SGD with Nesterov momentum
- **Learning Rate:** 0.1 (initial) with cosine annealing
- **Batch Size:** 64
- **Epochs:** 100
- **Data Split:** 70% train, 15% val, 15% test (stratified)
- **Augmentation:** None (can be added in dataset class)

### Logging and Monitoring

Each run logs:
- Epoch-level train/val loss and accuracy
- Learning rate at each epoch
- Best model checkpoints
- Periodic checkpoints (every 10 epochs)
- Final test set evaluation

### Visualizations

1. **Training Curves** (per experiment)
   - Train/Val loss vs epochs
   - Train/Val accuracy vs epochs
   
2. **Confusion Matrix** (per experiment)
   - 24×24 heatmap
   - Shows per-class performance
   
3. **Comparison Plots** (across experiments)
   - Bar chart of test accuracies
   - Scatter plot: val_acc vs test_acc
   - Individual training curves
   - Overlaid training curves

## Dataset Information

- **Total Samples:** 16,800 (.pt files)
- **Classes:** 24 relations
- **Samples per Class:** 700
- **Split:**
  - Train: 11,760 samples (70%)
  - Val: 2,520 samples (15%)
  - Test: 2,520 samples (15%)
- **Input Format:** Each .pt file contains:
  - `saliency_maps`: [3, 32, 32] torch.Tensor
  - `cross_attention_maps`: [3, 32, 32] torch.Tensor
  - `class_id`: int (0-23)
  - Other metadata fields

### Relation Classes

0. above
1. around_near
2. behind
3. below
4. carrying
5. drinking
6. eating
7. hanging from
8. holding
9. in
10. in front of
11. left of
12. looking at
13. on
14. playing with
15. pulling
16. pushing
17. riding
18. right of
19. sitting on
20. standing on
21. touching
22. using
23. wearing

## Command Examples

### Basic Training
```bash
# Default configuration
python3 train_wideresnet_saliency.py

# Specific fusion mode
python3 train_wideresnet_saliency.py --fusion_mode saliency_only

# Custom epochs and batch size
python3 train_wideresnet_saliency.py --epochs 150 --batch_size 128
```

### Architecture Variants
```bash
# Smaller model (fewer parameters)
python3 train_wideresnet_saliency.py --depth 28 --widen_factor 2

# Larger model (more capacity)
python3 train_wideresnet_saliency.py --depth 40 --widen_factor 10
```

### Different Layer Datasets
```bash
# Train on early layers (default)
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/early_layers

# Train on middle layers
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/middle_layers

# Train on late layers
python3 train_wideresnet_saliency.py --data_dir saliency_datasets/late_layers
```

### Hyperparameter Tuning
```bash
# Lower learning rate
python3 train_wideresnet_saliency.py --lr 0.05

# Higher dropout
python3 train_wideresnet_saliency.py --dropout 0.4

# Stronger regularization
python3 train_wideresnet_saliency.py --weight_decay 1e-3
```

## Expected Results

Based on typical WideResNet performance on similar tasks:

- **Training Accuracy:** 85-95%
- **Validation Accuracy:** 75-85%
- **Test Accuracy:** 75-85%

**Note:** Results will vary by fusion mode. The concat mode typically performs well as it preserves all information.

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python3 train_wideresnet_saliency.py --batch_size 32

# Or use a smaller model
python3 train_wideresnet_saliency.py --widen_factor 2
```

### Overfitting (Train acc >> Val acc)
```bash
# Increase dropout
python3 train_wideresnet_saliency.py --dropout 0.5

# Increase weight decay
python3 train_wideresnet_saliency.py --weight_decay 1e-3
```

### Slow Training
```bash
# Increase number of workers
python3 train_wideresnet_saliency.py --num_workers 8

# Enable AMP (mixed precision) - modify script to add
```

## Next Steps

After training:

1. **Analyze Results:**
   ```bash
   python3 compare_fusion_results.py
   ```

2. **Inspect Best Model:**
   - Load checkpoint: `best_model.pth`
   - Check confusion matrix for class-specific issues
   - Review classification report for detailed metrics

3. **Further Experiments:**
   - Try different layer datasets (early/middle/late)
   - Experiment with data augmentation
   - Try ensemble methods
   - Fine-tune hyperparameters for best fusion mode

4. **Deployment:**
   - Use best model for inference
   - Export to ONNX for production
   - Integrate into larger pipeline

## Files Summary

| File | Purpose | Usage |
|------|---------|-------|
| `train_wideresnet_saliency.py` | Main training script | Primary training tool |
| `run_all_fusion_experiments.sh` | Batch experiments | Run all fusion modes |
| `compare_fusion_results.py` | Compare results | Analyze experiment results |
| `test_wideresnet_setup.py` | System verification | Test before training |
| `quick_demo.sh` | Quick test run | 5-epoch demo |
| `WIDERESNET_README.md` | Documentation | Usage guide |

All files are ready to use! Start with `test_wideresnet_setup.py` to verify everything works.
