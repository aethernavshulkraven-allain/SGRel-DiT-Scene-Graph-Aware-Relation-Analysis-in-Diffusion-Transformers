# Saliency Map Dataset Generator

Multi-GPU pipeline for generating saliency map datasets from ConceptAttention for relationship classification.

## Overview

Generates 3 separate datasets with saliency maps from different transformer layer groups:
- **Early layers**: [0, 1, 2, 3, 4, 5, 6] - Abstract features
- **Middle layers**: [7, 8, 9, 10, 11, 12] - Intermediate features  
- **Late layers**: [13, 14, 15, 16, 17, 18] - Semantic features

For **24 relationship classes** with **2000 samples each** (48,000 samples per dataset).

## Dataset Structure

```
saliency_datasets/
├── early_layers/
│   ├── class_00_on/
│   │   ├── class_info.json
│   │   ├── sample_0000.pt
│   │   ├── sample_0001.pt
│   │   └── ...
│   ├── class_01_in/
│   └── ...
├── middle_layers/
│   └── ...
└── late_layers/
    └── ...
```

### Sample Data Format

Each `sample_XXXX.pt` file contains:
```python
{
    "saliency_maps": Tensor(num_concepts, 32, 32),      # Output space saliency
    "cross_attention_maps": Tensor(num_concepts, 32, 32), # Cross-attention maps
    "prompt": str,                                        # e.g., "shade on sidewalk"
    "concepts": List[str],                               # e.g., ["shade", "on", "sidewalk"]
    "class_id": int,                                     # 0-23
    "predicate": str,                                    # e.g., "on"
    "sample_id": int,                                    # Unique ID
    "layer_config": str,                                 # "early_layers", etc.
    "layer_indices": List[int],                          # [0,1,2,3,4,5,6]
    "metadata": dict                                     # Original VG metadata
}
```

## Usage

### 1. Generate All Datasets (Recommended)

Uses all 4 GPUs in parallel:

```bash
cd /home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers

python generate_saliency_dataset.py \
  --layer-config all \
  --num-gpus 4 \
  --samples-per-class 2000 \
  --output-dir /home/namanb/SBILab/CSE677/Project/saliency_datasets
```

### 2. Generate Single Dataset

Generate only one layer configuration:

```bash
# Early layers only
python generate_saliency_dataset.py --layer-config early_layers

# Middle layers only
python generate_saliency_dataset.py --layer-config middle_layers

# Late layers only
python generate_saliency_dataset.py --layer-config late_layers
```

### 3. Custom Configuration

```bash
python generate_saliency_dataset.py \
  --jsonl-path /path/to/your/data.jsonl \
  --output-dir /path/to/output \
  --layer-config early_layers \
  --num-gpus 2 \
  --samples-per-class 1000
```

### 4. Verify Generated Dataset

```bash
python dataset_loader.py \
  --dataset-path saliency_datasets/early_layers \
  --action verify
```

### 5. View Dataset Statistics

```bash
python dataset_loader.py \
  --dataset-path saliency_datasets/early_layers \
  --action stats
```

### 6. Load and Inspect Sample

```bash
python dataset_loader.py \
  --dataset-path saliency_datasets/early_layers \
  --action load \
  --sample-path saliency_datasets/early_layers/class_00_on/sample_0000.pt
```

## Using the Dataset in PyTorch

```python
from dataset_loader import SaliencyMapDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = SaliencyMapDataset("saliency_datasets/early_layers")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in dataloader:
    saliency_maps = batch["saliency_maps"]  # (batch, num_concepts, 32, 32)
    labels = batch["label"]                  # (batch,)
    predicates = batch["predicate"]          # List of strings
    
    # Your training code here
    ...
```

## Filter by Specific Relationships

```python
# Only load specific relationships
dataset = SaliencyMapDataset(
    "saliency_datasets/early_layers",
    predicates=["on", "in", "wearing", "holding"]  # Only these 4 classes
)
```

## Performance

- **4 GPUs**: ~500 samples per GPU per class
- **512x512 images**: 32x32 patch grid
- **Storage**: ~100-200MB per 1000 samples (depends on compression)

### Expected Generation Time

With 4 GPUs (A100/V100):
- Per sample: ~2-5 seconds (including generation + save)
- Per class (2000 samples): ~40-80 minutes
- Full dataset (24 classes): ~16-32 hours per layer config
- All 3 datasets: ~48-96 hours total

**Optimization**: Generation runs in parallel across all GPUs, so actual wall-clock time is divided by number of GPUs.

## The 24 Relationship Classes

Based on Visual Genome dataset:

**Note**: Some classes have fewer than 2000 samples in the source data (e.g., "right of": 401, "pushing": 430). For these classes, the script **oversamples with shuffling** to reach 2000 samples, ensuring all classes have equal representation in the training set.

| Class ID | Predicate      | Training Samples |
|----------|----------------|------------------|
| 0        | on             | 2000            |
| 1        | in             | 2000            |
| 2        | wearing        | 2000            |
| 3        | around/near    | 2000            |
| 4        | above          | 2000            |
| 5        | behind         | 2000            |
| 6        | holding        | 2000            |
| 7        | below          | 2000            |
| 8        | sitting on     | 2000            |
| 9        | hanging from   | 2000            |
| 10       | in front of    | 2000            |
| 11       | standing on    | 2000            |
| 12       | riding         | 2000            |
| 13       | looking at     | 2000            |
| 14       | carrying       | 2000            |
| 15       | eating         | 2000            |
| 16       | using          | 2000            |
| 17       | pulling        | 2000            |
| 18       | touching       | 2000            |
| 19       | playing with   | 2000            |
| 20       | drinking       | 2000            |
| 21       | left of        | 2000            |
| 22       | pushing        | 2000            |
| 23       | right of       | 2000            |

## Monitoring Progress

The script uses `tqdm` progress bars for each GPU. You'll see:

```
[GPU 0] on              : 100%|██████████| 500/500 [42:13<00:00, 5.06s/it]
[GPU 1] on              : 100%|██████████| 500/500 [42:15<00:00, 5.07s/it]
[GPU 2] on              : 100%|██████████| 500/500 [42:10<00:00, 5.06s/it]
[GPU 3] on              : 100%|██████████| 500/500 [42:18<00:00, 5.08s/it]
```

## Troubleshooting

### Out of Memory

Reduce batch size or use smaller images:
- Current: 512x512 → 32x32 patches
- Alternative: 256x256 → 16x16 patches (modify in script)

### GPU Not Available

Check GPU availability:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### Dataset Verification Failed

Common issues:
- Incomplete generation (interrupted process)
- Disk space issues
- Permission errors

Re-run generation for specific layer config only.

## Next Steps

After generating datasets, you can:

1. **Train a classifier** to predict relationships from saliency maps
2. **Compare layer effectiveness** (early vs middle vs late)
3. **Analyze which concepts are most important** for each relationship
4. **Visualize saliency patterns** across different relationships

## Notes

- Each GPU processes different samples simultaneously (no overlap)
- Samples use unique seeds for reproducibility
- Images are 512x512 with 4 denoising steps (Flux-Schnell)
- Saliency maps averaged across all 4 timesteps
