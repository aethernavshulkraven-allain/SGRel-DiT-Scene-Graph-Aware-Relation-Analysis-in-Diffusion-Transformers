# Project Structure

## Overview
Reorganized structure for SGRel-DiT project - Scene Graph-Aware Relation Analysis in Diffusion Transformers.

## Directory Layout

```
SGRel-DiT/
├── external/              # External dependencies
│   ├── ConceptAttention/  # ConceptAttention implementation
│   ├── ConceptAttention_/ # ConceptAttention fork
│   └── SGDiff/           # SGDiff baseline
│
├── src/                   # Core source code
│   ├── graph/            # Graph encoder modules
│   ├── flux/             # FLUX model integration
│   ├── concept-attention/ # Concept attention utilities
│   └── utils/            # Shared utilities
│
├── scripts/              # Executable scripts
│   ├── train/           # Training scripts
│   │   └── train_flux_graph_lora_diffusion.py
│   ├── evaluate/        # Evaluation scripts
│   │   ├── evaluate_graph_lora_comprehensive.py
│   │   └── eval_graph_lora_full_sampling.py
│   ├── data_prep/       # Data preparation
│   │   └── make_vg_quickwin_split.py
│   └── inference/       # Inference scripts
│
├── data/                 # Data files
│   ├── raw/             # Raw VG data
│   ├── processed/       # Preprocessed data
│   └── splits/          # Train/test splits
│
├── outputs/              # Generated outputs
│   ├── checkpoints/     # Model checkpoints
│   │   └── graph_flux_lora_diffusion/
│   ├── evaluations/     # Evaluation results
│   │   ├── eval_20epochs_full_with_spatial/
│   │   └── eval_test_10samples/
│   ├── logs/            # Training logs
│   └── generated_images/ # Sample generations
│
├── notebooks/            # Jupyter notebooks
│   ├── archive/         # Old exploration notebooks
│   ├── datagenerator.ipynb
│   ├── graph_exp.ipynb
│   └── sd3_example.ipynb
│
├── docs/                 # Documentation
│   ├── plan_lora.md     # LoRA training plan
│   ├── proj.md          # Project overview
│   └── results/         # Results and analysis
│
├── configs/              # Configuration files
│   ├── training/        # Training configs
│   └── evaluation/      # Evaluation configs
│
├── tests/                # Unit tests
│
└── experiments/          # Experiment tracking

## Old Structure
- relation-analysis/  # Legacy folder (being phased out)
- diffusers/          # Local diffusers fork (not moved)
```

## Key Locations

### Training
- **Main script**: `scripts/train/train_flux_graph_lora_diffusion.py`
- **Checkpoints**: `outputs/checkpoints/graph_flux_lora_diffusion/`
- **Config**: Training parameters in script args

### Evaluation  
- **Main script**: `scripts/evaluate/evaluate_graph_lora_comprehensive.py`
- **Results**: `outputs/evaluations/`
- **Metrics**: CLIP, LPIPS, MSE, spatial reasoning

### Data
- **Splits**: `data/splits/` (vg_quickwin_train/test.jsonl)
- **Raw data**: `data/raw/` (vocabularies, synsets)

### Models
- **Graph encoder**: `src/graph/sgdiff_encoder.py`
- **FLUX patches**: `src/flux/`
- **Source modules**: `src/`

## Migration Notes
- All files preserved (no deletions)
- Import paths may need updates
- Legacy `relation-analysis/` folder kept for reference
- External repos in `external/` remain unchanged
