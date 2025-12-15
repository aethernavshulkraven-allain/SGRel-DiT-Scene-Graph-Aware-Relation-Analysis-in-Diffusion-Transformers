# scripts/data_prep/

Data preparation and preprocessing scripts for VG (Visual Genome) dataset.

## Files

### make_vg_quickwin_split.py
**Purpose**: Create balanced train/test splits from Visual Genome for rapid LoRA experimentation.

**What it does**:
- Extracts scene graph triples (subject-predicate-object) from VG h5 files
- Filters to SGDiff-supported predicates (16 out of 24 canonical)
- Creates balanced splits with equal representation across relationship types
- Generates JSONL files with image paths, prompts, and metadata
- Produces training summary with predicate distribution statistics

**Key features**:
- **Predicate filtering**: Only includes relationships supported by SGDiff encoder:
  ```
  Supported: above, around/near, behind, below, carrying, eating, 
             hanging from, holding, in, in front of, looking at, 
             on, riding, sitting on, standing on, wearing
  
  Unsupported (excluded): drinking, left of, playing with, pulling, 
                           pushing, right of, touching, using
  ```
- **Balanced sampling**: Ensures each predicate has similar number of examples
- **Deduplication**: Removes duplicate (subject, predicate, object) triples
- **Metadata preservation**: Includes h5 indices for loading ground truth images during training

**Output format** (JSONL):
```json
{
  "prompt": "a photo of dog riding horse",
  "concepts": ["dog", "riding", "horse"],
  "triple": {
    "subject": "dog",
    "predicate": "riding",
    "object": "horse"
  },
  "meta": {
    "image_rel_path": "VG_100K/12345.jpg",
    "predicate_raw": "riding",
    "class_id": 10,
    "h5_split": "train",
    "img_idx": 42,
    "rel_idx": 7
  }
}
```

**Usage**:
```bash
python make_vg_quickwin_split.py \
  --vocab-path ../../external/SGDiff/datasets/vg/vocab.json \
  --train-h5 /path/to/train_vg.h5 \
  --images-dir /path/to/VG_100K/ \
  --out-dir ../../data/splits/ \
  --tag vg_quickwin \
  --train-total 800 \
  --test-total 160 \
  --seed 42
```

**Outputs**:
- `vg_quickwin_train.jsonl`: Training examples (800 samples)
- `vg_quickwin_test.jsonl`: Test/validation examples (160 samples)
- `vg_quickwin_summary.json`: Statistics and predicate counts

**Current dataset**:
- Train: 800 samples across 16 predicates (~50 per predicate)
- Test: 160 samples for evaluation
- Seed: 42 (reproducible)
- Located: `data/splits/vg_quickwin_train.jsonl`, `vg_quickwin_test.jsonl`

---

## Data Pipeline

1. **Raw VG data**: h5 files with scene graphs + image paths
2. **Filtering**: Select SGDiff-supported predicates only
3. **Splitting**: Create balanced train/test sets
4. **Training**: Load images on-the-fly, cache VAE latents
5. **Evaluation**: Generate images for test set triples

## Notes

- **Predicate vocabulary limitation**: SGDiff encoder was pretrained on specific VG predicates, so we're constrained to its vocabulary
- **Canonical mappings**: Multiple VG predicates can map to same canonical (e.g., "near", "next to" → "around/near")
- **Image loading**: Training script uses metadata to load ground truth images from VG directories
- **Extensibility**: To support more predicates, would need to retrain/extend SGDiff graph encoder
