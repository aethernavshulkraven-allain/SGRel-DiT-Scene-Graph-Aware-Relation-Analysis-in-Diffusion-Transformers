# src/

Core source code modules for graph-conditioned diffusion models.

## Directory Structure

### graph/
**Purpose**: Graph encoding and scene graph processing.

**Files**:
- `sgdiff_encoder.py`: SGDiff graph encoder integration
  - Wraps SGDiff's CGIP (Conditional Graph Injection Pipeline) encoder
  - Converts (subject, predicate, object) triples to graph embeddings
  - Outputs local (per-concept) and global (scene-level) graph tokens
  - Supports 16 VG predicates from SGDiff vocabulary

### flux/
**Purpose**: FLUX model modifications and graph conditioning.

**Files**:
- `graph_conditioned_flux.py`: Core graph injection mechanism
  - `patch_flux_for_graph()`: Modifies FLUX transformer to accept graph tokens
  - `set_graph_condition()`: Sets graph embeddings for generation
  - Injects graph tokens into specified transformer blocks (7-12)
  - Supports "concat" mode (prepend to context) or "add" mode (additive fusion)

- `lora.py`: LoRA (Low-Rank Adaptation) utilities
  - Attaches LoRA adapters to attention projections (Q, K, V, out)
  - Configurable rank and alpha parameters
  - Selective block targeting (e.g., blocks 7-12 only)
  - Save/load LoRA state dicts

### concept-attention/
**Purpose**: ConceptAttention saliency extraction (future work).

- Not actively used in current training pipeline
- Intended for computing attention-based saliency maps
- Would feed into L_rel_rank loss via frozen classifier

### data/
**Purpose**: Data loading and preprocessing utilities.

**Files**:
- Likely contains VG data loaders and preprocessing helpers
- Handles image loading, VAE encoding, text tokenization

### prompt_builder.py
**Purpose**: Convert predicates to natural language phrases.

**Functions**:
- `predicate_to_phrase()`: Maps canonical predicates to human-readable text
  - Example: "riding" → "riding", "on" → "on top of", "in" → "inside"
- Used for generating prompts: `"a photo of {subject} {phrase} {object}"`

### schema.py
**Purpose**: Data schemas and type definitions.

- Defines dataclasses/types for triples, prompts, metadata
- Ensures consistent data format across pipeline

### vg_loader.py
**Purpose**: Visual Genome dataset loader.

- Loads VG h5 files and image directories
- Parses scene graphs and object annotations
- Filters to supported predicates

### utils/
**Purpose**: Shared utility functions.

- Image processing helpers
- Checkpoint saving/loading
- Logging and experiment tracking
- Path management

### evaluation/
**Purpose**: Evaluation metrics and utilities (moved from scripts).

- CLIP scoring helpers
- LPIPS computation
- Spatial reasoning utilities
- Metric aggregation

### training/
**Purpose**: Training utilities and loss functions.

- Loss implementations (L_gen, L_rel_rank, L_gen_rel)
- Optimizer setup
- Learning rate schedulers
- Training loop helpers

### models/
**Purpose**: Model architecture definitions.

- Custom layers and modules
- Graph fusion mechanisms
- Future: standalone model definitions

### stage_b/
**Purpose**: Legacy/experimental Stage-B models.

- Not currently used
- Kept for reference

---

## Key Integrations

**Training flow**:
1. Load VG triple from `vg_loader.py`
2. Encode graph via `graph/sgdiff_encoder.py`
3. Inject into FLUX using `flux/graph_conditioned_flux.py`
4. Apply LoRA adapters from `flux/lora.py`
5. Compute losses and train

**Evaluation flow**:
1. Load checkpoint with LoRA weights
2. Generate images with g+ and g- graphs
3. Compute metrics using `evaluation/` utilities
4. Save results

## Import Notes

Most imports follow pattern:
```python
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.flux.graph_conditioned_flux import patch_flux_for_graph
```

Path setup in scripts:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```
