# external/

External dependencies and forked repositories.

## Repositories

### ConceptAttention/
**Purpose**: ConceptAttention saliency extraction implementation.

**Source**: https://github.com/helblazer811/ConceptAttention

**Paper**: "ConceptAttention: Attention Mechanisms for Concept-Level Scene Understanding in Diffusion Transformers"

**What it provides**:
- Attention-based saliency extraction from DiT models
- Per-concept saliency maps (subject, predicate, object)
- Sharp attention maps from transformer output space

**Usage in project**:
- Originally intended for L_rel_rank loss (frozen classifier on saliencies)
- Current status: Not actively used in 20-epoch training (pure generative loss)
- Future: Would feed saliencies to classifier for contrastive learning

**Key files**:
- Saliency extraction utilities
- Concept segmentation helpers
- Attention map processing

---

### ConceptAttention_/
**Purpose**: Fork/modified version of ConceptAttention.

**Status**: Duplicate or experimental fork, likely can be consolidated

**Recommendation**: Merge any local changes back to main `ConceptAttention/` and remove duplicate

---

### SGDiff/
**Purpose**: SGDiff baseline model and graph encoder.

**Source**: https://github.com/YangLing0818/SGDiff

**Paper**: "SGDiff: Scene Graph-Conditional Diffusion Models for Layout-to-Image Generation"

**What it provides**:
- **CGIP Graph Encoder**: Converts scene graph triples to embeddings
  - Local tokens: Per-concept embeddings (subject, predicate, object)
  - Global token: Scene-level graph representation
- **Pretrained weights**: Graph encoder trained on VG
- **VG vocabulary**: Predicate and object name mappings
- **Scene graph utilities**: Encoding, processing, visualization

**Usage in project**:
- Core dependency for graph conditioning
- `SGDiffGraphEncoder` wrapper in `src/graph/sgdiff_encoder.py`
- Vocabulary limits us to 16 supported predicates

**Key files**:
- `ldm/modules/cgip/`: Graph encoder implementation
- `datasets/vg/vocab.json`: VG predicate vocabulary
- Pretrained checkpoint loaded by encoder

**Limitations**:
- Fixed vocabulary (16 out of 24 canonical predicates supported)
- Pretrained on specific VG distribution
- Would need retraining to support additional predicates

---

## Why External?

These are **external repositories** (not our code):
- Keep separate to track upstream changes
- Enable easy updates via git pull
- Clear attribution for research work
- Avoid mixing external code with our implementations

## Updates

**Checking for updates**:
```bash
# ConceptAttention
cd external/ConceptAttention
git pull origin master

# SGDiff
cd external/SGDiff
git pull origin main
```

**Caution**: Updates may break integration - test after pulling

---

## Integration

**Import pattern**:
```python
# Graph encoder
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder

# Wrapper loads from external/SGDiff internally:
encoder = SGDiffGraphEncoder(
    vocab_path="external/SGDiff/datasets/vg/vocab.json",
    cgip_ckpt="external/SGDiff/checkpoints/cgip.pth"
)
```

**ConceptAttention** (future):
```python
from external.ConceptAttention.concept_attention import extract_saliency
```

---

## License Notes

- **ConceptAttention**: Check original repo for license
- **SGDiff**: Check original repo for license
- Our code: Separate license applies to `src/`, `scripts/`

When publishing, ensure proper citation of external work.

---

## Storage

- **ConceptAttention**: ~50-100 MB (code + any pretrained models)
- **SGDiff**: ~200-500 MB (code + CGIP checkpoint)
- **Total**: ~300-600 MB

Not included in main codebase size estimates.
