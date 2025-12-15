# notebooks/

Jupyter notebooks for exploration, analysis, and prototyping.

## Active Notebooks

### datagenerator.ipynb
**Purpose**: Dataset generation and exploration.

**Contents**:
- VG dataset loading and inspection
- Scene graph visualization
- Triple extraction and filtering
- Dataset statistics and distribution analysis

**Usage**: Understanding VG structure before creating train/test splits

---

### graph_exp.ipynb
**Purpose**: Graph encoder experimentation.

**Contents**:
- SGDiff graph encoder testing
- Graph embedding visualization
- Encoding different triple types
- Debugging graph conditioning pipeline

**Usage**: Verify graph encoder outputs before training

---

### sd3_example.ipynb
**Purpose**: Stable Diffusion 3 baseline experiments.

**Contents**:
- SD3 model loading
- Basic image generation
- Comparison with FLUX
- Initial graph conditioning attempts

**Usage**: Reference for SD3 integration (not currently used)

---

## Archive

### archive/
**Purpose**: Old/experimental notebooks kept for reference.

**Expected contents**:
- `trial.ipynb`: Early prototyping and experiments
- `saliency_classfiier.ipynb`: Saliency classifier training attempts
- `concept_attention_analysis.ipynb`: ConceptAttention saliency extraction
- `analyze_attn_maps.ipynb`: Attention map visualization

**Status**: Not actively maintained, kept for historical reference

---

## Notebook Workflow

**Typical exploration pattern**:
1. **Data exploration**: Use `datagenerator.ipynb` to understand VG structure
2. **Graph testing**: Use `graph_exp.ipynb` to verify encoder works
3. **Model prototyping**: Quick tests before writing full training scripts
4. **Analysis**: Visualize results, debug issues, plot metrics

**Moving to production**:
- Prototype in notebooks first
- Extract working code to `src/` modules
- Create executable scripts in `scripts/`
- Archive notebook once functionality is stable

---

## Running Notebooks

**Setup**:
```bash
# Install Jupyter if needed
pip install jupyter notebook

# Launch from project root
cd /path/to/SGRel-DiT
jupyter notebook notebooks/
```

**Kernel**: Use same Python environment as training (with diffusers, torch, etc.)

**GPU access**: Most notebooks assume CUDA available for model loading

---

## Notes

- **Version control**: Large outputs committed can bloat repo - clear outputs before committing
- **Paths**: Notebooks assume run from project root or `notebooks/` directory
- **Dependencies**: Same as main project (see `requirements.txt`)
- **Interactive use only**: For production runs, use scripts in `scripts/`
