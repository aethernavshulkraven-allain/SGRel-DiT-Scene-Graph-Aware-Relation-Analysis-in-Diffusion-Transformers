# C1 Completion Summary: Scene Graph Module Analysis

## Task Completion Report

This document summarizes the completion of Task C1: Locating and analyzing graph modules in the SGDiff repository.

---

## Deliverables

### 1. Primary Documentation
**File**: `GRAPH_MODULE_ANALYSIS.md`

Comprehensive analysis covering:
- Module locations and class hierarchies
- Node and edge representation mappings
- Message passing mechanisms with mathematical notation
- Integration with diffusion model
- Complete reference tables

### 2. Annotated Code Examples
**File**: `annotated_graph_code.py`

Executable Python code demonstrating:
- Node and edge initialization
- Single-layer message passing
- Multi-layer graph convolution
- Global and local feature extraction
- Complete pipeline integration

Both files provide formal, detailed mappings between mathematical notation and implementation.

---

## Key Findings

### Module Locations

| Component | File Path | Lines |
|-----------|-----------|-------|
| Main scene graph encoder | `ldm/modules/cgip/cgip.py` | 14-93 |
| GraphTripleConv (single layer) | `ldm/modules/cgip/cgip.py` | 149-213 |
| GraphTripleConvNet (multi-layer) | `ldm/modules/cgip/cgip.py` | 214-235 |
| Helper functions | `ldm/modules/cgip/tools.py` | 1-80 |
| Diffusion integration | `ldm/models/diffusion/ddpm.py` | 521-526 |
| UNet conditioning | `ldm/modules/diffusionmodules/openaimodel.py` | 715-729 |

### Node Representations

**Mathematical**: h_i ∈ R^512

**Implementation**:
```python
obj_vecs = self.obj_embeddings(objs)  # Shape: (O, 512)
```

**Properties**:
- Dimension: 512
- Initialization: Learned embedding lookup
- Updates: Via message aggregation from connected edges

### Edge Representations

**Mathematical**: r_ij ∈ R^512

**Implementation**:
```python
pred_vecs = self.pred_embeddings(predicates)  # Shape: (T, 512)
```

**Triple Structure**: (subject_idx, predicate_idx, object_idx)

**Properties**:
- Dimension: 512
- Initialization: Learned predicate embedding
- Updates: Via MLP transformation of triple concatenation

### Message Passing Flow

```
Stage 1: Triple Formation
    t_ij = [h_i || r_ij || h_j] ∈ R^1536

Stage 2: MLP Transformation
    [m_s^ij, r'_ij, m_o^ij] = MLP_1(t_ij)
    where m_s, m_o ∈ R^512, r' ∈ R^512

Stage 3: Message Aggregation
    m_i = Σ(messages from neighbors) / degree(i)

Stage 4: Node Update
    h'_i = MLP_2(m_i)

Repeated for L=5 layers
```

### Integration with Diffusion

**Global Features**:
```
h_global = Linear([AvgPool(objects) || AvgPool(predicates)])
Shape: (B, 512)
```

**Local Features**:
```
For each triple: t_local = [h_s || r || h_o]
Shape: (B, 15, 1536) → projected to (B, 15, 512)
```

**Final Conditioning**:
```
context = [local_features (15 tokens), global_features (1 token)]
Shape: (B, 16, 512)
Used in UNet cross-attention layers
```

---

## Notation Mapping Table

| Mathematical Notation | Implementation | Dimension | Description |
|----------------------|----------------|-----------|-------------|
| h_i | `obj_vecs[i]` | 512 | Object node embedding |
| r_ij | `pred_vecs[triple_idx]` | 512 | Predicate/edge embedding |
| (s, p, o) | `triples[idx]` | (3,) | Triple indices |
| E | `edges` | (T, 2) | Edge list |
| m_s^ij | `new_s_vecs` | 512 | Subject message |
| m_o^ij | `new_o_vecs` | 512 | Object message |
| m_i | `pooled_obj_vecs[i]` | 512 | Aggregated message |
| h'_i | `new_obj_vecs[i]` | 512 | Updated node |
| r'_ij | `new_p_vecs[triple_idx]` | 512 | Updated edge |
| t_local | `triple_vec` | 1536 | Local triple feature |
| h_global | `graph_global_fea` | 512 | Global graph feature |
| L | `num_layers` | 5 | Number of GCN layers |

---

## Configuration Parameters

From `config_vg.yaml`:

```yaml
cond_stage_config:
  target: ldm.modules.cgip.cgip.CGIPModel
  params:
    num_objs: 179        # Number of object categories (VG)
    num_preds: 46        # Number of predicate types (VG)
    layers: 5            # Number of graph convolution layers
    width: 512           # Hidden dimension
    embed_dim: 512       # Embedding dimension
    ckpt_path: pretrained/sip_vg.pt
```

---

## Message Passing Architecture Details

### Single Layer Components

1. **MLP_1**: Triple Processor
   - Input: 3 × 512 = 1536 (concatenated triple)
   - Hidden: 512
   - Output: 2 × 512 + 512 = 1536
   - Splits into: [subject_msg (512), predicate (512), object_msg (512)]

2. **Scatter-Add Aggregation**
   - Accumulates messages from all connected edges
   - Separate accumulation for subject and object roles
   - Supports both sum and average pooling

3. **MLP_2**: Node Updater
   - Input: 512 (aggregated messages)
   - Hidden: 512
   - Output: 512 (new node embedding)

### Multi-Layer Architecture

- **Layers**: 5 sequential graph convolutions
- **Dimension preservation**: 512 throughout all layers
- **Parameter sharing**: None (each layer has independent weights)
- **Residual connections**: Not used in this implementation

---

## Integration Points

### 1. Scene Graph to Diffusion
**Location**: `ldm/models/diffusion/ddpm.py:521-526`

```python
def get_learned_conditioning(self, c):
    c_local, c_global = self.cond_stage_model.encode_graph_local_global(c)
    c = {'c_local': c_local, 'c_global': c_global}
    return c
```

### 2. Conditioning in UNet
**Location**: `ldm/modules/diffusionmodules/openaimodel.py:715-729`

```python
def forward(self, x, timesteps=None, c_local=None, c_global=None):
    context_local = self.context_local_mlp(c_local)  # (B, 15, 512)
    context = torch.cat([context_local, c_global], dim=1)  # (B, 16, 512)
    
    # Use in cross-attention
    for module in self.input_blocks:
        h = module(h, emb, context)
```

---

## Validation

### Code Testing
The annotated code (`annotated_graph_code.py`) has been tested and verified:

```
Initial embeddings: (10, 512) objects, (12, 512) predicates
After 5 layers: (10, 512) objects, (12, 512) predicates  
Global features: (2, 512) per image
Local features: (2, 15, 1536) triple features per image
Final context: (2, 16, 512) for UNet conditioning
```

All dimensions match expected values from the formal analysis.

---

## Additional Observations

### Design Choices

1. **Separate node and edge updates**: Unlike standard GCN where only nodes are updated, SGDiff updates both object and predicate embeddings independently.

2. **Bidirectional message passing**: Messages flow from both subject and object roles in each triple, enabling richer information exchange.

3. **Dual conditioning**: The model uses both local (triple-level) and global (graph-level) features to condition the diffusion process.

4. **Fixed architecture**: All graph convolution layers use the same dimensions (512), simplifying the architecture and enabling potential future modifications.

### Computational Characteristics

- **Time complexity per layer**: O(|E| × D^2) where |E| is number of edges, D is dimension
- **Memory usage**: Linear in number of objects and triples
- **Parallelization**: Scatter operations are parallelized across edges

---

## Conclusion

Task C1 has been completed successfully with comprehensive documentation of:

1. Location of all graph processing modules
2. Detailed mapping of node and edge representations
3. Complete analysis of message passing mechanisms
4. Integration points with the diffusion model
5. Executable code examples with annotations

All deliverables use formal notation and avoid colloquial language as requested.

---

**Files Generated**:
- `GRAPH_MODULE_ANALYSIS.md` (Formal documentation)
- `annotated_graph_code.py` (Executable examples)
- `C1_COMPLETION_SUMMARY.md` (This file)

**Status**: Complete
**Date**: November 11, 2025
