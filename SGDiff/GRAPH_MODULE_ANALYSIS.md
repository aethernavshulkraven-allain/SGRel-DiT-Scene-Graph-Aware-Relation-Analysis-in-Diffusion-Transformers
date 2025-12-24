# Scene Graph Module Analysis: Node and Edge Representations

## Executive Summary

This document provides a comprehensive analysis of the scene graph processing modules in SGDiff, focusing on node and edge representations, message passing mechanisms, and integration with the diffusion model.

## 1. Primary Graph Module Locations

### 1.1 Core Implementation
**File**: `ldm/modules/cgip/cgip.py`

**Key Classes**:
- `CGIPModel`: Main scene graph encoder integrating with the diffusion model
- `GraphTripleConv`: Single layer of scene graph convolution
- `GraphTripleConvNet`: Multi-layer graph convolutional network

**Auxiliary File**: `ldm/modules/cgip/tools.py`
- Contains helper functions for tensor manipulation and scene graph encoding

### 1.2 Pretrained Module
**File**: `sg_image_pretraining/sgCLIP/module.py`
- Contains identical graph convolution implementations used for masked contrastive pretraining
- Same architecture as the main diffusion conditioning module

---

## 2. Node Representations

### 2.1 Object Node Embeddings (h_i)

#### Mathematical Notation Mapping
```
Objects in scene graph → Node embeddings h_i ∈ R^D
where D = embed_dim (typically 512)
```

#### Implementation Details

**Location**: `CGIPModel.__init__()` (line 33)
```python
self.obj_embeddings = nn.Embedding(num_objs + 1, embed_dim)
```

**Parameters**:
- `num_objs`: Number of object categories (179 for Visual Genome)
- `embed_dim`: Embedding dimension (512 in config_vg.yaml)
- Additional +1 for special `__image__` token

**Initialization Process** (line 71-75):
```python
obj_vecs = self.obj_embeddings(objs)  # Shape: (O, D)
# O = total number of objects across batch
# D = embed_dim = 512
```

#### Node Features
Each object node contains:
1. **Category embedding**: Learned embedding from vocabulary index
2. **Bounding box information**: Used in dataset but not directly in graph convolution
3. **Batch assignment**: `obj_to_img` tensor mapping objects to images

---

## 3. Edge Representations

### 3.1 Relationship/Predicate Embeddings (r_ij)

#### Mathematical Notation Mapping
```
Triple (subject, predicate, object) → Edge representation r_ij ∈ R^D
where predicate → pred_vecs ∈ R^D
```

#### Implementation Details

**Location**: `CGIPModel.__init__()` (line 34)
```python
self.pred_embeddings = nn.Embedding(num_preds, embed_dim)
```

**Parameters**:
- `num_preds`: Number of predicate types (46 for Visual Genome)
- `embed_dim`: Embedding dimension (512)

**Triple Structure** (line 69-71):
```python
s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)
```

Where:
- `s`: Subject object indices
- `p`: Predicate indices
- `o`: Object indices
- `T`: Total number of triples

**Predicate Embedding** (line 74):
```python
pred_vecs = self.pred_embeddings(p)  # Shape: (T, D)
```

---

## 4. Message Passing Mechanism

### 4.1 GraphTripleConv Architecture

**Location**: `GraphTripleConv.forward()` (lines 174-210)

#### Stage 1: Triple Formation
```python
# Extract subject and object vectors
cur_s_vecs = obj_vecs[s_idx]  # Shape: (T, Din)
cur_o_vecs = obj_vecs[o_idx]  # Shape: (T, Din)

# Concatenate triple: [subject, predicate, object]
cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
# Shape: (T, 3*Din)
```

**Mathematical Representation**:
```
t_ij = [h_i || r_ij || h_j]
where || denotes concatenation
h_i: subject node embedding
r_ij: predicate embedding
h_j: object node embedding
```

#### Stage 2: Triple Transformation (net1)
```python
net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
new_t_vecs = self.net1(cur_t_vecs)  # Shape: (T, 2*H + Dout)
```

**MLP Structure**:
- Input: 3D → Hidden: H → Output: 2H + D_out
- H = hidden_dim (typically 512)
- D_out = output_dim (typically 512)

**Output Decomposition**:
```python
new_s_vecs = new_t_vecs[:, :H]                    # Shape: (T, H)
new_p_vecs = new_t_vecs[:, H:(H + Dout)]         # Shape: (T, Dout)
new_o_vecs = new_t_vecs[:, (H + Dout):(2*H + Dout)]  # Shape: (T, H)
```

**Mathematical Representation**:
```
[m_s^ij, r'_ij, m_o^ij] = MLP([h_i || r_ij || h_j])

where:
m_s^ij ∈ R^H: message from subject
m_o^ij ∈ R^H: message from object
r'_ij ∈ R^Dout: updated predicate embedding
```

#### Stage 3: Message Aggregation
```python
pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

# Scatter-add messages from subject and object roles
s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)
```

**Mathematical Representation**:
```
m_i = Σ_{j:(i,r,j)∈E} m_s^ij + Σ_{k:(k,r,i)∈E} m_o^ki

where E is the set of all edges/triples
```

#### Stage 4: Pooling (Optional Average)
```python
if self.pooling == 'avg':
    obj_counts = torch.zeros(O, dtype=dtype, device=device)
    ones = torch.ones(T, dtype=dtype, device=device)
    obj_counts = obj_counts.scatter_add(0, s_idx, ones)
    obj_counts = obj_counts.scatter_add(0, o_idx, ones)
    obj_counts = obj_counts.clamp(min=1)
    pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
```

**Mathematical Representation** (for average pooling):
```
m_i = m_i / |{edges connected to node i}|
```

#### Stage 5: Node Update (net2)
```python
net2_layers = [hidden_dim, hidden_dim, output_dim]
new_obj_vecs = self.net2(pooled_obj_vecs)  # Shape: (O, Dout)
```

**Mathematical Representation**:
```
h'_i = MLP_2(m_i)
```

**Final Output**:
```python
return new_obj_vecs, new_p_vecs
# new_obj_vecs: Shape (O, Dout) - updated object embeddings
# new_p_vecs: Shape (T, Dout) - updated predicate embeddings
```

---

### 4.2 GraphTripleConvNet: Multi-layer Architecture

**Location**: `GraphTripleConvNet` (lines 214-235)

```python
class GraphTripleConvNet(nn.Module):
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, 
                 pooling='avg', mlp_normalization='none'):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))
    
    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs
```

**Configuration** (from config_vg.yaml):
- `layers`: 5 (number of graph convolution layers)
- `width`: 512 (hidden dimension)
- `embed_dim`: 512 (input/output dimension)

**Mathematical Representation**:
```
h^(0) = initial object embeddings
r^(0) = initial predicate embeddings

For l = 1 to L:
    h^(l), r^(l) = GraphTripleConv(h^(l-1), r^(l-1), edges)

Output: h^(L), r^(L)
```

---

## 5. Integration with Diffusion Model

### 5.1 Scene Graph Encoding Pipeline

**Location**: `CGIPModel.encode_graph_local_global()` (lines 62-92)

#### Step 1: Initial Embeddings
```python
obj_vecs = self.obj_embeddings(objs)      # Shape: (O, 512)
pred_vecs = self.pred_embeddings(p)       # Shape: (T, 512)
```

#### Step 2: Graph Convolution
```python
# Single layer
obj_vecs, pred_vecs = self.graph_conv(obj_vecs, pred_vecs, edges)

# Multi-layer network (5 layers)
obj_vecs, pred_vecs = self.graph_net(obj_vecs, pred_vecs, edges)
```

#### Step 3: Global Feature Extraction
```python
# Pool object features per image
obj_fea = self.pool_samples(obj_vecs, obj_to_img)  # Shape: (B, 512)

# Pool predicate features per image
pred_fea = self.pool_samples(pred_vecs, triples_to_img)  # Shape: (B, 512)

# Combine and project
graph_global_fea = self.graph_projection(
    torch.cat([obj_fea, pred_fea], dim=1)
)  # Shape: (B, 512)
```

**Mathematical Representation**:
```
For each image i in batch:
    h_global^i = Project([AvgPool(h_i^objects) || AvgPool(r_i^predicates)])
```

#### Step 4: Local Feature Extraction
```python
# Extract subject and object vectors for each triple
s_obj_vec, o_obj_vec = obj_vecs[s], obj_vecs[o]

# Concatenate to form triple representation
triple_vec = torch.cat([s_obj_vec, pred_vecs, o_obj_vec], dim=1)
# Shape: (T, 1536)  [512 + 512 + 512]

# Organize triples per image (max 15 per image)
graph_local_fea = create_tensor_by_assign_samples_to_img(
    samples=triple_vec,
    sample_to_img=triples_to_img,
    max_sample_per_img=self.max_relationships_per_image,
    batch_size=batch_size
)  # Shape: (B, 15, 1536)
```

**Mathematical Representation**:
```
For each triple (i,r,j):
    t_local^(i,r,j) = [h_i || r_ij || h_j] ∈ R^(3*512)

For each image in batch:
    LocalFeatures = [t_1, t_2, ..., t_min(K,15)] (padded to 15 triples)
```

---

### 5.2 Conditioning the Diffusion Model

**Location**: `LatentDiffusion.get_learned_conditioning()` (lines 521-526, ddpm.py)

```python
def get_learned_conditioning(self, c):
    c_local, c_global = self.cond_stage_model.encode_graph_local_global(c)
    c_local = c_local.detach()  # Shape: (B, 15, 1536)
    c_global = c_global.unsqueeze(1).detach()  # Shape: (B, 1, 512)
    c = {'c_local': c_local, 'c_global': c_global}
    return c
```

**Location**: `UNetModel.forward()` (lines 715-729, openaimodel.py)

```python
def forward(self, x, timesteps=None, c_local=None, c_global=None):
    # Project local features to match global dimension
    context_local = self.context_local_mlp(c_local)  # (B, 15, 512)
    
    # Concatenate local and global context
    context = torch.cat([context_local, c_global], dim=1)  # (B, 16, 512)
    
    # Use context in cross-attention layers of UNet
    for module in self.input_blocks:
        h = module(h, emb, context)
```

**Context Dimension**:
- Local: 15 triples × 512 dimensions = (B, 15, 512)
- Global: 1 graph summary × 512 dimensions = (B, 1, 512)
- Combined: (B, 16, 512) fed to cross-attention in UNet

---

## 6. Summary: Node and Edge Mapping

### 6.1 Complete Mapping Table

| Concept | Mathematical Notation | Implementation | Dimension |
|---------|----------------------|----------------|-----------|
| Object node | h_i | `obj_vecs[i]` | 512 |
| Predicate/edge | r_ij | `pred_vecs[triple_idx]` | 512 |
| Triple | (s, p, o) | `triples[idx] = [subject_idx, pred_idx, object_idx]` | (3,) indices |
| Edge structure | E | `edges = [[s_0, o_0], [s_1, o_1], ...]` | (T, 2) |
| Subject message | m_s^ij | `new_s_vecs` | 512 |
| Object message | m_o^ij | `new_o_vecs` | 512 |
| Aggregated message | m_i | `pooled_obj_vecs[i]` | 512 |
| Updated node | h'_i | `new_obj_vecs[i]` | 512 |
| Updated edge | r'_ij | `new_p_vecs[triple_idx]` | 512 |
| Local triple feature | t_local | `[h_s || r || h_o]` | 1536 |
| Global graph feature | h_global | `graph_global_fea` | 512 |

### 6.2 Message Passing Flow

```
Initial State:
    h^(0)_i ← obj_embeddings(object_category_i)
    r^(0)_ij ← pred_embeddings(predicate_ij)

For each layer l = 1 to 5:
    For each triple (i, r, j):
        t_ij = [h^(l-1)_i || r^(l-1)_ij || h^(l-1)_j]
        [m_s, r'_ij, m_o] = MLP_1(t_ij)
    
    For each node i:
        m_i = Aggregate({m_s from outgoing edges, m_o from incoming edges})
        h^(l)_i = MLP_2(m_i)

Final State:
    Object features: h^(5)_i for all nodes
    Predicate features: r^(5)_ij for all edges

Global Conditioning:
    h_global = Project([AvgPool(h^(5)) || AvgPool(r^(5))])

Local Conditioning:
    For each triple: t_local = [h^(5)_s || r^(5) || h^(5)_o]
```

---

## 7. Key Architectural Insights

### 7.1 Dual Representation
The model maintains separate but interacting representations:
- **Node features** (h_i): Updated via message aggregation from connected triples
- **Edge features** (r_ij): Updated via MLP transformation of triple concatenation

### 7.2 Message Passing Pattern
- **Bidirectional**: Messages flow from both subject and object roles
- **Separate channels**: Subject and object messages are computed separately before aggregation
- **Edge updates**: Predicates are updated independently of node aggregation

### 7.3 Hierarchical Conditioning
The diffusion model receives two levels of scene graph information:
- **Local**: Individual triple features for fine-grained control
- **Global**: Pooled graph summary for overall scene understanding

### 7.4 Dimension Consistency
All embeddings maintain 512 dimensions throughout:
- Initial embeddings: 512
- Hidden representations: 512
- Final features: 512
- This enables residual connections and stable training

---

## 8. File Reference Summary

### Primary Implementation Files
1. `ldm/modules/cgip/cgip.py`: Main scene graph encoder
   - Lines 14-93: CGIPModel class
   - Lines 149-213: GraphTripleConv (single layer)
   - Lines 214-235: GraphTripleConvNet (multi-layer)

2. `ldm/modules/cgip/tools.py`: Helper functions
   - Lines 5-25: Tensor organization utilities

3. `ldm/models/diffusion/ddpm.py`: Integration with diffusion
   - Lines 521-526: Conditioning extraction

4. `ldm/modules/diffusionmodules/openaimodel.py`: UNet architecture
   - Lines 715-729: Context integration in forward pass

### Configuration
- `config_vg.yaml`: Lines 67-71 specify CGIPModel parameters
  - num_objs: 179
  - num_preds: 46
  - layers: 5
  - width: 512
  - embed_dim: 512

---

This analysis provides a complete mapping between mathematical notation and implementation for the scene graph processing modules in SGDiff.
