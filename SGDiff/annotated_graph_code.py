"""
Annotated Code Examples: Scene Graph Message Passing

This file provides detailed annotations mapping the implementation to mathematical notation
for the scene graph processing in SGDiff.
"""

import torch
import torch.nn as nn

# ==============================================================================
# SECTION 1: Node and Edge Initialization
# ==============================================================================

class SceneGraphEmbeddings:
    """
    Initial embeddings for nodes (objects) and edges (predicates).
    
    Mathematical Notation:
        h_i^(0) = Embed_obj(category_i)  for object nodes
        r_ij^(0) = Embed_pred(predicate_ij)  for edges/predicates
    """
    
    def __init__(self, num_objs=179, num_preds=46, embed_dim=512):
        """
        Args:
            num_objs: Number of object categories (179 for VG)
            num_preds: Number of predicate types (46 for VG)
            embed_dim: Embedding dimension (512)
        """
        # Object embedding layer: vocabulary_idx → h_i ∈ R^512
        self.obj_embeddings = nn.Embedding(num_objs + 1, embed_dim)
        
        # Predicate embedding layer: predicate_idx → r_ij ∈ R^512
        self.pred_embeddings = nn.Embedding(num_preds, embed_dim)
    
    def forward(self, objs, predicates):
        """
        Args:
            objs: Tensor of shape (O,) containing object category indices
            predicates: Tensor of shape (T,) containing predicate indices
        
        Returns:
            obj_vecs: Tensor of shape (O, 512) - h_i for each object
            pred_vecs: Tensor of shape (T, 512) - r_ij for each predicate
        
        Mathematical:
            For each object i: h_i^(0) = obj_embeddings[objs[i]]
            For each predicate j: r_j^(0) = pred_embeddings[predicates[j]]
        """
        obj_vecs = self.obj_embeddings(objs)      # (O, 512) - node embeddings
        pred_vecs = self.pred_embeddings(predicates)  # (T, 512) - edge embeddings
        
        return obj_vecs, pred_vecs


# ==============================================================================
# SECTION 2: Message Passing - Single Layer
# ==============================================================================

class AnnotatedGraphTripleConv(nn.Module):
    """
    Single layer of scene graph convolution with detailed annotations.
    
    Mathematical Operations:
    
    1. Triple Formation:
        t_ij = [h_i || r_ij || h_j]  where || is concatenation
    
    2. MLP Transformation:
        [m_s^ij, r'_ij, m_o^ij] = MLP_1(t_ij)
        where m_s^ij, m_o^ij ∈ R^H (hidden dim)
              r'_ij ∈ R^D (output dim)
    
    3. Message Aggregation:
        m_i = Σ_{j:(i,r,j)∈E} m_s^ij + Σ_{k:(k,r,i)∈E} m_o^ki
    
    4. Pooling (optional):
        m_i = m_i / |neighbors(i)|
    
    5. Node Update:
        h'_i = MLP_2(m_i)
    """
    
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, pooling='avg'):
        super().__init__()
        self.input_dim = input_dim    # D_in = 512
        self.output_dim = output_dim  # D_out = 512
        self.hidden_dim = hidden_dim  # H = 512
        self.pooling = pooling
        
        # MLP_1: Processes concatenated triples
        # Input: 3 * D_in (subject + predicate + object)
        # Output: 2 * H + D_out (subject_msg + new_pred + object_msg)
        self.net1 = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim + output_dim)
        )
        
        # MLP_2: Updates node representations
        # Input: H (aggregated messages)
        # Output: D_out (new node embedding)
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Args:
            obj_vecs: (O, D_in) - Current object node embeddings h_i
            pred_vecs: (T, D_in) - Current predicate embeddings r_ij
            edges: (T, 2) - Edge list [[s_0, o_0], [s_1, o_1], ...]
                   where s_idx is subject, o_idx is object
        
        Returns:
            new_obj_vecs: (O, D_out) - Updated node embeddings h'_i
            new_p_vecs: (T, D_out) - Updated predicate embeddings r'_ij
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O = obj_vecs.size(0)  # Number of objects
        T = pred_vecs.size(0)  # Number of triples
        Din = self.input_dim
        H = self.hidden_dim
        Dout = self.output_dim
        
        # Extract subject and object indices from edges
        # Mathematical: For edge (i, r, j): s_idx = i, o_idx = j
        s_idx = edges[:, 0].contiguous()  # Shape: (T,)
        o_idx = edges[:, 1].contiguous()  # Shape: (T,)
        
        # ------------------------------------------------------------------
        # STEP 1: Triple Formation
        # Mathematical: t_ij = [h_i || r_ij || h_j]
        # ------------------------------------------------------------------
        cur_s_vecs = obj_vecs[s_idx]  # (T, Din) - h_i for each subject
        cur_o_vecs = obj_vecs[o_idx]  # (T, Din) - h_j for each object
        
        # Concatenate: [subject, predicate, object]
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        # Shape: (T, 3*Din) = (T, 1536)
        
        # ------------------------------------------------------------------
        # STEP 2: MLP Transformation
        # Mathematical: [m_s^ij, r'_ij, m_o^ij] = MLP_1(t_ij)
        # ------------------------------------------------------------------
        new_t_vecs = self.net1(cur_t_vecs)
        # Shape: (T, 2*H + Dout) = (T, 1536)
        
        # Decompose output into three components:
        new_s_vecs = new_t_vecs[:, :H]                    # (T, H) - m_s^ij
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]          # (T, Dout) - r'_ij
        new_o_vecs = new_t_vecs[:, (H + Dout):(2*H + Dout)]  # (T, H) - m_o^ij
        
        # ------------------------------------------------------------------
        # STEP 3: Message Aggregation via Scatter-Add
        # Mathematical: m_i = Σ_{neighbors} messages
        # ------------------------------------------------------------------
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)
        
        # Expand indices for scatter operation
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)  # (T, H)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)  # (T, H)
        
        # Accumulate subject messages: For edge (i→j), add m_s to node i
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        
        # Accumulate object messages: For edge (i→j), add m_o to node j
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)
        
        # Mathematical after aggregation:
        # pooled_obj_vecs[i] = Σ_{j:(i,r,j)∈E} m_s^ij + Σ_{k:(k,r,i)∈E} m_o^ki
        
        # ------------------------------------------------------------------
        # STEP 4: Pooling (Average)
        # Mathematical: m_i = m_i / degree(i)
        # ------------------------------------------------------------------
        if self.pooling == 'avg':
            # Count number of edges connected to each node
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            
            # Count outgoing edges for subjects
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            # Count incoming edges for objects
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)
            
            # Avoid division by zero for isolated nodes
            obj_counts = obj_counts.clamp(min=1)
            
            # Average pooling
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
        
        # ------------------------------------------------------------------
        # STEP 5: Node Update via MLP
        # Mathematical: h'_i = MLP_2(m_i)
        # ------------------------------------------------------------------
        new_obj_vecs = self.net2(pooled_obj_vecs)
        # Shape: (O, Dout)
        
        return new_obj_vecs, new_p_vecs


# ==============================================================================
# SECTION 3: Multi-Layer Message Passing
# ==============================================================================

class AnnotatedGraphTripleConvNet(nn.Module):
    """
    Multi-layer graph convolutional network.
    
    Mathematical:
        h^(0), r^(0) = initial embeddings
        
        For l = 1 to L:
            h^(l), r^(l) = GraphTripleConv(h^(l-1), r^(l-1), edges)
        
        Output: h^(L), r^(L)
    
    Configuration (config_vg.yaml):
        L = 5 layers
        input_dim = 512
        hidden_dim = 512
        output_dim = 512
    """
    
    def __init__(self, input_dim=512, num_layers=5, hidden_dim=512, pooling='avg'):
        super().__init__()
        self.num_layers = num_layers  # L = 5
        
        # Create L graph convolution layers
        self.gconvs = nn.ModuleList()
        for _ in range(num_layers):
            self.gconvs.append(AnnotatedGraphTripleConv(
                input_dim=input_dim,
                output_dim=input_dim,  # Keep same dimension
                hidden_dim=hidden_dim,
                pooling=pooling
            ))
    
    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Args:
            obj_vecs: (O, 512) - Initial node embeddings h^(0)
            pred_vecs: (T, 512) - Initial edge embeddings r^(0)
            edges: (T, 2) - Edge list
        
        Returns:
            obj_vecs: (O, 512) - Final node embeddings h^(L)
            pred_vecs: (T, 512) - Final edge embeddings r^(L)
        
        Mathematical:
            Layer 0: h^(0), r^(0) (input)
            Layer 1: h^(1), r^(1) = Conv(h^(0), r^(0), edges)
            Layer 2: h^(2), r^(2) = Conv(h^(1), r^(1), edges)
            ...
            Layer 5: h^(5), r^(5) = Conv(h^(4), r^(4), edges)
            
            Return: h^(5), r^(5)
        """
        for layer_idx in range(self.num_layers):
            gconv = self.gconvs[layer_idx]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
            # After each layer:
            # obj_vecs shape: (O, 512)
            # pred_vecs shape: (T, 512)
        
        return obj_vecs, pred_vecs


# ==============================================================================
# SECTION 4: Global and Local Feature Extraction
# ==============================================================================

class AnnotatedFeatureExtraction:
    """
    Extract global and local features from processed scene graph.
    
    Mathematical:
        Global: h_global = Project([AvgPool(h^(L)) || AvgPool(r^(L))])
        Local: For each triple (i,r,j): t_local = [h_i^(L) || r_ij^(L) || h_j^(L)]
    """
    
    @staticmethod
    def pool_samples_per_image(samples, sample_to_img):
        """
        Pool features by averaging per image in batch.
        
        Args:
            samples: (N_total, D) - Features across entire batch
            sample_to_img: (N_total,) - Image index for each sample
        
        Returns:
            pooled: (B, D) - Average features per image
        
        Mathematical:
            For image i: pooled[i] = (1/|S_i|) Σ_{s∈S_i} samples[s]
            where S_i is the set of samples belonging to image i
        """
        dtype, device = samples.dtype, samples.device
        N_total, D = samples.size()
        B = sample_to_img.max().item() + 1  # Batch size
        
        # Initialize output
        out = torch.zeros(B, D, dtype=dtype, device=device)
        
        # Scatter-add: accumulate samples per image
        idx = sample_to_img.view(N_total, 1).expand(N_total, D)
        out = out.scatter_add(0, idx, samples)
        
        # Count samples per image
        ones = torch.ones(N_total, dtype=dtype, device=device)
        counts = torch.zeros(B, dtype=dtype, device=device)
        counts = counts.scatter_add(0, sample_to_img, ones)
        counts = counts.clamp(min=1)
        
        # Average pooling
        out = out / counts.view(B, 1)
        
        return out  # (B, D)
    
    @staticmethod
    def extract_global_features(obj_vecs, pred_vecs, obj_to_img, triples_to_img, 
                                embed_dim=512):
        """
        Extract global graph features per image.
        
        Args:
            obj_vecs: (O, 512) - Final node embeddings h^(L)
            pred_vecs: (T, 512) - Final edge embeddings r^(L)
            obj_to_img: (O,) - Image assignment for objects
            triples_to_img: (T,) - Image assignment for triples
        
        Returns:
            graph_global_fea: (B, 512) - Global features per image
        
        Mathematical:
            For each image i:
                obj_pool_i = AvgPool(h^(L) for objects in image i)
                pred_pool_i = AvgPool(r^(L) for predicates in image i)
                h_global_i = Linear([obj_pool_i || pred_pool_i])
        """
        # Pool object features per image
        obj_fea = AnnotatedFeatureExtraction.pool_samples_per_image(
            obj_vecs, obj_to_img
        )  # (B, 512)
        
        # Pool predicate features per image
        pred_fea = AnnotatedFeatureExtraction.pool_samples_per_image(
            pred_vecs, triples_to_img
        )  # (B, 512)
        
        # Concatenate and project
        graph_projection = nn.Linear(embed_dim * 2, embed_dim)  # 1024 → 512
        graph_global_fea = graph_projection(
            torch.cat([obj_fea, pred_fea], dim=1)
        )  # (B, 512)
        
        return graph_global_fea
    
    @staticmethod
    def extract_local_features(obj_vecs, pred_vecs, triples, triples_to_img,
                               max_triples_per_image=15, batch_size=None):
        """
        Extract local triple features per image.
        
        Args:
            obj_vecs: (O, 512) - Final node embeddings h^(L)
            pred_vecs: (T, 512) - Final edge embeddings r^(L)
            triples: (T, 3) - Triples [subject_idx, pred_idx, object_idx]
            triples_to_img: (T,) - Image assignment for triples
            max_triples_per_image: Maximum triples to keep per image (15)
        
        Returns:
            graph_local_fea: (B, 15, 1536) - Local triple features
        
        Mathematical:
            For each triple (i, r_idx, j):
                t_local = [h_i^(L) || r_ij^(L) || h_j^(L)] ∈ R^1536
            
            For each image:
                LocalFeatures = [t_1, t_2, ..., t_K] (padded/truncated to 15)
        """
        # Extract subject and object indices
        s = triples[:, 0]  # Subject indices
        o = triples[:, 2]  # Object indices
        
        # Get subject and object vectors
        s_obj_vec = obj_vecs[s]  # (T, 512)
        o_obj_vec = obj_vecs[o]  # (T, 512)
        
        # Concatenate to form triple vectors
        triple_vec = torch.cat([s_obj_vec, pred_vecs, o_obj_vec], dim=1)
        # Shape: (T, 1536) = (T, 512+512+512)
        
        # Organize triples per image (with padding/truncation)
        graph_local_fea = organize_samples_by_image(
            samples=triple_vec,
            sample_to_img=triples_to_img,
            max_samples_per_img=max_triples_per_image,
            batch_size=batch_size
        )  # (B, 15, 1536)
        
        return graph_local_fea


def organize_samples_by_image(samples, sample_to_img, max_samples_per_img, batch_size):
    """
    Organize samples into fixed-size arrays per image.
    
    Args:
        samples: (N, D) - Feature vectors
        sample_to_img: (N,) - Image assignment
        max_samples_per_img: Maximum samples per image (e.g., 15)
        batch_size: Number of images
    
    Returns:
        organized: (B, max_samples_per_img, D)
    
    Implementation:
        For each image:
            - If fewer than max: pad with zeros
            - If more than max: truncate
            - Result: exactly max_samples_per_img vectors
    """
    device = samples.device
    D = samples.shape[1]
    result = []
    
    for img_idx in range(batch_size):
        # Get samples for this image
        mask = (sample_to_img == img_idx)
        img_samples = samples[mask]  # (K, D) where K varies
        
        K = img_samples.shape[0]
        
        if K > max_samples_per_img:
            # Truncate
            img_samples = img_samples[:max_samples_per_img]
        elif K < max_samples_per_img:
            # Pad with zeros
            padding = torch.zeros(
                max_samples_per_img - K, D, 
                dtype=samples.dtype, 
                device=device
            )
            img_samples = torch.cat([img_samples, padding], dim=0)
        
        result.append(img_samples.unsqueeze(0))  # (1, max_samples, D)
    
    organized = torch.cat(result, dim=0)  # (B, max_samples, D)
    return organized


# ==============================================================================
# SECTION 5: Complete Pipeline Example
# ==============================================================================

def complete_pipeline_example():
    """
    Complete example showing the full pipeline from scene graph to conditioning.
    
    Mathematical Flow:
        1. Input: Scene graph G = (V, E) with |V| objects, |E| triples
        2. Initialize: h^(0) = Embed(objects), r^(0) = Embed(predicates)
        3. Process: h^(L), r^(L) = GraphConvNet(h^(0), r^(0), edges)
        4. Extract: h_global, t_local = Features(h^(L), r^(L))
        5. Condition: UNet(x_t, t, [t_local, h_global])
    """
    
    # Example scene graph
    batch_size = 2
    num_objects = 10  # O = 10 objects total across batch
    num_triples = 12  # T = 12 triples total across batch
    embed_dim = 512
    
    # Step 1: Initialize embeddings
    embeddings = SceneGraphEmbeddings(num_objs=179, num_preds=46, embed_dim=512)
    
    # Example data (would come from dataloader)
    objs = torch.randint(0, 179, (num_objects,))  # Object categories
    predicates = torch.randint(0, 46, (num_triples,))  # Predicate types
    edges = torch.randint(0, num_objects, (num_triples, 2))  # Edge list
    obj_to_img = torch.tensor([0,0,0,0,0,1,1,1,1,1])  # Image assignment
    triples_to_img = torch.tensor([0]*6 + [1]*6)  # Triple assignment
    triples_full = torch.cat([
        edges[:, 0:1], 
        predicates.unsqueeze(1), 
        edges[:, 1:2]
    ], dim=1)  # (T, 3)
    
    # Step 2: Get initial embeddings
    # h^(0), r^(0)
    obj_vecs, pred_vecs = embeddings.forward(objs, predicates)
    print(f"Initial embeddings:")
    print(f"  obj_vecs: {obj_vecs.shape}")  # (10, 512)
    print(f"  pred_vecs: {pred_vecs.shape}")  # (12, 512)
    
    # Step 3: Apply graph convolutions (5 layers)
    # h^(0) → h^(1) → ... → h^(5)
    graph_net = AnnotatedGraphTripleConvNet(
        input_dim=512, num_layers=5, hidden_dim=512
    )
    obj_vecs_final, pred_vecs_final = graph_net.forward(obj_vecs, pred_vecs, edges)
    print(f"\nAfter 5 layers of message passing:")
    print(f"  obj_vecs_final: {obj_vecs_final.shape}")  # (10, 512)
    print(f"  pred_vecs_final: {pred_vecs_final.shape}")  # (12, 512)
    
    # Step 4: Extract global features
    # h_global = f(h^(5), r^(5))
    global_fea = AnnotatedFeatureExtraction.extract_global_features(
        obj_vecs_final, pred_vecs_final, obj_to_img, triples_to_img
    )
    print(f"\nGlobal features:")
    print(f"  global_fea: {global_fea.shape}")  # (2, 512)
    
    # Step 5: Extract local features
    # t_local = [h_s || r || h_o] for each triple
    local_fea = AnnotatedFeatureExtraction.extract_local_features(
        obj_vecs_final, pred_vecs_final, triples_full, triples_to_img,
        max_triples_per_image=15, batch_size=batch_size
    )
    print(f"\nLocal features:")
    print(f"  local_fea: {local_fea.shape}")  # (2, 15, 1536)
    
    # Step 6: Prepare conditioning for UNet
    # Context = [local_features (15 tokens), global_features (1 token)]
    global_fea_expanded = global_fea.unsqueeze(1)  # (2, 1, 512)
    
    # Project local features to match dimension
    local_projection = nn.Linear(1536, 512)
    local_fea_proj = local_projection(local_fea)  # (2, 15, 512)
    
    # Concatenate for cross-attention
    context = torch.cat([local_fea_proj, global_fea_expanded], dim=1)
    print(f"\nFinal conditioning context:")
    print(f"  context: {context.shape}")  # (2, 16, 512)
    print(f"  - 15 local triple tokens + 1 global token")
    print(f"  - Each token is 512-dimensional")
    
    return context


if __name__ == "__main__":
    print("="*70)
    print("Scene Graph Message Passing: Annotated Example")
    print("="*70)
    context = complete_pipeline_example()
    print("\n" + "="*70)
    print("Pipeline complete. Context ready for UNet cross-attention.")
    print("="*70)
