
## 1. What ConceptAttention actually gives you (in SD3 terms)

ConceptAttention is designed exactly for **multi-modal DiTs like Flux and SD3** (they explicitly mention both and use the same rectified-flow formulation as SD3). ([arXiv][1])

Key ideas you want to reuse:

1. **Multi-modal DiT anatomy**

   * Image tokens + text tokens go through **MMAttn layers** (multi-modal attention).
   * Image/text have separate Q/K/V projections but share the attention operation.
   * There are **separate residual streams** for image and text, modulated by time and a global CLIP vector (adaptive LayerNorm). ([arXiv][1])
2. **Concept tokens added as a third stream**

   * You introduce extra "concept tokens" for words like "dog", "bucket", "person", encoded via T5 (same as prompt encoding). ([arXiv][1])
   * For each DiT layer $\ell$, you:

     * Layer-norm concept embeddings.
     * Project them with **the same text projection matrices** (Q/K/V) used in the MMAttn layer. ([arXiv][1])
3. **One-directional update: concepts read from image, but do not affect generation**

   * You concatenate concept K/V with image K/V, and update **only concept tokens** with attention over [image + concepts].
   * Image and prompt tokens continue to do their usual self-attention among themselves, completely ignoring the concept tokens. ([arXiv][1])
   * This means:

     * You get a **full concept residual stream** updated layer by layer.
     * You do not change the image generation at all.
4. **Saliency in attention output space**

   * For each layer, you now have:

     * image output vectors $o^{\text{img}}_{\ell,i}$ (one per image token),
     * concept output vectors $o^{\text{concept}}_{\ell,c}$ (one per concept).
   * They show that a **simple dot product** between these vectors gives very sharp, interpretable saliency maps:

     $$
     S_{\ell,c}(i) = \langle o^{\text{img}}_{\ell,i}, o^{\text{concept}}_{\ell,c} \rangle
     $$

     which is much better than cross-attention maps. ([arXiv][1])
   * You can aggregate across layers (mean, max) to get a final map, or inspect layerwise to see **where** a concept becomes well localized. ([arXiv][1])

So ConceptAttention turns each MMAttn layer of SD3 into:

* the usual image/text computation (unchanged), plus
* a side-channel concept stream that lets you read out, per layer, **where each concept lives spatially**.

That is exactly what you need to decide "which layers encode structure".

---

## 2. Where this plugs into your current work

You already have:

* SD3 DreamBooth / base pipeline running (diffusers SD3).
* SGDiff / graph encoder environment ready.
* A small set of **relation prompts** (e.g. "dog in a bucket", "a person touching a dog") for structural evaluation.

Now you want to replace your earlier "raw attention mass heuristic" with ConceptAttention-style analysis.

At a high level:

1. **Prompts** give you:

   * normal T5-encoded prompt tokens (used by SD3 as usual),
   * plus you choose a small set of **concept words** you care about, e.g.

     * "dog", "bucket", "person", "chair", "cat", etc.
2. **For each SD3 DiT layer during sampling**, you:

   * run the normal MMAttn as usual (no change),
   * in parallel, do the ConceptAttention concept update:

     * concept Q/K/V from text projections,
     * one-directional attention over image+concept K/V,
     * propagate concept residual stream.
3. At the end of each layer, you:

   * take image outputs and concept outputs,
   * compute saliency maps $S_{\ell,c}$ for each concept.
4. From these maps, you derive **per-layer structural scores**, and pick the layers that most "explain" spatial relations.

This gives you a principled, DiT-native alternative to handcrafted structure scores.

---

## 3. Step-by-step: using ConceptAttention for Task (1)

Task (1) =

> "Analyze the generation process of SD3 and attention maps of different blocks to understand which layer contributes the most to structural generation (interactions like looking at, touching, on top of)."

Here is the concrete way to wire it, using what you already have.

### 3.1 Choose prompts and concepts

Reuse the relational prompts we already designed, but now also extract concept words:

Example prompt:

> "a photo of a dog in a bucket"

Concept list:

* `["dog", "bucket"]`

For each prompt:

1. Fix a small set of seeds S (say 0, 1, 2).
2. Fix sampler, steps, CFG, resolution.

You will run SD3 as usual for each (prompt, seed).

### 3.2 Insert ConceptAttention into the SD3 DiT

Architectural wiring (no code, just what needs to happen):

1. **Before the first DiT block**:

   * Take each concept word (e.g. "dog", "bucket"), encode it with **the same T5 encoder** that SD3 uses for prompts. This gives you initial concept embeddings $e^{(0)}_c$. ([arXiv][1])
   * Initialize a **concept residual stream** with these embeddings (size = number of concepts × dim).
2. **At each MMAttn block $\ell$**:
   SD3 already computes:

   * text stream: prompt tokens → modulated → $Q/K/V_{\text{text}}$
   * image stream: latent patches → modulated → $Q/K/V_{\text{img}}$

   You now add:

   * Layer-norm the current concept embeddings $e^{(\ell)}$.
   * Project them with the **text projection matrices** from this layer:

     * $Q^{(\ell)}_{\text{concept}} = W_Q^{(\ell,\text{text})} \cdot \text{LN}(e^{(\ell)})$
     * $K^{(\ell)}_{\text{concept}} = W_K^{(\ell,\text{text})} \cdot \text{LN}(e^{(\ell)})$
     * $V^{(\ell)}_{\text{concept}} = W_V^{(\ell,\text{text})} \cdot \text{LN}(e^{(\ell)})$ ([arXiv][1])
   * Concatenate **image and concept keys/values**:

     * $K^{(\ell)}_{\text{CA}} = [K^{(\ell)}_{\text{img}} \,|\, K^{(\ell)}_{\text{concept}}]$
     * $V^{(\ell)}_{\text{CA}} = [V^{(\ell)}_{\text{img}} \,|\, V^{(\ell)}_{\text{concept}}]$ ([arXiv][1])
   * Compute **ConceptAttention**:

     * concept queries attend over image+concept K/V:
       $$
       ^{(\ell)}_{\text{concept}} = \text{Attn}\big(Q^{(\ell)}_{\text{concept}}, K^{(\ell)}_{\text{CA}}, V^{(\ell)}_{\text{CA}}\big)
       $$

       as in eq. (9) in the paper. ([arXiv][1])
   * Meanwhile, SD3's image and text tokens do their usual attention among **themselves only**, ignoring concept tokens, exactly as in eq. (10). ([arXiv][1])
   * Pass concept outputs through **the same output projection + MLP + adaptive LayerNorm** machinery used for text, updating the concept residual stream $e^{(\ell+1)}$. ([arXiv][1])
3. **Collect outputs per layer**:

   * At each block $\ell$, cache:

     * image output vectors $o^{\text{img}}_{\ell,i}$ (after attention+MLP, per token),
     * concept output vectors $o^{\text{concept}}_{\ell,c}$.

This gives you layerwise concept and image representations without touching how SD3 generates images.

### 3.3 Compute per-layer saliency maps for each concept

For each layer $\ell$ and concept $c$:

1. For each image token $i$, compute:

   $$
   _{\ell,c}(i) = \langle o^{\text{img}}_{\ell,i}, o^{\text{concept}}_{\ell,c} \rangle
   $$

   as in eq. (13) in the paper. ([arXiv][1])
2. Reshape $S_{\ell,c}$ back to the latent grid (H/patch, W/patch) to get a 2D saliency map for that concept at that layer.
3. Optionally normalize per layer (e.g. softmax or min–max per concept) if you want comparable scales.

At this point you can **visualize** these maps for a few prompts to see qualitatively when "dog" and "bucket" become well localized in SD3.

---

## 4. Turn these saliency maps into "structural layer scores"

You ultimately care about **relations between objects**, not just single-object localization. ConceptAttention gives you good single-concept maps; you now build relational metrics on top.

For a given prompt with two concepts A and B (e.g. "dog", "bucket"):

1. For each layer $\ell$, construct masks:

   * Threshold or take top-k% of $S_{\ell,A}$ and $S_{\ell,B}$ to get binary masks $M_{\ell,A}, M_{\ell,B}$ in the latent grid.
2. Compute a **geometric relation score** for the target relation:

   * "A on top of B":

     * get center of mass or bounding box for A and B from masks,
     * define score
       $$
       ^{(\ell)}_{\text{on-top}} = \mathbb{1}\big( y_A < y_B \big) \cdot f(\text{vertical distance}, \text{overlap})
       $$

       (more refined if you like, but simple is fine for ranking layers).
   * "A touching B":

     * compute minimal pixel distance between $M_{\ell,A}$ and $M_{\ell,B}$,
     * score = decreasing function of distance.
   * "A left of B":

     * compare x-coordinates of centers.
3. Aggregate over prompts and seeds:

   * For each layer $\ell$, average the relation scores over:

     * all seeds for a prompt,
     * all prompts in the same relation family.
   * This gives you a **structural importance score**:

     $$
     \text{StructScore}(\ell) = \mathbb{E}_{\text{prompts,seeds}}[R^{(\ell)}_{\text{relation}}]
     $$
4. Rank layers by StructScore and mark:

   * high-score layers → strong structural/relational encoding,
   * low-score layers → often more texture / style than geometry.

This replaces the earlier heuristic "attention mass between object token sets" with a more grounded ConceptAttention-based measure.

---

## 5. How SGDiff fits into this, without doing 3b–3d yet

Even though we are only closing Task (1), your SGDiff setup is already useful conceptually:

* SGDiff's **scene graphs** give you a formal description of relations like `dog IN bucket`, `person TOUCH dog`, `cube ON cube`.
* You can use those graphs to:

  * generate text prompts systematically (node labels → nouns, edge labels → relation phrases),
  * define which concept words to track with ConceptAttention (one per node).

Later, when you go to:

* **3b**: train LoRA on structural blocks, conditioned on a graph embedding,
* **3c**: vary graphs, check structural changes at fixed prompt,
* **3d**: do DreamRelation-style injection,

you already know **which SD3 layers** are best to target, because ConceptAttention gave you a principled structural ranking.

---

## 6. Summary in "wiring" language

To answer your core question "how do we wire this into what we've already done?":

1. Use your existing SD3 pipeline, but wrap the MM-DiT forward pass with:

   * "concept stream" initialization (T5 encodings of key nouns),
   * ConceptAttention update in every MMAttn block, sharing text projections and modulations.
2. During sampling for your relation prompts:

   * log image and concept outputs per block,
   * compute dot-product saliency maps $S_{\ell,c}$ for each concept and layer.
3. From these maps:

   * derive object masks at each layer,
   * compute simple relational geometry scores for each layer,
   * average over prompts/seeds to get StructScore($\ell$).
4. Take the top-k structural layers as **candidates for LoRA and graph conditioning** in later tasks.

If you want, I can next write a compact, task-oriented `conceptattention_for_sd3.md` that you can drop into your repo, with sections: "What to change in the DiT forward," "What to log," "How to compute layer scores," and "How to pick layers for LoRA."

[1]: https://arxiv.org/html/2502.04320v1
