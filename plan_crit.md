## 0. High-level pipeline

We build a **relation-analysis pipeline** with two branches:

1. **Geometric relations branch**
   Use ConceptAttention on SD3 to get saliency maps for concepts, extract **analytic geometric features**, train a small MLP to classify geometric relations, and use per-layer accuracy to identify **structural layers**.

2. **Semantic / non-geometric relations branch**
   For relations that depend on pose, semantics, or interaction (hold, wear, ride, look-at), build a **small diffusion-style relation tower** on top of SD3 features + ConceptAttention concept streams, inspired by Diff-VRD and DIFFUSIONHOI. Use its performance per layer to see where semantic relations live.

Later (not now), use:

* SGDiff / scene-graph encoder to produce **graph embeddings**,
* inject those into SD3 structural layers via **LoRA adapters** to enforce scene graphs.

Below is the step-by-step version.

---

## 1. Stage A – Dataset of triples, prompts, concepts

**A1. Start from Visual Genome triples**

Each annotation gives a triplet
((o_i, r_{ij}, o_j)) = (subject, relation, object).

For each triple:

* keep the **object categories** (o_i, o_j) and **relation label** (r_{ij}) (one of ~24 classes),
* optionally filter to a subset of relations (geometric vs semantic).

VG is standard for VRD / HOI; many VRD / diffusion works use it or its derivatives. ([ACM Digital Library][1])

**A2. Build text prompts**

Turn each triple into a simple prompt for SD3, e.g.

* `"a photo of <o_i> <r_ij_phrase> <o_j> on a plain background"`

Here (r_{ij}) label is discrete (`on`, `under`, `left of`, `holding`, …), while the phrase is its natural-language surface form.

**A3. Define “concept tokens” per example**

For ConceptAttention we need explicit concept list:

* concepts = `[o_i, r_ij, o_j]`
  e.g. `["dog", "on top of", "bucket"]`.

These correspond to specific token positions in the T5 / text encoder.

**A4. Split relations**

For the rest of the pipeline, split relation labels into:

* **geometric** (above, below, left of, right of, in, overlapping, around, near),
* **semantic / HOI-style** (holding, wearing, riding, using, looking at, pushing, pulling …).

This matches how VRD / HOI papers often distinguish spatial from interaction predicates. ([ACM Digital Library][1])

---

## 2. Stage B – Instrument SD3 with ConceptAttention

We follow ConceptAttention for a multi-modal DiT like SD3 or Flux: image tokens + text tokens + an extra **concept stream**. ([arXiv][2])

For each prompt:

**B1. Run SD3 generation**

* Use SD3 (or Flux) rectified-flow backbone: latent → DiT → image. ([arXiv][2])
* Use fixed sampler, steps, and seed for reproducibility.

**B2. ConceptAttention side stream**

At each DiT block (\ell):

1. Encode each concept word with the same T5 encoder as the prompt → initial embeddings (e^{(0)}*{o_i}, e^{(0)}*{r_{ij}}, e^{(0)}_{o_j}).
2. For layer (\ell), layer-norm and project concepts with the **text** Q/K/V matrices of that block (as in ConceptAttention). ([OpenReview][3])
3. Concatenate **image** and **concept** K/V, and run attention where **concept queries attend to image+concept**, but image/text ignore concepts. This yields updated concept outputs at each block.
4. Cache:

   * image token outputs (o^{\text{img}}_{\ell}(x,y)),
   * concept token outputs (o^{\text{concept}}*{\ell}(c)) for (c\in{o_i,r*{ij},o_j}).

**B3. Concept saliency maps**

For each layer (\ell), concept (c), compute saliency map

[
S_{\ell,c}(x,y) = \langle o^{\text{img}}*{\ell}(x,y),, o^{\text{concept}}*{\ell}(c)\rangle
]

as in ConceptAttention; this gives a high-quality spatial map for each concept. ([OpenReview][3])

We can average over timesteps or pick a representative diffusion step so that each map is **indexed by layer only**, since later we care about layers, not time.

---

## 3. Stage C – Geometric relations branch (analytic + MLP)

This branch handles the geometric relations using **analytic features** from saliency maps.

### C1. Choose candidate layers

Pick a small subset of DiT blocks:

* early: e.g. layers 2, 4,
* mid: e.g. layers 8, 12,
* late: e.g. layers 16, 20.

This mirrors what interpretability papers do when probing DiT/UNet at several depths. ([OpenReview][3])

For each chosen block (\ell) you have three saliency maps:

* (S_{\ell, o_i}(x,y)),
* (S_{\ell, r_{ij}}(x,y)) (optional here),
* (S_{\ell, o_j}(x,y)).

### C2. Analytic geometric features per block

Normalize saliencies:

[
\tilde S_{\ell,c}(x,y)=\frac{S_{\ell,c}(x,y)}{\sum_{x,y}S_{\ell,c}(x,y)}.
]

For each block (\ell) and pair ((o_i,o_j)) compute:

* Centers
  ( (x_i,y_i) = \sum_{x,y}(x,y)\tilde S_{\ell,o_i}(x,y) ),
  ( (x_j,y_j) = \sum_{x,y}(x,y)\tilde S_{\ell,o_j}(x,y) ).
* Offsets
  (\Delta x = x_i - x_j), (\Delta y = y_i - y_j).
* Spread / size from second moments or bounding box of high-saliency region.
* Overlap / IoU between the two maps:
  [
  \text{Overlap} = \sum_{x,y}\min\big(\tilde S_{\ell,o_i},\tilde S_{\ell,o_j}\big).
  ]
* Distance between supports (for “touching / near”).

Optionally add a few scalar stats of the relation map (S_{\ell,r_{ij}}), like its overlap with each object map.

Collect them into a low-dimensional feature vector

[
z_{\ell} \in \mathbb{R}^{d} \quad (\text{say } d \approx 10\text{–}20)
]

per example and per block.

### C3. MLP classifier for geometric relations

For geometric relations only:

* Input: (z_{\ell})
* Output: distribution over geometric relation classes (left_of, right_of, on, under, in, overlapping, near, none).
* Loss: cross-entropy with label (r_{ij}^{\text{prompt}}).

Two design options:

1. **One MLP per layer**:
   Train separate small MLPs (f_\ell) on data from block (\ell).
   Compare validation accuracy (Acc_\ell) across layers.

2. **Shared MLP + layer index**:
   Concatenate one-hot or embedding of layer index into (z_\ell) and train a single MLP, then evaluate accuracy restricted to each layer.

In both cases, blocks with higher accuracy on geometric relations are **structural for geometry**.

This is conceptually similar to using diffusion features as backbone for VRD/HOI (as DIFFUSIONHOI does) but now with ConceptAttention-derived geometric features instead of raw UNet features. ([Proceedings NeurIPS][4])

---

## 4. Stage D – Semantic / HOI relations branch (relation diffusion tower)

For semantic relations (holding, wearing, riding, using, looking-at), pure geometry is not enough. Here we borrow from Diff-VRD and DIFFUSIONHOI.

### D1. Relation embedding diffusion head (Diff-VRD-style)

Diff-VRD treats **relation embeddings** as continuous variables and learns a diffusion model in relation space, conditioned on subject/object features. ([arXiv][5])

We adapt this to SD3 + ConceptAttention:

1. For each block (\ell), extract:

   * concept outputs (o^{\text{concept}}*{\ell}(o_i)), (o^{\text{concept}}*{\ell}(o_j)),
   * optionally pooled image features in the saliency region of each object,
   * relation text embedding (T5 embedding of (r_{ij})).

2. Concatenate these to form a conditioning vector

   [
   h_{\ell} = [o^{\text{concept}}*{\ell}(o_i), o^{\text{concept}}*{\ell}(o_j), \text{pooled image features}, \text{relation text embedding}]
   ]

3. Define a **1D diffusion process** over a relation latent (z_t) (small dimension, e.g. 128), conditioned on (h_\ell) as in Diff-VRD:

   * forward: Gaussian noise schedule on (z),
   * reverse: small DiT/MLP predicting noise given ((z_t, t, h_\ell)).

4. Train this tower so that the final denoised (z_0) is close to a **target relation embedding**, for example:

   * the text embedding of the predicate,
   * or a teacher embedding from a pretrained VRD / HOI model.

5. Add a linear classifier on (z_0) to predict the semantic relation class and train with cross-entropy jointly.

DIFFUSIONHOI uses diffusion features to detect HOI by learning relation prompts in embedding space and conditioning generation/detection on them. ([Proceedings NeurIPS][4])
Our tower is similar in spirit but:

* it uses **ConceptAttention-aware SD3 features** as conditioning instead of standard CNN features,
* it aims at **decoding relation labels** for interpretability and layer probing, not for real-time detection.

### D2. Which layers does this tower see?

To probe layer importance, repeat this for different subsets of blocks:

* “early-tower”: use (h_{\ell}) from early layers only (e.g. 2, 4).
* “mid-tower”: use mid layers (8,12).
* “late-tower”: use late layers (16,20).

Each tower has the same architecture but different inputs. Compare their validation accuracy on semantic relations:

* high accuracy for mid-tower ⇒ semantic relations most decodable at mid depth.

This is analogous to “where is knowledge stored” analyses for transformers, but now tailored to relations and diffusion features. ([OpenReview][3])

---

## 5. Stage E – Aggregate layer importance

From both branches we get per-layer scores:

* geometric branch: (Acc^{\text{geom}}_\ell) from the analytic-feature MLPs,
* semantic branch: contribution of each layer to relation tower accuracy (via ablations or tower variants).

We can define an overall **structural importance score** per block

[
\text{StructScore}(\ell) =
\alpha, Acc^{\text{geom}}*\ell +
\beta, Acc^{\text{sem}}*\ell
]

with weights (\alpha,\beta) depending on how much you care about geometric vs semantic relations.

Blocks with highest StructScore are then tagged as **structural relation layers**.

Optionally validate this by a small B-LoRA style experiment: add LoRA only on those blocks and see if you can efficiently edit relations while keeping style unchanged, similar to block-localized editing works. ([OpenReview][3])

---

## 6. Stage F – Storing the dataset for later SGDiff + LoRA

For each example you should persist:

* prompt and VG triple ((o_i,r_{ij},o_j)),
* concept token indices,
* for each chosen block (\ell):

  * saliency maps (S_{\ell,o_i}, S_{\ell,r_{ij}}, S_{\ell,o_j}),
  * analytic features (z_\ell),
  * concept outputs (o^{\text{concept}}_{\ell}(\cdot)),
  * simple graph descriptor (2-node + 1-edge mini-graph).

This “relation analysis dataset” is then the base for the later step:

* use a **graph encoder** (SGDiff-style) to embed richer scene graphs into a vector (g), ([Medium][6])
* feed (g) into the **structural layers identified above** via LoRA modulation (global conditioning or extra tokens),
* train these LoRA adapters so SD3 respects full scene graphs, not just single relations.

We do not implement this now, but the current pipeline is built so that extending to that is natural.

---

[1]: https://dl.acm.org/doi/abs/10.1145/3240508.3240668?utm_source=chatgpt.com "Context-Dependent Diffusion Network for Visual ..."
[2]: https://arxiv.org/pdf/2502.04320?utm_source=chatgpt.com "Diffusion Transformers Learn Highly Interpretable Features"
[3]: https://openreview.net/pdf/0c0d18d37b37dcea402702c826f6d94c4e4b4b4e.pdf?utm_source=chatgpt.com "Diffusion Transformers Learn Highly Interpretable Features"
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2a54def490213ee10631b991c5acc6b5-Abstract-Conference.html?utm_source=chatgpt.com "Human-Object Interaction Detection Collaborated with ..."
[5]: https://arxiv.org/abs/2504.12100?utm_source=chatgpt.com "Generalized Visual Relation Detection with Diffusion Models"
[6]: https://medium.com/digital-mind/diffusion-transformer-and-rectified-flow-for-conditional-image-generation-997075c12e2f?utm_source=chatgpt.com "Diffusion Transformer and Rectified Flow ..."
