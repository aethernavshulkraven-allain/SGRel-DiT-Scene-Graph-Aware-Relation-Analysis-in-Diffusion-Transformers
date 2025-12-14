Got it: **FLUX.1-schnell**, and you are targeting **dual-stream (double) blocks 7–12 out of 0–18**. Diffusers defines this as `num_layers=19` dual-stream blocks (plus 38 single-stream blocks). ([Hugging Face][1])

Below is a **one-night, quick-win plan** that is implementable and gives you a clean before/after story, without masks, using **ConceptAttention saliencies** and your **frozen 24-way relation classifier**. ConceptAttention’s key point is that saliency from **DiT attention output space** is sharp and can be extracted without extra training. ([arXiv][2])

---

## 1) Core change: train without full sampling, evaluate with full sampling

You currently compute saliency on fully sampled images. Keep that for **evaluation**.

For **training**, do **teacher-forced** (single-timestep) updates:

* You have VG images (x) and graphs (g). VG is designed for object relationships/scene graphs, so this is the intended supervision. ([arXiv][3])
* FLUX is a **rectified flow transformer**; training is a per-timestep regression in latent space (flow/velocity style). Full sampling is just the inference-time integrator and is not needed to define (L_{\text{gen}}). ([Hugging Face][4])

This single change is what makes the overnight run feasible: 2 forward passes per batch (for (g^+) and (g^-)), not 2 full generations.

---

## Note on relation classes (SGDiff vocab constraint)

This plan describes a **24-class canonical predicate set**.

In this codebase, the graph encoder is the **legacy SGDiff CGIP encoder**, which is tied to the **VG predicate vocabulary** shipped with SGDiff. That vocabulary does **not** cover all 24 canonical predicates, so the LoRA adaptation experiments are run on the **SGDiff-supported 16-class subset**:

`['above','around/near','behind','below','carrying','eating','hanging from','holding','in','in front of','looking at','on','riding','sitting on','standing on','wearing']`

The missing 8 canonical predicates under SGDiff vocab are:

`['drinking','left of','playing with','pulling','pushing','right of','touching','using']`

We keep the frozen classifier `C` as the original 24-way model, but training/evaluation only samples examples whose predicates are in the 16 supported classes.

---

## 2) What you train and what you freeze

### Trainable

1. **LoRA** on **double blocks 7–12 only**, attention projections only:

* `attn.to_q, attn.to_k, attn.to_v, attn.to_out.0` ([Hugging Face][5])

1. Your **graph→token projection** layers (small MLP/linear).

### Frozen

* Base FLUX weights outside those LoRA adapters
* Text encoders + VAE (standard in diffusers FLUX LoRA training setups) ([Hugging Face][6])
* Your relation classifier (C)

Implementation detail (robust):

* Attach LoRA to attention modules normally, then set `requires_grad=False` for all LoRA params whose name does **not** include `double_blocks.(7|8|9|10|11|12)`.

---

## 3) Losses (your 3-term objective is enough)

Yes, the following is enough for a quick win:

$$
L = L_{\text{gen}} + \lambda L_{\text{rel-rank}} + \alpha L_{\text{gen-rel}}
$$

### 3.1 $L_{\text{gen}}$: generative anchor (teacher-forced)

Per example $(x, c, g^+)$:

1. VAE encode image: $z_0 = \mathrm{Enc}(x)$
2. Sample timestep $t$ and noise / interpolation rule → get $z_t$
3. Run FLUX once with conditioning $(c, g^+)$ at $t$
4. Compute the native rectified-flow / flow-matching regression MSE in latent space.

This keeps the model “image-valid” while you push relations. ([Hugging Face][4])

### 3.2 $L_{\text{rel-rank}}$: contrastive relation loss using your frozen classifier (main win)

This is where your setup is ideal.

For each VG triplet with true relation class $y \in \{1,\dots,24\}$:

* Build positive graph $g^+$ (GT edge).
* Build negative graph $g^-$ (corrupted edge), see Section 4.
* Use the **same** $(z_t,t)$ and same prompt $c$ for both.

Compute ConceptAttention saliencies from **double blocks 7–12**:

* $(S_s, S_p, S_o)$ for subject / predicate / object concept tokens (your existing extraction). ([arXiv][2])
  Feed to frozen classifier:
* $\ell^+ = C(S_s^+, S_p^+, S_o^+)$
* $\ell^- = C(S_s^-, S_p^-, S_o^-)$

Margin ranking:

$$
L_{\text{rel-rank}}=\max(0,; m - \ell^+_y + \ell^-_y)
$$

Why ranking beats CE here:

* It forces **“correct graph > corrupted graph”** instead of just “predict y”, so it directly optimizes graph sensitivity.
* This aligns with relation-centric LoRA work (DreamRelation explicitly uses relational contrastive loss and motivates LoRA placement via Q/K/V roles). ([CVF Open Access][7])

### 3.3 $L_{\text{gen-rel}}$: saliency-weighted generative loss (no masks)

Goal: make the generative gradient budget focus where the relation token is “active”, without external masks.

* Build a weight map from predicate saliency under $g^+$:

  * $W = \mathrm{detach}(\mathrm{normalize}(\sigma(S_p^+)))$
* Reweight the per-patch latent regression residual:

  $$
  L_{\text{gen-rel}} = | W \odot (\text{pred} - \text{target}) |^2
  $$

Critical: **detach $W$**. Otherwise the model can lower loss by changing saliency rather than improving generation.

---

## 4) Negative graph design (this decides whether you see a gain overnight)

You said you have “all types” of relations in the 24 classes. Use a **mixture of negative types**.

### 4.1 Directional predicates: swap (fast, strong)

For predicates like:

* left-of / right-of
* above / below
* in-front-of / behind

Make $g^- = (o, p, s)$. You still evaluate $\ell^-_y$ for the original class $y$, so the model is penalized if it continues to support the original relation under the swapped graph.

### 4.2 Non-directional predicates: predicate replacement (hard negatives)

For predicates like:

* holding, wearing, looking-at, on, under, inside-of (many are asymmetric but not “opposites” via swap)

Do:

* Keep objects fixed: $(s, p', o)$
* Choose $p'$ as a **hard negative**. Best quick method:

  * precompute a confusion matrix of your frozen classifier on baseline (no LoRA)
  * for each true $p$, pick top-1 or top-2 most-confused $p'$

This amplifies gradient signal compared to random negatives.

---

## 5) Training loop (exact sequence per step)

Assume batch of B triplets.

### Precompute / cache (big speedup)

* Cache VAE latents $z_0$ for all train images (disk or RAM).
* Cache text embeddings for prompts if your prompt set is stable.
  Diffusers FLUX training recipes emphasize tricks like latent caching to make runs practical. ([Hugging Face][6])

### Per iteration

1. Load cached $z_0$, prompt embeddings, graph $g^+$, label $y$
2. Sample $t$ (recommend: uniform in an interval that corresponds to “mid/low noise”, see 6.2)
3. Build $z_t$
4. Forward pass with $g^+$:

   * compute $L_{\text{gen}}$
   * extract $(S_s^+,S_p^+,S_o^+)$ (blocks 7–12)
   * logits $\ell^+$
5. Forward pass with $g^-$ (same $(z_t,t,c)$):

   * extract $(S_s^-,S_p^-,S_o^-)$
   * logits $\ell^-$
6. Compute:

   * $L_{\text{rel-rank}}$
   * $L_{\text{gen-rel}}$ using detached $W(S_p^+)$
7. Backprop, update only:

   * LoRA in blocks 7–12 (attn projections)
   * graph projector

### Logging every N steps

* mean $L_{\text{gen}}$, $L_{\text{rel-rank}}$
* mean margin $\Delta = \ell^+_y - \ell^-_y$
* relation accuracy on $g^+$ under frozen classifier (optional, but margin is more diagnostic)

---

## 6) Hyperparameters that usually “just work” in small runs

### 6.1 Loss weights

Start simple, one run:

* margin $m = 1.0$
* $\lambda = 0.3$
* $\alpha = 0.0$ for the first ~30–50% of steps, then $\alpha = 0.1$

If you only get one run, keep $\alpha=0.1$ from the start.

### 6.2 Timestep sampling

Since you care about **final-image relations**, prefer timesteps closer to the “low-noise / late” region (but not only at the very end, which can reduce learning signal). Practically:

* sample $t$ from a truncated range (example: upper 50% of denoising trajectory)
  This keeps saliency more correlated with visible structure while still giving gradients.

### 6.3 LoRA rank

* rank 4 or 8 for a quick run (higher rank increases capacity but also risk of drift).
  Diffusers FLUX LoRA docs and recipes commonly target attention modules specifically. ([Hugging Face][5])

---

## 7) Evaluation: what you generate and show

Keep evaluation as you currently do: **full sampling** with fixed seeds.

### 7.1 Fixed test suite

Pick ~100 triplets (or fewer if compute tight), and fix:

* prompt $c$
* seed(s): 3 seeds per prompt is enough
* graphs: $g^+$ and $g^-$

### 7.2 Metrics (quantitative)

For each sample:

* generate image with $g^+$ and $g^-$
* compute ConceptAttention saliencies on the generated image (same blocks 7–12) ([arXiv][2])
* frozen classifier → logits $\ell^+,\ell^-$

Report:

1. **Accuracy on $g^+$** (relation correctness proxy)
2. **Mean margin** $\mathbb{E}[\ell^+_y - \ell^-_y]$
3. **Win rate**: $\Pr(\ell^+_y > \ell^-_y)$

Margin + win rate are the cleanest “graph sensitivity” measures.

### 7.3 Visual panel (qualitative)

For each prompt/seed, make a 2×2:

* Base + $g^+$
* Base + $g^-$
* Tuned + $g^+$
* Tuned + $g^-$

Pick 8–12 best illustrative cases.

---

## Why this plan fits your constraints

* No masks.
* Uses your existing ConceptAttention saliency pipeline (attention-output-space saliency). ([arXiv][2])
* Uses a frozen classifier, avoiding “classifier adapts to LoRA”.
* Only modifies **double blocks 7–12**, leaving the rest untouched.
* Uses contrastive graph supervision, which matches relation-centric customization literature. ([CVF Open Access][7])
* Stays consistent with FLUX’s dual-stream architecture and LoRA targeting conventions in diffusers. ([Hugging Face][1])

If you want, paste your current negative-graph construction (swap vs predicate replacement logic). That is the single highest leverage knob for getting a visible margin gain overnight.

[1]: https://huggingface.co/docs/diffusers/api/models/flux_transformer
[2]: https://arxiv.org/abs/2502.04320?utm_source=chatgpt.com
[3]: https://arxiv.org/abs/1602.07332?utm_source=chatgpt.com
[4]: https://huggingface.co/black-forest-labs/FLUX.1-schnell?utm_source=chatgpt.com
[5]: https://huggingface.co/spaces/multimodalart/Cosmos-Predict2-2B/blob/main/diffusers_repo/examples/dreambooth/README_flux.md?utm_source=chatgpt.com
[6]: https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora?utm_source=chatgpt.com
[7]: https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_DreamRelation_Relation-Centric_Video_Customization_ICCV_2025_paper.pdf?utm_source=chatgpt.com
