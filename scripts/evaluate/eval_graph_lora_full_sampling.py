#!/usr/bin/env python3
"""Full-sampling evaluation for graph-conditioned Flux LoRA (plan_lora.md §7).

Runs full sampling (inference-time integrator) and measures graph sensitivity using:
- ConceptAttention saliency from Flux double blocks 7–12 (configurable)
- Frozen 24-way relation classifier C

Reports (base and tuned):
- Accuracy on g+
- Mean margin E[logit(g+)_y - logit(g-)_y]
- Win rate P(logit(g+)_y > logit(g-)_y)

Note: this runs on the SGDiff-supported 16/24 predicate subset (filters automatically).
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
local_diffusers = REPO_ROOT / "diffusers" / "src"
if local_diffusers.exists() and str(local_diffusers) not in sys.path:
    sys.path.insert(0, str(local_diffusers))

from diffusers import FluxPipeline

from relation_analysis.data.relations import _DEFAULT_PREDICATES, default_predicate_map
from relation_analysis.flux.graph_conditioned_flux import clear_graph_condition, patch_flux_for_graph, set_graph_condition
from relation_analysis.flux.lora import LinearWithLoRA
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.stage_b.concepts import build_concept_inputs
from relation_analysis.stage_b.tracer import compute_saliency, concept_attention_step

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CLASS_NAMES: List[str] = sorted([spec.name for spec in _DEFAULT_PREDICATES])
CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _dtype_from_str(s: str):
    s = s.lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _inject_lora_into_blocks(transformer, block_indices: Iterable[int], rank: int, alpha: float):
    for idx in block_indices:
        if idx < 0 or idx >= len(transformer.transformer_blocks):
            continue
        block = transformer.transformer_blocks[idx]
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for name in ("to_q", "to_k", "to_v"):
            proj = getattr(attn, name, None)
            if isinstance(proj, nn.Linear) and not isinstance(proj, LinearWithLoRA):
                setattr(attn, name, LinearWithLoRA(proj, rank=rank, alpha=alpha))
        to_out = getattr(attn, "to_out", None)
        if isinstance(to_out, nn.ModuleList) and len(to_out) > 0 and isinstance(to_out[0], nn.Linear):
            if not isinstance(to_out[0], LinearWithLoRA):
                to_out[0] = LinearWithLoRA(to_out[0], rank=rank, alpha=alpha)


def _extract_checkpoint_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "transformer_state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def _extract_checkpoint_config(obj) -> dict:
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        return obj["config"]
    return {}


@dataclass
class StageAItem:
    index: int
    prompt: str
    concepts: List[str]
    subject: str
    predicate: str  # canonical predicate label
    object: str


def load_stage_a_items(path: Path, max_examples: int) -> List[StageAItem]:
    items: List[StageAItem] = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if max_examples and idx >= max_examples:
                break
            obj = json.loads(line)
            triple = obj["triple"]
            items.append(
                StageAItem(
                    index=idx,
                    prompt=obj["prompt"],
                    concepts=list(obj["concepts"]),
                    subject=triple["subject"],
                    predicate=triple["predicate"],
                    object=triple["object"],
                )
            )
    return items


class _WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return torch.add(shortcut, out)


class _WRNNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, dropRate: float = 0.0):
        super().__init__()
        layers = [
            _WRNBasicBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate)
            for i in range(int(nb_layers))
        ]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, num_classes: int = 24, widen_factor: int = 8, dropRate: float = 0.3, in_channels: int = 3):
        super().__init__()
        nC = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        self.conv1 = nn.Conv2d(in_channels, nC[0], 3, padding=1, bias=False)
        self.block1 = _WRNNetworkBlock(n, nC[0], nC[1], stride=1, dropRate=dropRate)
        self.block2 = _WRNNetworkBlock(n, nC[1], nC[2], stride=2, dropRate=dropRate)
        self.block3 = _WRNNetworkBlock(n, nC[2], nC[3], stride=2, dropRate=dropRate)
        self.bn1 = nn.BatchNorm2d(nC[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nC[3], num_classes)
        self.nChannels = nC[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def load_frozen_classifier(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = WideResNet(depth=28, num_classes=24, widen_factor=8, dropRate=0.3, in_channels=3).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Classifier checkpoint is not a state_dict dict: {type(state)}")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class AveragingConceptAttentionCollector:
    """Records mean saliency across selected blocks and diffusion steps."""

    def __init__(self, transformer, concept_states0: torch.Tensor, record_layers: Sequence[int]):
        self.transformer = transformer
        self.initial_concept_states = concept_states0
        self.concept_states = concept_states0
        self.record_layers = set(int(x) for x in record_layers)
        self._orig_forwards: List = []
        self.saliency_sum: Optional[torch.Tensor] = None
        self.count: int = 0

    def __enter__(self):
        for idx, block in enumerate(self.transformer.transformer_blocks):
            orig = block.forward
            self._orig_forwards.append(orig)

            def wrapper(b=block, i=idx, orig_forward=orig):
                def _wrapped(*args, **kwargs):
                    # Reset per transformer forward call (new diffusion step).
                    if i == 0:
                        self.concept_states = self.initial_concept_states.clone()
                    enc_out, img_out = orig_forward(*args, **kwargs)
                    concept_out = concept_attention_step(b, self.concept_states, img_out)
                    if i in self.record_layers:
                        sal = compute_saliency(img_out, concept_out).float()
                        self.saliency_sum = sal if self.saliency_sum is None else (self.saliency_sum + sal)
                        self.count += 1
                    self.concept_states = concept_out
                    return enc_out, img_out

                return _wrapped

            block.forward = wrapper()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for block, orig in zip(self.transformer.transformer_blocks, self._orig_forwards):
            block.forward = orig
        return False

    def mean_saliency(self) -> torch.Tensor:
        if self.saliency_sum is None or self.count == 0:
            raise RuntimeError("No saliency recorded; check record_layers and transformer structure.")
        return self.saliency_sum / float(self.count)


def saliency_to_classifier_input(saliency: torch.Tensor, h_tokens: int, w_tokens: int, out_size: int) -> torch.Tensor:
    if saliency.ndim != 3:
        raise ValueError(f"Expected saliency (B,3,tokens), got {tuple(saliency.shape)}")
    b, c, t = saliency.shape
    if t != h_tokens * w_tokens:
        raise ValueError(f"Token mismatch: saliency tokens={t} expected={h_tokens*w_tokens} (h={h_tokens},w={w_tokens})")
    sal = torch.softmax(saliency.float(), dim=-1).view(b, c, h_tokens, w_tokens)
    if h_tokens != out_size or w_tokens != out_size:
        sal = F.interpolate(sal, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return sal


def load_pipeline(model_id: str, torch_dtype, device: torch.device, cpu_offload: bool) -> FluxPipeline:
    cache_dir = PROJECT_ROOT / "hf_flux_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_tok(subfolder: str):
        return AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, use_fast=False, cache_dir=cache_dir)

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.tokenizer = _load_tok("tokenizer")
    pipe.tokenizer_2 = _load_tok("tokenizer_2")
    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        # Avoid moving the huge text encoders to GPU; they can OOM a 24GB card. We keep them on CPU and provide
        # precomputed prompt embeddings to `pipe(...)` so generation uses only the transformer+VAE on GPU.
        pipe.transformer.to(device, dtype=torch_dtype)
        pipe.vae.to(device, dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_raw_predicate_map(vocab_path: Path) -> Dict[str, List[str]]:
    import json as _json

    vocab = _json.loads(vocab_path.read_text())
    predicate_map = default_predicate_map()
    canon_to_raw: Dict[str, List[str]] = {}
    for p in vocab.get("pred_idx_to_name", []):
        if not isinstance(p, str) or p.startswith("__"):
            continue
        spec = predicate_map.canonicalize(p)
        if spec is None:
            continue
        canon_to_raw.setdefault(spec.name, []).append(p)
    for k in canon_to_raw:
        canon_to_raw[k] = sorted(set(canon_to_raw[k]))
    return canon_to_raw


def make_grid(img00: Image.Image, img01: Image.Image, img10: Image.Image, img11: Image.Image) -> Image.Image:
    w, h = img00.size
    grid = Image.new("RGB", (2 * w, 2 * h))
    grid.paste(img00, (0, 0))
    grid.paste(img01, (w, 0))
    grid.paste(img10, (0, h))
    grid.paste(img11, (w, h))
    return grid


def parse_args():
    p = argparse.ArgumentParser(description="Full-sampling eval for graph-conditioned Flux LoRA (plan_lora.md §7).")
    p.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "stage_a" / "vg_stage_a.jsonl")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "graph_lora_eval_full_sampling")
    p.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds.")
    p.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-panels", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--graph-mode", type=str, choices=["token", "temb"], required=True)
    p.add_argument("--block-start", type=int, default=7)
    p.add_argument("--block-end", type=int, default=13)

    p.add_argument("--lora-checkpoint", type=Path, required=True)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=float, default=None)

    p.add_argument("--classifier-ckpt", type=Path, default=PROJECT_ROOT / "scripts" / "runs" / "wrn_mode-saliency_norm-none_smooth-0_abl-none.pt")
    p.add_argument("--classifier-in-size", type=int, default=32)

    p.add_argument("--vocab-path", type=Path, default=REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json")
    p.add_argument("--cgip-ckpt", type=Path, default=REPO_ROOT / "SGDiff" / "pretrained" / "sip_vg.pt")
    p.add_argument("--graph-encoder-device", type=str, default="cpu")

    p.add_argument("--hard-negative-topk", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaking/random negatives.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    device = torch.device(args.device)
    torch_dtype = _dtype_from_str(args.dtype)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "base").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "tuned").mkdir(parents=True, exist_ok=True)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    # Load Stage A examples (prompt + canonical triple).
    items = load_stage_a_items(args.input, args.max_examples)
    if not items:
        raise RuntimeError(f"No examples found in {args.input}")

    # Build canonical->raw predicate map for SGDiff graph encoder.
    canon_to_raw = build_raw_predicate_map(args.vocab_path)
    supported = sorted(set(canon_to_raw.keys()) & set(CLASS_NAME_TO_ID.keys()))
    missing = sorted([c for c in CLASS_NAMES if c not in supported])
    print(f"Canonical predicates: {len(CLASS_NAMES)}")
    print(f"SGDiff-supported predicates: {len(supported)}")
    if missing:
        print(f"Unsupported under SGDiff vocab ({len(missing)}): {missing}")

    # Prepare graph encoder (legacy SGDiff CGIP).
    graph_encoder = SGDiffGraphEncoder(
        vocab_path=args.vocab_path,
        ckpt_path=args.cgip_ckpt,
        device=args.graph_encoder_device,
    )

    # Load pipeline and patch for graph conditioning.
    pipe = load_pipeline(args.model_id, torch_dtype=torch_dtype, device=device, cpu_offload=bool(args.cpu_offload))
    transformer = patch_flux_for_graph(pipe.transformer, mode=args.graph_mode, block_range=range(args.block_start, args.block_end))

    # Load tuned checkpoint config to infer LoRA params when not provided.
    ckpt_obj = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = _extract_checkpoint_config(ckpt_obj)
    lora_rank = args.lora_rank if args.lora_rank is not None else int(ckpt_cfg.get("lora_rank", 16))
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else float(ckpt_cfg.get("lora_alpha", float(lora_rank)))

    # Inject LoRA modules now so "base" == LoRA no-op (B=0) and "tuned" just loads weights.
    _inject_lora_into_blocks(transformer, range(args.block_start, args.block_end), rank=lora_rank, alpha=lora_alpha)

    classifier = load_frozen_classifier(args.classifier_ckpt, device=device)

    # Precompute prompt + concept states per example.
    concept_states0: Dict[int, torch.Tensor] = {}
    prompt_embeds0: Dict[int, torch.Tensor] = {}
    pooled_prompt_embeds0: Dict[int, torch.Tensor] = {}
    text_device = next(pipe.text_encoder.parameters()).device if pipe.text_encoder is not None else torch.device("cpu")
    t5_device = next(pipe.text_encoder_2.parameters()).device if pipe.text_encoder_2 is not None else torch.device("cpu")
    for item in items:
        with torch.no_grad():
            p_emb, p_pool, _txt_ids = pipe.encode_prompt(
                prompt=item.prompt,
                prompt_2=None,
                device=text_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
        prompt_embeds0[item.index] = p_emb.to(device=device, dtype=torch_dtype)
        pooled_prompt_embeds0[item.index] = p_pool.to(device=device, dtype=torch_dtype)

        embeds = build_concept_inputs(
            tokenizer=pipe.tokenizer_2,
            text_encoder=pipe.text_encoder_2,
            concepts=item.concepts,
            device=t5_device,
            dtype=torch_dtype,
        ).to(device=device, dtype=torch_dtype)
        concept_states0[item.index] = transformer.context_embedder(embeds)

    directional = {"left of", "right of", "above", "below", "in front of", "behind"}
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)

    def choose_raw_predicate(canon: str) -> str:
        raw = canon_to_raw.get(canon)
        if not raw:
            raise KeyError(f"Canonical predicate not in SGDiff vocab mapping: {canon}")
        return raw[0]

    def pick_hard_negative_canonical(true_canon: str) -> Optional[str]:
        if args.hard_negative_topk <= 0:
            return None
        true_id = CLASS_NAME_TO_ID.get(true_canon)
        if true_id is None:
            return None
        row = confusion[true_id].clone()
        row[true_id] = 0
        if int(row.sum().item()) == 0:
            return None
        k = min(int(args.hard_negative_topk), row.numel())
        top_ids = torch.topk(row, k=k).indices.tolist()
        candidates = [CLASS_NAMES[i] for i in top_ids if CLASS_NAMES[i] in supported and CLASS_NAMES[i] != true_canon]
        return random.choice(candidates) if candidates else None

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds parsed to empty list")

    record_layers = range(args.block_start, args.block_end)
    h_tokens = args.height // 16
    w_tokens = args.width // 16

    # Fixed g- choices per example (computed during base pass).
    neg_triplets_raw: Dict[int, Tuple[str, str, str]] = {}
    neg_pred_canon: Dict[int, str] = {}

    def run_one(
        item: StageAItem,
        triplet_raw: Tuple[str, str, str],
        seed: int,
        phase: str,
        which: str,
    ) -> Tuple[torch.Tensor, Optional[Path]]:
        with torch.no_grad():
            graph_local, graph_global = graph_encoder.encode_batch([triplet_raw])
            graph_local = graph_local.to(device=device, dtype=torch_dtype)
            graph_global = graph_global.to(device=device, dtype=torch_dtype)
            set_graph_condition(transformer, graph_local=graph_local, graph_global=graph_global)
            gen = torch.Generator(device=device).manual_seed(seed)
            with AveragingConceptAttentionCollector(transformer, concept_states0[item.index], record_layers=record_layers) as tracer:
                out_type = "pil" if args.save_images else "latent"
                out = pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds0[item.index],
                    pooled_prompt_embeds=pooled_prompt_embeds0[item.index],
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    height=args.height,
                    width=args.width,
                    output_type=out_type,
                )
            clear_graph_condition(transformer)
            sal = tracer.mean_saliency()

            x = saliency_to_classifier_input(sal, h_tokens=h_tokens, w_tokens=w_tokens, out_size=args.classifier_in_size)
            logits = classifier(x)

            img_path = None
            if args.save_images:
                img = out.images[0]
                img_path = out_dir / "images" / phase / f"ex{item.index:05d}_seed{seed}_{which}.png"
                img.save(img_path)
            return logits.detach().cpu(), img_path

    results: List[dict] = []

    # -------------------------
    # Base pass (no-op LoRA).
    # -------------------------
    for item in items:
        canon = item.predicate
        if canon not in CLASS_NAME_TO_ID:
            continue
        if canon not in supported:
            continue
        label = int(CLASS_NAME_TO_ID[canon])
        raw_pred_pos = choose_raw_predicate(canon)
        triplet_pos_raw = (item.subject, raw_pred_pos, item.object)

        logits_pos_by_seed: Dict[int, torch.Tensor] = {}
        pred_pos_by_seed: Dict[int, int] = {}
        for seed in seeds:
            logits_pos, _ = run_one(item, triplet_pos_raw, seed=seed, phase="base", which="pos")
            pred_pos = int(logits_pos.argmax(dim=1).item())
            confusion[label, pred_pos] += 1
            logits_pos_by_seed[int(seed)] = logits_pos
            pred_pos_by_seed[int(seed)] = pred_pos

        if canon in directional:
            triplet_neg_raw = (item.object, raw_pred_pos, item.subject)
            neg_canon = canon
        else:
            neg_canon = pick_hard_negative_canonical(canon)
            if neg_canon is None:
                neg_canon = random.choice([c for c in supported if c != canon])
            raw_pred_neg = choose_raw_predicate(neg_canon)
            triplet_neg_raw = (item.subject, raw_pred_neg, item.object)

        neg_triplets_raw[item.index] = triplet_neg_raw
        neg_pred_canon[item.index] = neg_canon

        for seed in seeds:
            logits_pos = logits_pos_by_seed[int(seed)]
            pred_pos = pred_pos_by_seed[int(seed)]
            logits_neg, _ = run_one(item, triplet_neg_raw, seed=seed, phase="base", which="neg")

            score_pos = float(logits_pos[0, label].item())
            score_neg = float(logits_neg[0, label].item())
            results.append(
                {
                    "example": item.index,
                    "seed": seed,
                    "y": label,
                    "predicate": canon,
                    "negative_predicate": neg_canon,
                    "phase": "base",
                    "pred_pos": pred_pos,
                    "acc_pos": int(pred_pos == label),
                    "score_pos": score_pos,
                    "score_neg": score_neg,
                    "margin": score_pos - score_neg,
                    "win": int(score_pos > score_neg),
                }
            )

    # -------------------------
    # Tuned pass (load ckpt).
    # -------------------------
    state_dict = _extract_checkpoint_state_dict(ckpt_obj)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] Loaded tuned checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

    for item in items:
        canon = item.predicate
        if canon not in CLASS_NAME_TO_ID:
            continue
        if canon not in supported:
            continue
        label = int(CLASS_NAME_TO_ID[canon])
        raw_pred_pos = choose_raw_predicate(canon)
        triplet_pos_raw = (item.subject, raw_pred_pos, item.object)

        triplet_neg_raw = neg_triplets_raw.get(item.index)
        neg_canon = neg_pred_canon.get(item.index)
        if triplet_neg_raw is None or neg_canon is None:
            continue

        for seed in seeds:
            logits_pos, _ = run_one(item, triplet_pos_raw, seed=seed, phase="tuned", which="pos")
            logits_neg, _ = run_one(item, triplet_neg_raw, seed=seed, phase="tuned", which="neg")
            pred_pos = int(logits_pos.argmax(dim=1).item())
            score_pos = float(logits_pos[0, label].item())
            score_neg = float(logits_neg[0, label].item())
            results.append(
                {
                    "example": item.index,
                    "seed": seed,
                    "y": label,
                    "predicate": canon,
                    "negative_predicate": neg_canon,
                    "phase": "tuned",
                    "pred_pos": pred_pos,
                    "acc_pos": int(pred_pos == label),
                    "score_pos": score_pos,
                    "score_neg": score_neg,
                    "margin": score_pos - score_neg,
                    "win": int(score_pos > score_neg),
                }
            )

    # Save raw results
    raw_path = out_dir / "results.jsonl"
    with raw_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    # Aggregate metrics
    def summarize(phase: str) -> dict:
        rows = [r for r in results if r["phase"] == phase]
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "acc_pos": float(sum(r["acc_pos"] for r in rows) / len(rows)),
            "mean_margin": float(sum(r["margin"] for r in rows) / len(rows)),
            "win_rate": float(sum(r["win"] for r in rows) / len(rows)),
        }

    summary = {"base": summarize("base"), "tuned": summarize("tuned")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Summary:")
    print(json.dumps(summary, indent=2))

    # Optional: build 2×2 panels
    if args.save_images and args.make_panels:
        for item in items:
            for seed in seeds:
                base_pos = out_dir / "images" / "base" / f"ex{item.index:05d}_seed{seed}_pos.png"
                base_neg = out_dir / "images" / "base" / f"ex{item.index:05d}_seed{seed}_neg.png"
                tuned_pos = out_dir / "images" / "tuned" / f"ex{item.index:05d}_seed{seed}_pos.png"
                tuned_neg = out_dir / "images" / "tuned" / f"ex{item.index:05d}_seed{seed}_neg.png"
                if not (base_pos.exists() and base_neg.exists() and tuned_pos.exists() and tuned_neg.exists()):
                    continue
                img00 = Image.open(base_pos).convert("RGB")
                img01 = Image.open(base_neg).convert("RGB")
                img10 = Image.open(tuned_pos).convert("RGB")
                img11 = Image.open(tuned_neg).convert("RGB")
                grid = make_grid(img00, img01, img10, img11)
                grid.save(out_dir / "panels" / f"ex{item.index:05d}_seed{seed}.png")


if __name__ == "__main__":
    main()
