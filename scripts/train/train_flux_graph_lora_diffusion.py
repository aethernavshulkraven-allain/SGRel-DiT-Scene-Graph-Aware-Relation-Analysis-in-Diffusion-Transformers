#!/usr/bin/env python3
"""Teacher-forced (single-timestep) graph-conditioned Flux LoRA training.

Key idea (rectified flow / flow matching):
- We do NOT run full multi-step sampling during training.
- Each step samples a random t in [0,1], builds z_t = (1-t) * z0 + t * noise, and regresses
  the flow/velocity target (noise - z0) in latent space.

Optional: contrastive negative-graph training
- For each (x, g+) we can build a corrupted graph g- and do a second forward pass.
- We then encourage the model to fit g+ better than g- on the same (z_t, t).

Note: the canonical predicate set is 24 classes, but the legacy SGDiff VG vocab only supports
16/24 of them. This script automatically filters to the SGDiff-supported subset.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
local_diffusers = REPO_ROOT / "diffusers" / "src"
if local_diffusers.exists() and str(local_diffusers) not in sys.path:
    sys.path.insert(0, str(local_diffusers))

from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.flux.graph_conditioned_flux import patch_flux_for_graph, set_graph_condition
from relation_analysis.data.relations import _DEFAULT_PREDICATES, default_predicate_map
from relation_analysis.prompt_builder import predicate_to_phrase
from relation_analysis.stage_b.tracer import compute_saliency, concept_attention_step

try:
    import transformers
    if not hasattr(transformers, "SiglipImageProcessor"):
        class _StubSiglipImageProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
        transformers.SiglipImageProcessor = _StubSiglipImageProcessor
    if not hasattr(transformers, "SiglipVisionModel"):
        class _StubSiglipVisionModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
        transformers.SiglipVisionModel = _StubSiglipVisionModel
except ImportError:
    pass

from diffusers import FluxPipeline

CLASS_NAMES: List[str] = sorted([spec.name for spec in _DEFAULT_PREDICATES])
CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class TrainConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda"
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = False
    gradient_checkpointing: bool = False  # Disabled: conflicts with graph conditioning
    graph_mode: str = "token"  # token|temb
    block_start: int = 7
    block_end: int = 13
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lr: float = 1e-4
    batch_size: int = 1
    epochs: int = 5
    height: int = 512
    width: int = 512
    num_workers: int = 4
    log_every: int = 10
    val_every: int = 100
    save_every: int = 500
    max_train_samples: int = -1
    max_val_samples: int = 100
    seed: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "graph_flux_lora_diffusion"

    # Timestep sampling (sample t ~ U[t_min, t_max]).
    t_min: float = 0.0
    t_max: float = 1.0
    
    # Dataset paths
    vocab_path: str = str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json")
    train_h5: str = str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "train.h5")
    val_h5: str = str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "val.h5")
    vg_images_dir: str = str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "images")
    cgip_ckpt: str = str(REPO_ROOT / "SGDiff" / "pretrained" / "sip_vg.pt")
    graph_encoder_device: str = "cpu"

    # Loss weights (plan_lora.md §3)
    lambda_rel_rank: float = 0.3
    alpha_gen_rel: float = 0.0
    rel_rank_margin: float = 1.0

    # Frozen relation classifier C for L_rel-rank
    classifier_ckpt: Path = PROJECT_ROOT / "scripts" / "runs" / "wrn_mode-saliency_norm-none_smooth-0_abl-none.pt"
    classifier_in_size: int = 32

    # Negative graph usage (plan_lora.md §4)
    use_negative_graph: bool = True
    hard_negative_topk: int = 2  # 0=random; >0 uses classifier confusion top-k (plan_lora.md §4.2)
    
    # Prompt template
    prompt_template: str = "a photo of {subject} {predicate} {object}"
    max_sequence_length: int = 512

    # Optional speedups (plan_lora.md §5)
    latent_cache_dir: Optional[Path] = None  # if set, caches VAE latents z0 on disk

    # Optional fixed example splits (tiny subsets for quick wins).
    # If provided, training/validation draw from these JSONL files instead of scanning H5.
    train_examples_jsonl: Optional[Path] = None
    val_examples_jsonl: Optional[Path] = None
    dry_run: bool = False


class VGImageDataset(Dataset):
    """Visual Genome dataset that loads actual images with their scene graph triplets."""
    
    def __init__(self, vocab_path: str, h5_path: str, images_dir: str, max_samples: int = -1):
        self.vocab = self._load_vocab(vocab_path)
        self.images_dir = Path(images_dir)
        self.h5_path = str(h5_path)
        self._h5 = None
        self.predicate_map = default_predicate_map()

        self.pred_id_to_canonical: Dict[int, str] = {}
        self.pred_id_to_class: Dict[int, int] = {}
        for pred_id, pred_name in enumerate(self.vocab.get("pred_idx_to_name", [])):
            spec = self.predicate_map.canonicalize(pred_name)
            if spec is None:
                continue
            class_id = CLASS_NAME_TO_ID.get(spec.name)
            if class_id is None:
                continue
            self.pred_id_to_canonical[int(pred_id)] = spec.name
            self.pred_id_to_class[int(pred_id)] = int(class_id)

        # Load lightweight indices in the parent process, then re-open H5 per worker.
        with h5py.File(self.h5_path, "r") as h5:
            self.rels_per_image = h5["relationships_per_image"][()]
            raw_paths = h5["image_paths"][()]
            rel_preds = h5["relationship_predicates"][()]
        self.image_paths = [
            p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in raw_paths
        ]
        
        # Keep only images that contain at least one predicate in the canonical 24-class map
        # (plan_lora.md assumes 24-way relation supervision).
        is_mappable = np.zeros(len(self.vocab.get("pred_idx_to_name", [])), dtype=bool)
        for pred_id in self.pred_id_to_class.keys():
            if 0 <= pred_id < is_mappable.shape[0]:
                is_mappable[pred_id] = True
        slot_mask = np.arange(rel_preds.shape[1])[None, :] < self.rels_per_image[:, None]
        mappable_mask = is_mappable[rel_preds] & slot_mask
        valid = mappable_mask.any(axis=1)
        self.valid_indices = np.nonzero(valid)[0].tolist()
        
        if max_samples > 0:
            self.valid_indices = self.valid_indices[:max_samples]

    def _require_h5(self):
        if self._h5 is None:
            h5 = h5py.File(self.h5_path, "r")
            self._h5 = h5
            self.relationship_subjects = h5["relationship_subjects"]
            self.relationship_objects = h5["relationship_objects"]
            self.relationship_predicates = h5["relationship_predicates"]
            self.object_names_h5 = h5["object_names"]

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
            
    def _load_vocab(self, path: str):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        self._require_h5()
        img_idx = self.valid_indices[idx]
        
        # Get a random relationship for this image that maps to one of the 24 canonical predicates.
        count = int(self.rels_per_image[img_idx])
        candidates = []
        for rel_idx in range(count):
            p_id = int(self.relationship_predicates[img_idx, rel_idx])
            if p_id in self.pred_id_to_class:
                candidates.append(rel_idx)
        if not candidates:
            raise RuntimeError(f"No canonical (24-class) relationships found for image index {img_idx}")
        rel_idx = random.choice(candidates)
        
        s_idx = int(self.relationship_subjects[img_idx, rel_idx])
        o_idx = int(self.relationship_objects[img_idx, rel_idx])
        p_idx = int(self.relationship_predicates[img_idx, rel_idx])
        
        # Get names from vocab (vocab uses list indexing, not dict)
        subj_obj_idx = int(self.object_names_h5[img_idx, s_idx])
        obj_obj_idx = int(self.object_names_h5[img_idx, o_idx])
        
        subject = self.vocab["object_idx_to_name"][subj_obj_idx]
        obj = self.vocab["object_idx_to_name"][obj_obj_idx]
        predicate = self.vocab["pred_idx_to_name"][p_idx]
        canonical_predicate = self.pred_id_to_canonical.get(p_idx)
        class_id = self.pred_id_to_class.get(p_idx)
        if canonical_predicate is None or class_id is None:
            raise RuntimeError(f"Predicate id {p_idx} did not map to a canonical class (unexpected).")

        rel_path = self.image_paths[img_idx]
        img_path = self.images_dir / str(rel_path)
        if not img_path.exists():
            raise FileNotFoundError(f"VG image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        
        return image, str(rel_path), (subject, predicate, obj), canonical_predicate, int(class_id)


class VGExamplesJSONLDataset(Dataset):
    """Dataset backed by a JSONL list of fixed triplet examples.

    Each JSONL line must contain:
      - triple: {subject, predicate (canonical), object}
      - meta: {image_rel_path, predicate_raw, class_id}
    """

    def __init__(self, examples_jsonl: Path, images_dir: str, max_samples: int = -1):
        self.examples_jsonl = Path(examples_jsonl)
        self.images_dir = Path(images_dir)
        rows: List[dict] = []
        with open(self.examples_jsonl, "r") as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"No examples found in {self.examples_jsonl}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        triple = row["triple"]
        meta = row["meta"]
        subject = triple["subject"]
        obj = triple["object"]
        canonical_predicate = triple["predicate"]
        predicate_raw = meta["predicate_raw"]
        class_id = int(meta["class_id"])
        image_rel_path = meta["image_rel_path"]

        img_path = self.images_dir / str(image_rel_path)
        if not img_path.exists():
            raise FileNotFoundError(f"VG image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        return image, str(image_rel_path), (subject, predicate_raw, obj), canonical_predicate, int(class_id)


def collate_images(batch):
    """Collate batch of images and triplets."""
    images, image_keys, triplets, canonical_preds, class_ids = zip(*batch)
    return (
        list(images),
        list(image_keys),
        list(triplets),
        list(canonical_preds),
        torch.tensor(class_ids, dtype=torch.long),
    )


def _dtype_from_str(s: str):
    if s == "float32":
        return torch.float32
    elif s == "bfloat16":
        return torch.bfloat16
    elif s == "float16":
        return torch.float16
    return torch.float32


def _inject_lora_into_blocks(transformer, block_indices, rank: int, alpha: float):
    """Inject LoRA into FluxAttention projections (plan_lora.md §2).

    Matches: attn.to_q, attn.to_k, attn.to_v, attn.to_out[0] only.
    """
    from relation_analysis.flux.lora import LinearWithLoRA

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


def _trainable_params(transformer, graph_encoder=None):
    """Collect trainable parameters: LoRA + graph projections."""
    params = []
    
    # LoRA parameters
    for name, param in transformer.named_parameters():
        if any(key in name for key in ["lora_A", "lora_B", "graph_local_proj", "graph_global_proj", "graph_global_to_temb"]):
            if param.requires_grad:
                params.append(param)
    
    return params


def _filtered_state_dict(transformer):
    keep = {}
    for name, tensor in transformer.state_dict().items():
        if any(key in name for key in ["graph_local_proj", "graph_global_proj", "graph_global_to_temb", "lora_A", "lora_B"]):
            keep[name] = tensor.detach().cpu()
    return keep


def build_pipeline(cfg: TrainConfig, torch_dtype):
    """Build Flux pipeline with proper tokenizers."""
    cache_dir = PROJECT_ROOT / "hf_flux_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_tok(subfolder: str):
        kwargs = dict(subfolder=subfolder, use_fast=False, cache_dir=cache_dir)
        try:
            return AutoTokenizer.from_pretrained(cfg.model_id, **kwargs)
        except Exception:
            return AutoTokenizer.from_pretrained(cfg.model_id, **kwargs, force_download=True)

    clip_tok = _load_tok("tokenizer")
    t5_tok = _load_tok("tokenizer_2")
    
    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch_dtype)
    pipe.tokenizer = clip_tok
    pipe.tokenizer_2 = t5_tok
    
    # Move all components to GPU for faster text encoding
    if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
        pipe.enable_model_cpu_offload(gpu_id=int(cfg.device.split(":")[1]) if ":" in cfg.device else 0)
    else:
        if cfg.device.startswith("cuda"):
            pipe.transformer.to(cfg.device, dtype=torch_dtype)
            pipe.vae.to(cfg.device, dtype=torch_dtype)
            # Move text encoders to GPU for faster encoding
            if pipe.text_encoder is not None:
                pipe.text_encoder.to(cfg.device, dtype=torch_dtype)
            if pipe.text_encoder_2 is not None:
                pipe.text_encoder_2.to(cfg.device, dtype=torch_dtype)
    
    pipe.set_progress_bar_config(disable=True)
    return pipe


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
    """Frozen 24-way relation classifier C (plan_lora.md §3.2)."""

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


class DifferentiableConceptAttentionCollector(contextlib.AbstractContextManager):
    """Collects ConceptAttention saliency during a transformer forward pass (plan_lora.md §3.2).

    - Keeps gradients (no .detach()).
    - Averages saliency across `record_layers`.
    """

    def __init__(self, transformer, concept_states: torch.Tensor, record_layers: Sequence[int]):
        self.transformer = transformer
        self.concept_states = concept_states
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
                    enc_out, img_out = orig_forward(*args, **kwargs)
                    concept_out = concept_attention_step(b, self.concept_states, img_out)
                    if i in self.record_layers:
                        sal = compute_saliency(img_out, concept_out)
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
            raise RuntimeError("No saliency was recorded. Check record_layers and transformer structure.")
        return self.saliency_sum / float(self.count)

def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    torch_dtype = _dtype_from_str(cfg.dtype)
    device = torch.device(cfg.device)
    
    print("=" * 80)
    print("GRAPH-CONDITIONED FLUX TRAINING (TEACHER-FORCED + REL-RANK)")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Mode: {cfg.graph_mode}")
    print(f"LoRA rank: {cfg.lora_rank}")
    print(f"Blocks: {cfg.block_start}-{cfg.block_end}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Image size: {cfg.height}x{cfg.width}")
    print(f"lambda_rel_rank: {cfg.lambda_rel_rank}")
    print(f"alpha_gen_rel: {cfg.alpha_gen_rel}")
    print("=" * 80)
    
    # Data
    if cfg.train_examples_jsonl is not None:
        train_ds = VGExamplesJSONLDataset(
            cfg.train_examples_jsonl, cfg.vg_images_dir, max_samples=cfg.max_train_samples
        )
    else:
        train_ds = VGImageDataset(cfg.vocab_path, cfg.train_h5, cfg.vg_images_dir, max_samples=cfg.max_train_samples)

    if cfg.val_examples_jsonl is not None:
        val_ds = VGExamplesJSONLDataset(cfg.val_examples_jsonl, cfg.vg_images_dir, max_samples=cfg.max_val_samples)
    else:
        val_ds = VGImageDataset(cfg.vocab_path, cfg.val_h5, cfg.vg_images_dir, max_samples=cfg.max_val_samples)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, collate_fn=collate_images)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, collate_fn=collate_images)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    if cfg.dry_run:
        img, key, triple, canon, cid = train_ds[0]
        print("DRY RUN: first train example")
        print(f"  image_key: {key}")
        print(f"  triple: {triple}")
        print(f"  canonical_predicate: {canon} (class_id={cid})")
        return
    
    # Models
    graph_encoder = SGDiffGraphEncoder(vocab_path=cfg.vocab_path, ckpt_path=cfg.cgip_ckpt, 
                                      device=cfg.graph_encoder_device)
    pipe = build_pipeline(cfg, torch_dtype)
    classifier = load_frozen_classifier(Path(cfg.classifier_ckpt), device=device)
    
    # Patch transformer for graph conditioning
    transformer = patch_flux_for_graph(pipe.transformer, mode=cfg.graph_mode, 
                                      block_range=range(cfg.block_start, cfg.block_end))
    
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Freeze base weights
    for p in transformer.parameters():
        p.requires_grad = False
    
    # Unfreeze graph projections
    for proj in [transformer.graph_local_proj, transformer.graph_global_proj, 
                 transformer.graph_global_to_temb]:
        if proj is not None:
            for p in proj.parameters():
                p.requires_grad = True
    
    # Inject LoRA
    _inject_lora_into_blocks(transformer, range(cfg.block_start, cfg.block_end), 
                            rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in _trainable_params(transformer) if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer
    params = _trainable_params(transformer)
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    predicate_map = default_predicate_map()
    canon_to_raw_preds: Dict[str, List[str]] = {}
    for p in graph_encoder.vocab.get("pred_idx_to_name", []):
        if not isinstance(p, str) or p.startswith("__"):
            continue
        spec = predicate_map.canonicalize(p)
        if spec is None:
            continue
        canon_to_raw_preds.setdefault(spec.name, []).append(p)

    supported_canonical = sorted([c for c in canon_to_raw_preds.keys() if c in CLASS_NAME_TO_ID])
    missing_canonical = sorted([c for c in CLASS_NAMES if c not in supported_canonical])
    print(f"Canonical predicates: {len(CLASS_NAMES)}")
    print(f"SGDiff-supported predicates: {len(supported_canonical)}")
    if missing_canonical:
        print(f"Unsupported under SGDiff vocab ({len(missing_canonical)}): {missing_canonical}")
    directional = {"left of", "right of", "above", "below", "in front of", "behind"}
    confusion_counts = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)

    def pick_hard_negative_canonical(true_canon: str) -> Optional[str]:
        """Pick a hard negative predicate based on classifier confusion (plan_lora.md §4.2)."""
        if cfg.hard_negative_topk <= 0:
            return None
        true_id = CLASS_NAME_TO_ID.get(true_canon)
        if true_id is None:
            return None
        row = confusion_counts[true_id].clone()
        row[true_id] = 0  # exclude the true class
        if int(row.sum().item()) == 0:
            return None
        k = min(int(cfg.hard_negative_topk), row.numel())
        top_ids = torch.topk(row, k=k).indices.tolist()
        candidates = [
            CLASS_NAMES[i]
            for i in top_ids
            if CLASS_NAMES[i] in supported_canonical and CLASS_NAMES[i] != true_canon and canon_to_raw_preds.get(CLASS_NAMES[i])
        ]
        return random.choice(candidates) if candidates else None

    def make_negative_triples(pos_triples: List[Tuple[str, str, str]], canonical_preds: List[str]) -> List[Tuple[str, str, str]]:
        neg = []
        for (s, p_raw, o), canon in zip(pos_triples, canonical_preds):
            if canon in directional:
                neg.append((o, p_raw, s))
                continue
            canon_neg = pick_hard_negative_canonical(canon)
            if canon_neg is None:
                choices = [c for c in supported_canonical if c != canon]
                canon_neg = random.choice(choices) if choices else None
            if canon_neg is None:
                neg.append((o, p_raw, s))
                continue
            raw_choices = canon_to_raw_preds.get(canon_neg, [])
            if not raw_choices:
                neg.append((o, p_raw, s))
                continue
            neg.append((s, random.choice(raw_choices), o))
        return neg
    
    def encode_prompt_batch(prompts: List[str]):
        """Encode text prompts."""
        # Run heavy text encoders on CPU to avoid VRAM OOM; then move embeddings to GPU.
        text_device = next(pipe.text_encoder.parameters()).device if pipe.text_encoder is not None else torch.device("cpu")
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=text_device,
            num_images_per_prompt=1,
            max_sequence_length=int(cfg.max_sequence_length),
        )
        return prompt_embeds, pooled, text_ids
    
    def encode_images(images: List[Image.Image]):
        """Encode images to latent space using VAE."""
        # Preprocess images
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((cfg.height, cfg.width)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        vae_device = next(pipe.vae.parameters()).device
        vae_dtype = next(pipe.vae.parameters()).dtype
        pixel_values = torch.stack([transform(img) for img in images]).to(vae_device, dtype=vae_dtype)
        
        # Encode with VAE
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        
        return latents.to(device=device, dtype=torch_dtype)

    def encode_images_cached(images: List[Image.Image], image_keys: List[str]):
        """VAE-encode images to z0 with optional on-disk caching (plan_lora.md §5)."""
        if cfg.latent_cache_dir is None:
            return encode_images(images)
        if len(images) != len(image_keys):
            raise ValueError("encode_images_cached: images and image_keys must have the same length")

        cache_dir = Path(cfg.latent_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{cfg.height}x{cfg.width}.pt"

        latents_list: List[Optional[torch.Tensor]] = [None] * len(images)
        missing_images: List[Image.Image] = []
        missing_indices: List[int] = []
        missing_paths: List[Path] = []

        for i, key in enumerate(image_keys):
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
            path = cache_dir / f"{digest}{suffix}"
            if path.exists():
                latents_list[i] = torch.load(path, map_location="cpu", weights_only=False)
            else:
                missing_images.append(images[i])
                missing_indices.append(i)
                missing_paths.append(path)

        if missing_images:
            encoded = encode_images(missing_images)
            for idx, path, lat in zip(missing_indices, missing_paths, encoded):
                lat_cpu = lat.detach().to("cpu", dtype=torch.float16)
                tmp = path.with_suffix(path.suffix + ".tmp")
                torch.save(lat_cpu, tmp)
                os.replace(tmp, path)
                latents_list[idx] = lat_cpu

        stacked = torch.stack([t for t in latents_list if t is not None], dim=0)
        if stacked.shape[0] != len(images):
            raise RuntimeError("Latent cache assembly failed (unexpected None entries).")
        return stacked.to(device=device, dtype=torch_dtype)

    def encode_concept_states(concepts_batch: List[List[str]]) -> torch.Tensor:
        """Encode [subject, predicate, object] concepts via text_encoder_2 and context_embedder."""
        flat: List[str] = [c for triplet in concepts_batch for c in triplet]
        t5_device = next(pipe.text_encoder_2.parameters()).device
        inputs = pipe.tokenizer_2(
            flat,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(t5_device)
        attention_mask = inputs.attention_mask.to(t5_device)
        out = pipe.text_encoder_2(input_ids)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.view(len(concepts_batch), len(concepts_batch[0]), -1)
        pooled = pooled.to(device=device, dtype=torch_dtype)
        return pipe.transformer.context_embedder(pooled)

    def saliency_to_classifier_input(saliency: torch.Tensor, h_tokens: int, w_tokens: int) -> torch.Tensor:
        """Convert (B, 3, tokens) saliency to classifier input (B, 3, S, S)."""
        if saliency.ndim != 3:
            raise ValueError(f"Expected saliency (B,3,tokens), got {tuple(saliency.shape)}")
        b, c, t = saliency.shape
        if t != h_tokens * w_tokens:
            raise ValueError(f"Token mismatch: saliency tokens={t} expected={h_tokens*w_tokens}")
        # Match dataset generation: per-concept softmax over image tokens.
        sal = torch.softmax(saliency.float(), dim=-1).view(b, c, h_tokens, w_tokens)
        if h_tokens != cfg.classifier_in_size or w_tokens != cfg.classifier_in_size:
            sal = F.interpolate(sal, size=(cfg.classifier_in_size, cfg.classifier_in_size), mode="bilinear", align_corners=False)
        return sal

    def run_batch(
        images: List[Image.Image],
        image_keys: List[str],
        triples: List[Tuple[str, str, str]],
        canonical_preds: List[str],
        labels: torch.Tensor,
        training: bool = True,
    ):
        """Run one training step (plan_lora.md §5)."""
        predicate_phrases = [predicate_to_phrase(p) for p in canonical_preds]
        prompts = [
            cfg.prompt_template.format(subject=s, predicate=pred_phrase, object=o)
            for (s, _p_raw, o), pred_phrase in zip(triples, predicate_phrases)
        ]
        
        # Encode prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_batch(prompts)
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch_dtype)
        text_ids = text_ids.to(device=device, dtype=torch_dtype)

        # Encode concepts (subject / predicate / object) for ConceptAttention tracing.
        concepts_batch = [[s, pred_phrase, o] for (s, _p_raw, o), pred_phrase in zip(triples, predicate_phrases)]
        with torch.no_grad():
            concept_states0 = encode_concept_states(concepts_batch)
        
        # Encode graph (positive)
        with torch.no_grad():
            graph_local_pos, graph_global_pos = graph_encoder.encode_batch(triples)
        graph_local_pos = graph_local_pos.to(device=device, dtype=torch_dtype)
        graph_global_pos = graph_global_pos.to(device=device, dtype=torch_dtype)
        
        # Encode images to latents (optionally cached)
        latents = encode_images_cached(images, image_keys)
        
        # Sample random timesteps
        if not (0.0 <= cfg.t_min <= cfg.t_max <= 1.0):
            raise ValueError(f"Invalid timestep range: t_min={cfg.t_min}, t_max={cfg.t_max} (expected 0<=t_min<=t_max<=1)")
        timesteps = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(len(triples), device=device, dtype=torch_dtype)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Add noise to latents (forward diffusion)
        # For flow matching: noisy = (1-t)*x + t*noise
        noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * latents + timesteps.view(-1, 1, 1, 1) * noise
        
        # Pack latents for Flux (use actual latent dimensions)
        packed_noisy_latents = pipe._pack_latents(
            noisy_latents,
            batch_size=len(triples),
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )
        
        # Prepare image IDs for PACKED latents (H//2, W//2 after packing)
        # The packing operation reduces spatial dims by 2x in each dimension
        latent_image_ids = pipe._prepare_latent_image_ids(
            batch_size=latents.shape[0],
            height=latents.shape[2] // 2,
            width=latents.shape[3] // 2,
            device=device,
            dtype=torch_dtype
        )

        # For flow matching, target is: velocity = noise - latents (plan_lora.md §3.1)
        target = noise - latents
        packed_target = pipe._pack_latents(
            target,
            batch_size=len(triples),
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        # Forward (positive graph) and collect saliency from double blocks 7-12.
        set_graph_condition(transformer, graph_local=graph_local_pos, graph_global=graph_global_pos)
        with DifferentiableConceptAttentionCollector(
            transformer,
            concept_states=concept_states0.clone(),
            record_layers=range(cfg.block_start, cfg.block_end),
        ) as tracer_pos:
            model_pred_pos = transformer(
                hidden_states=packed_noisy_latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timesteps,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                return_dict=False,
                graph_local=graph_local_pos,
                graph_global=graph_global_pos,
            )[0]
            saliency_pos = tracer_pos.mean_saliency()

        err_pos = (model_pred_pos.float() - packed_target.float()).pow(2).mean(dim=2)  # (B, tokens)
        loss_gen = err_pos.mean()

        # L_gen-rel: predicate-saliency-weighted flow-matching loss (plan_lora.md §3.3)
        loss_gen_rel = torch.tensor(0.0, device=device)
        if cfg.alpha_gen_rel > 0:
            w = torch.sigmoid(saliency_pos[:, 1, :])
            w = w / w.mean(dim=1, keepdim=True).clamp(min=1e-6)
            w = w.detach()
            loss_gen_rel = (w * err_pos).mean()

        loss_rel_rank = torch.tensor(0.0, device=device)
        margin_mean = torch.tensor(0.0, device=device)
        acc_pos = torch.tensor(0.0, device=device)
        if cfg.use_negative_graph and cfg.lambda_rel_rank > 0:
            triples_neg = make_negative_triples(triples, canonical_preds)
            with torch.no_grad():
                graph_local_neg, graph_global_neg = graph_encoder.encode_batch(triples_neg)
            graph_local_neg = graph_local_neg.to(device=device, dtype=torch_dtype)
            graph_global_neg = graph_global_neg.to(device=device, dtype=torch_dtype)
            set_graph_condition(transformer, graph_local=graph_local_neg, graph_global=graph_global_neg)
            with DifferentiableConceptAttentionCollector(
                transformer,
                concept_states=concept_states0.clone(),
                record_layers=range(cfg.block_start, cfg.block_end),
            ) as tracer_neg:
                _ = transformer(
                    hidden_states=packed_noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timesteps,
                    img_ids=latent_image_ids,
                    txt_ids=text_ids,
                    return_dict=False,
                    graph_local=graph_local_neg,
                    graph_global=graph_global_neg,
                )[0]
                saliency_neg = tracer_neg.mean_saliency()

            # Frozen classifier on ConceptAttention saliency (plan_lora.md §3.2)
            x_pos = saliency_to_classifier_input(saliency_pos, h_tokens=latents.shape[2] // 2, w_tokens=latents.shape[3] // 2)
            x_neg = saliency_to_classifier_input(saliency_neg, h_tokens=latents.shape[2] // 2, w_tokens=latents.shape[3] // 2)
            logits_pos = classifier(x_pos)
            logits_neg = classifier(x_neg)

            labels_ = labels.to(device)
            with torch.no_grad():
                preds = logits_pos.argmax(dim=1).detach().cpu()
                true_cpu = labels_.detach().cpu()
                for t, p in zip(true_cpu.tolist(), preds.tolist()):
                    if 0 <= t < confusion_counts.shape[0] and 0 <= p < confusion_counts.shape[1]:
                        confusion_counts[t, p] += 1
            score_pos = logits_pos[torch.arange(labels_.shape[0], device=device), labels_]
            score_neg = logits_neg[torch.arange(labels_.shape[0], device=device), labels_]

            loss_rel_rank = F.relu(cfg.rel_rank_margin - score_pos + score_neg).mean()
            margin_mean = (score_pos - score_neg).mean()
            acc_pos = (logits_pos.argmax(dim=1) == labels_).float().mean()

        loss_total = loss_gen + cfg.lambda_rel_rank * loss_rel_rank + cfg.alpha_gen_rel * loss_gen_rel
        return (
            loss_total,
            loss_gen.detach(),
            loss_rel_rank.detach(),
            loss_gen_rel.detach(),
            margin_mean.detach(),
            acc_pos.detach(),
        )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        transformer.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for images, image_keys, triples, canonical_preds, labels in pbar:
            optim.zero_grad(set_to_none=True)
            
            loss, loss_gen, loss_rank, loss_gen_rel, margin, acc_pos = run_batch(
                images, image_keys, triples, canonical_preds, labels, training=True
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(_trainable_params(transformer), 1.0)
            
            optim.step()
            
            if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
                pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            
            global_step += 1
            
            if global_step % cfg.log_every == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "gen": f"{loss_gen.item():.4f}",
                        "rank": f"{loss_rank.item():.4f}",
                        "gen_rel": f"{loss_gen_rel.item():.4f}",
                        "Δ": f"{margin.item():.3f}",
                        "acc+": f"{acc_pos.item():.2f}",
                    }
                )
                # Also print a newline heartbeat so nohup logs advance even when tqdm output is suppressed.
                print(
                    f"[step {global_step}] loss={loss.item():.4f} gen={loss_gen.item():.4f} "
                    f"rank={loss_rank.item():.4f} Δ={margin.item():.3f} acc+={acc_pos.item():.2f}",
                    flush=True,
                )
            
            # Validation
            if global_step % cfg.val_every == 0:
                transformer.eval()
                val_losses = []
                with torch.no_grad():
                    for images, image_keys, triples, canonical_preds, labels in val_loader:
                        loss, _, _, _, _, _ = run_batch(images, image_keys, triples, canonical_preds, labels, training=False)
                        val_losses.append(loss.item())
                        if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
                            pipe.transformer.to("cpu")
                            torch.cuda.empty_cache()
                
                mean_val_loss = sum(val_losses) / max(1, len(val_losses))
                print(f"\n[Step {global_step}] val_loss={mean_val_loss:.4f}")
                
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    ckpt_path = cfg.output_dir / f"best_graph_lora_{cfg.graph_mode}.pt"
                    torch.save({
                        "config": asdict(cfg),
                        "state_dict": _filtered_state_dict(transformer),
                        "optimizer": optim.state_dict(),
                        "step": global_step,
                        "val_loss": mean_val_loss,
                    }, ckpt_path)
                    print(f"Saved best checkpoint to {ckpt_path}")
                
                transformer.train()
            
            # Save checkpoint
            if global_step % cfg.save_every == 0:
                ckpt_path = cfg.output_dir / f"graph_lora_{cfg.graph_mode}_step{global_step}.pt"
                torch.save({
                    "config": asdict(cfg),
                    "state_dict": _filtered_state_dict(transformer),
                    "optimizer": optim.state_dict(),
                    "step": global_step,
                }, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Final save
    ckpt_path = cfg.output_dir / f"final_graph_lora_{cfg.graph_mode}.pt"
    torch.save({
        "config": asdict(cfg),
        "state_dict": _filtered_state_dict(transformer),
        "optimizer": optim.state_dict(),
        "step": global_step,
    }, ckpt_path)
    print(f"\nTraining complete! Saved final checkpoint to {ckpt_path}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Graph-conditioned Flux LoRA training (teacher-forced + rel-rank).")
    p.add_argument("--model-id", default=TrainConfig.model_id)
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--dtype", default=TrainConfig.dtype, choices=["float32", "bfloat16", "float16"])
    p.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.enable_cpu_offload,
        help="Enable model CPU offload (may be slower but uses less VRAM).",
    )
    p.add_argument("--graph-mode", default=TrainConfig.graph_mode, choices=["token", "temb"])
    p.add_argument("--block-start", type=int, default=TrainConfig.block_start)
    p.add_argument("--block-end", type=int, default=TrainConfig.block_end)
    p.add_argument("--lora-rank", type=int, default=TrainConfig.lora_rank)
    p.add_argument("--lora-alpha", type=float, default=TrainConfig.lora_alpha)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--height", type=int, default=TrainConfig.height)
    p.add_argument("--width", type=int, default=TrainConfig.width)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--val-every", type=int, default=TrainConfig.val_every)
    p.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.gradient_checkpointing,
        help="Enable gradient checkpointing on the Flux transformer to reduce activation memory.",
    )
    p.add_argument("--vocab-path", default=TrainConfig.vocab_path)
    p.add_argument("--train-h5", default=TrainConfig.train_h5)
    p.add_argument("--val-h5", default=TrainConfig.val_h5)
    p.add_argument("--vg-images-dir", default=TrainConfig.vg_images_dir)
    p.add_argument("--cgip-ckpt", default=TrainConfig.cgip_ckpt)
    p.add_argument("--graph-encoder-device", default=TrainConfig.graph_encoder_device)
    p.add_argument("--prompt-template", default=TrainConfig.prompt_template)
    p.add_argument(
        "--max-sequence-length",
        type=int,
        default=TrainConfig.max_sequence_length,
        help="T5 max sequence length for Flux prompt encoding (default 512). Lowering this speeds up CPU text encoding.",
    )
    p.add_argument("--t-min", type=float, default=TrainConfig.t_min)
    p.add_argument("--t-max", type=float, default=TrainConfig.t_max)
    p.add_argument("--lambda-rel-rank", type=float, default=TrainConfig.lambda_rel_rank)
    p.add_argument("--alpha-gen-rel", type=float, default=TrainConfig.alpha_gen_rel)
    p.add_argument("--rel-rank-margin", type=float, default=TrainConfig.rel_rank_margin)
    p.add_argument("--classifier-ckpt", type=Path, default=TrainConfig.classifier_ckpt)
    p.add_argument("--classifier-in-size", type=int, default=TrainConfig.classifier_in_size)
    p.add_argument(
        "--use-negative-graph",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.use_negative_graph,
        help="Run a second forward pass with a corrupted graph g-.",
    )
    p.add_argument(
        "--hard-negative-topk",
        type=int,
        default=TrainConfig.hard_negative_topk,
        help="For non-directional predicates, replace with a top-k confused class under the frozen classifier (0=random).",
    )
    p.add_argument("--max-train-samples", type=int, default=TrainConfig.max_train_samples)
    p.add_argument("--max-val-samples", type=int, default=TrainConfig.max_val_samples)
    p.add_argument("--latent-cache-dir", type=Path, default=TrainConfig.latent_cache_dir)
    p.add_argument(
        "--train-examples-jsonl",
        type=Path,
        default=TrainConfig.train_examples_jsonl,
        help="Optional JSONL of fixed train examples (see make_vg_quickwin_split.py). Overrides H5 scanning.",
    )
    p.add_argument(
        "--val-examples-jsonl",
        type=Path,
        default=TrainConfig.val_examples_jsonl,
        help="Optional JSONL of fixed val/test examples (never used for gradients). Overrides val.h5 scanning.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate dataset/split wiring and exit before loading Flux.",
    )
    
    args = p.parse_args()
    
    return TrainConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        enable_cpu_offload=args.cpu_offload,
        graph_mode=args.graph_mode,
        block_start=args.block_start,
        block_end=args.block_end,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        height=args.height,
        width=args.width,
        num_workers=args.num_workers,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        seed=args.seed,
        output_dir=args.output_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        vocab_path=args.vocab_path,
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        vg_images_dir=args.vg_images_dir,
        cgip_ckpt=args.cgip_ckpt,
        graph_encoder_device=args.graph_encoder_device,
        lambda_rel_rank=args.lambda_rel_rank,
        alpha_gen_rel=args.alpha_gen_rel,
        rel_rank_margin=args.rel_rank_margin,
        classifier_ckpt=args.classifier_ckpt,
        classifier_in_size=args.classifier_in_size,
        prompt_template=args.prompt_template,
        max_sequence_length=args.max_sequence_length,
        t_min=args.t_min,
        t_max=args.t_max,
        use_negative_graph=args.use_negative_graph,
        hard_negative_topk=args.hard_negative_topk,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        latent_cache_dir=args.latent_cache_dir,
        train_examples_jsonl=args.train_examples_jsonl,
        val_examples_jsonl=args.val_examples_jsonl,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    train(parse_args())
