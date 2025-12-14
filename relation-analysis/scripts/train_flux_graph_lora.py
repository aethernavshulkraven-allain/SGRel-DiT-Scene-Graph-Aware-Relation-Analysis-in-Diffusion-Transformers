#!/usr/bin/env python3
"""
Lightweight graph-conditioned Flux fine-tuning with LoRA over the middle blocks.

What this does:
- Encodes single-triplet scene graphs with SGDiff's CGIP GNN (frozen).
- Injects the graph embeddings into Flux's middle transformer blocks (token concat or temb add).
- Trains only the graph projection layers + attention LoRA adapters + a small predicate classifier head.
  The objective is a simple predicate classification over VG relationships; it gives the LoRA layers a
  supervised signal without touching the base Flux weights.

This is a starter script meant to get graph conditioning + LoRA plumbing in place. For real image-quality
training you will want a diffusion loss; this keeps things light so you can iterate on the wiring first.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Keep tokenizer parallelism quiet.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------------------------------------------------------------------------
# Temporary stubs for older transformers installations missing SigLIP.
# -------------------------------------------------------------------------
try:
    import transformers
    if not hasattr(transformers, "SiglipImageProcessor"):
        class _StubSiglipImageProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
        transformers.SiglipImageProcessor = _StubSiglipImageProcessor  # type: ignore
    if not hasattr(transformers, "SiglipVisionModel"):
        class _StubSiglipVisionModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
        transformers.SiglipVisionModel = _StubSiglipVisionModel  # type: ignore
except Exception:
    # If transformers itself is missing, we let the downstream import raise a clearer error.
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Make local diffusers importable if not installed
local_diffusers = REPO_ROOT / "diffusers" / "src"
if local_diffusers.exists() and str(local_diffusers) not in sys.path:
    sys.path.insert(0, str(local_diffusers))

from diffusers import FluxPipeline

from relation_analysis.flux.graph_conditioned_flux import clear_graph_condition, patch_flux_for_graph, set_graph_condition
from relation_analysis.flux.lora import LinearWithLoRA, inject_lora
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder


@dataclass
class TrainConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda:3"
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = True
    gradient_checkpointing: bool = True
    graph_mode: str = "token"  # token|temb
    block_start: int = 7
    block_end: int = 13  # exclusive
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lr: float = 1e-4
    batch_size: int = 2
    epochs: int = 1
    seed: int = 0
    height: int = 256
    width: int = 256
    num_workers: int = 2
    log_every: int = 10
    max_train_samples: int = None
    max_val_samples: int = 256
    graph_encoder_device: str = "cpu"
    vocab_path: Path = REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json"
    train_h5: Path = REPO_ROOT / "SGDiff" / "datasets" / "vg" / "train.h5"
    val_h5: Path = REPO_ROOT / "SGDiff" / "datasets" / "vg" / "val.h5"
    cgip_ckpt: Path = REPO_ROOT / "SGDiff" / "pretrained" / "sip_vg.pt"
    output_dir: Path = PROJECT_ROOT / "outputs" / "graph_flux_lora"
    prompt_template: str = "a photo of {subject} {predicate} {object}"


class VGPredicateDataset(Dataset):
    """
    Samples a single relationship per image from the VG HDF5 split and returns
    (subject_name, predicate_name, object_name, predicate_idx).
    """

    def __init__(self, vocab_path: Path, h5_path: Path, max_samples: int = None):
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.h5 = h5py.File(h5_path, "r")
        self.rels_per_image = self.h5["relationships_per_image"][()]
        self.relationship_subjects = self.h5["relationship_subjects"]
        self.relationship_objects = self.h5["relationship_objects"]
        self.relationship_predicates = self.h5["relationship_predicates"]
        self.object_names = self.h5["object_names"]
        self.valid_indices = [i for i, c in enumerate(self.rels_per_image) if c > 0]
        if max_samples is not None:
            self.valid_indices = self.valid_indices[:max_samples]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        img_idx = self.valid_indices[idx]
        count = int(self.rels_per_image[img_idx])
        rel_idx = random.randrange(count)
        s_idx = int(self.relationship_subjects[img_idx, rel_idx])
        o_idx = int(self.relationship_objects[img_idx, rel_idx])
        p_idx = int(self.relationship_predicates[img_idx, rel_idx])

        subj_name = self.vocab["object_idx_to_name"][int(self.object_names[img_idx, s_idx])]
        obj_name = self.vocab["object_idx_to_name"][int(self.object_names[img_idx, o_idx])]
        pred_name = self.vocab["pred_idx_to_name"][p_idx]
        return subj_name, pred_name, obj_name, p_idx


def collate_triples(batch: List[Tuple[str, str, str, int]]):
    triples = [(s, p, o) for s, p, o, _ in batch]
    labels = torch.tensor([label for _, _, _, label in batch], dtype=torch.long)
    return triples, labels


def _dtype_from_str(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _inject_lora_middle_blocks(transformer, block_indices: Sequence[int], rank: int, alpha: float):
    targets = ["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    for idx in block_indices:
        if idx < 0 or idx >= len(transformer.transformer_blocks):
            continue
        inject_lora(transformer.transformer_blocks[idx], targets, rank=rank, alpha=alpha)


def _trainable_params(transformer, classifier):
    trainable = []
    for name, param in transformer.named_parameters():
        if any(key in name for key in ["graph_local_proj", "graph_global_proj", "graph_global_to_temb", "lora_A", "lora_B"]):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    trainable += list(classifier.parameters())
    return trainable


def _filtered_state_dict(transformer, classifier):
    keep = {}
    for name, tensor in transformer.state_dict().items():
        if any(key in name for key in ["graph_local_proj", "graph_global_proj", "graph_global_to_temb", "lora_A", "lora_B"]):
            keep[name] = tensor.cpu()
    keep["classifier"] = classifier.state_dict()
    return keep


def build_pipeline(cfg: TrainConfig, torch_dtype):
    cache_dir = PROJECT_ROOT / "hf_flux_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Force slow tokenizers; Flux accepts PreTrainedTokenizer but rejects fast types here.
    clip_tok = AutoTokenizer.from_pretrained(
        cfg.model_id, subfolder="tokenizer", use_fast=False, cache_dir=cache_dir, force_download=True
    )
    t5_tok = AutoTokenizer.from_pretrained(
        cfg.model_id, subfolder="tokenizer_2", use_fast=False, cache_dir=cache_dir, force_download=True
    )
    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch_dtype)
    # Overwrite tokenizers after load to bypass class check.
    pipe.tokenizer = clip_tok
    pipe.tokenizer_2 = t5_tok
    if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
        pipe.enable_model_cpu_offload(gpu_id=int(cfg.device.split(":")[1]) if ":" in cfg.device else 0)
    else:
        pipe = pipe.to(cfg.device, dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    torch_dtype = _dtype_from_str(cfg.dtype)
    device = torch.device(cfg.device)

    # Data
    train_ds = VGPredicateDataset(cfg.vocab_path, cfg.train_h5, max_samples=cfg.max_train_samples)
    val_ds = VGPredicateDataset(cfg.vocab_path, cfg.val_h5, max_samples=cfg.max_val_samples)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_triples)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_triples)

    # Models
    graph_encoder = SGDiffGraphEncoder(vocab_path=cfg.vocab_path, ckpt_path=cfg.cgip_ckpt, device=cfg.graph_encoder_device)
    pipe = build_pipeline(cfg, torch_dtype)
    transformer = patch_flux_for_graph(pipe.transformer, mode=cfg.graph_mode, block_range=range(cfg.block_start, cfg.block_end))
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    # Freeze base weights, then re-enable the graph projections
    for p in transformer.parameters():
        p.requires_grad = False
    for proj in [transformer.graph_local_proj, transformer.graph_global_proj, transformer.graph_global_to_temb]:
        for p in proj.parameters():
            p.requires_grad = True
    _inject_lora_middle_blocks(transformer, range(cfg.block_start, cfg.block_end), rank=cfg.lora_rank, alpha=cfg.lora_alpha)

    num_preds = len(graph_encoder.vocab["pred_idx_to_name"])
    classifier = nn.Linear(transformer.out_channels or transformer.config.in_channels, num_preds).to(device)

    params = _trainable_params(transformer, classifier)
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def encode_prompt_batch(prompts: List[str]):
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=128,
        )
        return prompt_embeds, pooled, text_ids

    def prepare_latent_inputs(batch_size: int):
        num_channels_latents = transformer.config.in_channels // 4
        latents, img_ids = pipe.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=cfg.height,
            width=cfg.width,
            dtype=torch_dtype,
            device=device,
            generator=None,
            latents=None,
        )
        return latents, img_ids

    def run_batch(triples: List[Tuple[str, str, str]], labels: torch.Tensor):
        prompts = [cfg.prompt_template.format(subject=s, predicate=p, object=o) for s, p, o in triples]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_batch(prompts)
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch_dtype)
        text_ids = text_ids.to(device=device, dtype=torch_dtype)
        with torch.no_grad():
            graph_local, graph_global = graph_encoder.encode_batch(triples)
        graph_local = graph_local.to(device=device, dtype=torch_dtype)
        graph_global = graph_global.to(device=device, dtype=torch_dtype)
        latents, img_ids = prepare_latent_inputs(batch_size=len(triples))

        timesteps = torch.rand(len(triples), device=device, dtype=torch_dtype)
        set_graph_condition(transformer, graph_local=graph_local, graph_global=graph_global)
        sample = transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=text_ids,
            return_dict=False,
            graph_local=graph_local,
            graph_global=graph_global,
        )[0]
        pooled = sample.mean(dim=1).float()
        logits = classifier(pooled)
        labels = labels.to(device)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
        return loss, acc

    global_step = 0
    for epoch in range(cfg.epochs):
        transformer.train()
        classifier.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for triples, labels in pbar:
            optim.zero_grad(set_to_none=True)
            loss, acc = run_batch(triples, labels)
            loss.backward()
            optim.step()
            if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
                pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            global_step += 1
            if global_step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

        # quick val
        transformer.eval()
        classifier.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for triples, labels in val_loader:
                loss, acc = run_batch(triples, labels)
                val_losses.append(loss.item())
                val_accs.append(acc)
                if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
                    pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
        mean_val_loss = sum(val_losses) / max(1, len(val_losses))
        mean_val_acc = sum(val_accs) / max(1, len(val_accs))
        print(f"[val] loss={mean_val_loss:.3f} acc={mean_val_acc:.3f}")

    ckpt = {
        "config": asdict(cfg),
        "state_dict": _filtered_state_dict(transformer, classifier),
    }
    ckpt_path = cfg.output_dir / f"graph_lora_{cfg.graph_mode}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved LoRA/projection checkpoint to {ckpt_path}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Graph-conditioned Flux LoRA training (predicate classification proxy).")
    p.add_argument("--model-id", type=str, default=TrainConfig.model_id, help="Flux model id")
    p.add_argument("--device", type=str, default=TrainConfig.device, help="cuda|cpu")
    p.add_argument("--dtype", type=str, default=TrainConfig.dtype, help="bfloat16|float16|float32")
    p.add_argument("--graph-mode", type=str, default=TrainConfig.graph_mode, choices=["token", "temb"])
    p.add_argument("--block-start", type=int, default=TrainConfig.block_start)
    p.add_argument("--block-end", type=int, default=TrainConfig.block_end)
    p.add_argument("--lora-rank", type=int, default=TrainConfig.lora_rank)
    p.add_argument("--lora-alpha", type=float, default=TrainConfig.lora_alpha)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.enable_cpu_offload,
        help="Enable model CPU offload to reduce GPU memory.",
    )
    p.add_argument("--height", type=int, default=TrainConfig.height)
    p.add_argument("--width", type=int, default=TrainConfig.width)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=TrainConfig.max_val_samples)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.gradient_checkpointing,
        help="Enable gradient checkpointing on the Flux transformer to reduce activation memory.",
    )
    p.add_argument(
        "--graph-encoder-device",
        type=str,
        default=TrainConfig.graph_encoder_device,
        help="Where to run the SGDiff CGIP graph encoder (cpu|cuda). Keeping this on CPU saves VRAM.",
    )
    p.add_argument("--vocab-path", type=str, default=str(TrainConfig.vocab_path))
    p.add_argument("--train-h5", type=str, default=str(TrainConfig.train_h5))
    p.add_argument("--val-h5", type=str, default=str(TrainConfig.val_h5))
    p.add_argument("--cgip-ckpt", type=str, default=str(TrainConfig.cgip_ckpt))
    p.add_argument("--output-dir", type=str, default=str(TrainConfig.output_dir))
    p.add_argument("--prompt-template", type=str, default=TrainConfig.prompt_template)
    args = p.parse_args()
    cfg = TrainConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        graph_mode=args.graph_mode,
        block_start=args.block_start,
        block_end=args.block_end,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        enable_cpu_offload=args.cpu_offload,
        height=args.height,
        width=args.width,
        num_workers=args.num_workers,
        log_every=args.log_every,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        gradient_checkpointing=args.gradient_checkpointing,
        graph_encoder_device=args.graph_encoder_device,
        vocab_path=Path(args.vocab_path),
        train_h5=Path(args.train_h5),
        val_h5=Path(args.val_h5),
        cgip_ckpt=Path(args.cgip_ckpt),
        output_dir=Path(args.output_dir),
        prompt_template=args.prompt_template,
    )
    return cfg


if __name__ == "__main__":
    train(parse_args())
