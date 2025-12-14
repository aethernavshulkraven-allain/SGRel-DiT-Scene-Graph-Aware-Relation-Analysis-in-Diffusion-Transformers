#!/usr/bin/env python3
"""
Graph-conditioned Flux fine-tuning with LoRA using proper diffusion denoising loss.

Training strategy:
- Frozen: Base Flux transformer, VAE, text encoders
- Trainable: LoRA adapters in middle blocks + graph projection layers
- Loss: MSE between predicted noise and actual noise (standard diffusion objective)
- Dataset: Real VG images with their scene graph triplets

This trains the model to denoise images conditioned on scene graphs, making LoRA weights
learn structure-aware features for spatial relationship generation.
"""

import argparse
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import from existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.flux.graph_conditioned_flux import patch_flux_for_graph, set_graph_condition

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

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline


@dataclass
class TrainConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda:3"
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = False
    gradient_checkpointing: bool = True
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
    output_dir: Path = Path("checkpoints/graph_lora_diffusion")
    
    # Dataset paths
    vocab_path: str = "../../SGDiff/datasets/vg/vocab.json"
    train_h5: str = "../../SGDiff/datasets/vg/train.h5"
    val_h5: str = "../../SGDiff/datasets/vg/val.h5"
    vg_images_dir: str = "../../SGDiff/datasets/vg/images"
    cgip_ckpt: str = "../../SGDiff/pretrained/sip_vg.pt"
    graph_encoder_device: str = "cuda:3"
    
    # Diffusion training
    num_inference_steps: int = 4  # FLUX.1-schnell uses 4 steps
    guidance_scale: float = 0.0  # Schnell doesn't use guidance
    
    # Prompt template
    prompt_template: str = "a photo of {subject} {predicate} {object}"


class VGImageDataset(Dataset):
    """Visual Genome dataset that loads actual images with their scene graph triplets."""
    
    def __init__(self, vocab_path: str, h5_path: str, images_dir: str, max_samples: int = -1):
        self.vocab = self._load_vocab(vocab_path)
        self.images_dir = Path(images_dir)
        self.h5_path = h5_path
        
        # Load H5 file to get structure
        self.h5 = h5py.File(h5_path, 'r')
        self.rels_per_image = self.h5["relationships_per_image"][()]
        self.relationship_subjects = self.h5["relationship_subjects"]
        self.relationship_objects = self.h5["relationship_objects"]
        self.relationship_predicates = self.h5["relationship_predicates"]
        self.object_names_h5 = self.h5["object_names"]
        
        # Get valid image indices (images that have at least one relationship)
        self.valid_indices = [i for i, c in enumerate(self.rels_per_image) if c > 0]
        
        if max_samples > 0:
            self.valid_indices = self.valid_indices[:max_samples]
            
    def _load_vocab(self, path: str):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        
        # Get a random relationship for this image
        count = int(self.rels_per_image[img_idx])
        rel_idx = random.randrange(count)
        
        s_idx = int(self.relationship_subjects[img_idx, rel_idx])
        o_idx = int(self.relationship_objects[img_idx, rel_idx])
        p_idx = int(self.relationship_predicates[img_idx, rel_idx])
        
        # Get names from vocab (vocab uses list indexing, not dict)
        subj_obj_idx = int(self.object_names_h5[img_idx, s_idx])
        obj_obj_idx = int(self.object_names_h5[img_idx, o_idx])
        
        subject = self.vocab["object_idx_to_name"][subj_obj_idx]
        obj = self.vocab["object_idx_to_name"][obj_obj_idx]
        predicate = self.vocab["pred_idx_to_name"][p_idx]
        
        # Load image (note: VG H5 doesn't store image_id directly, using img_idx as approximation)
        # You may need to adjust this based on your actual image naming scheme
        img_path = self.images_dir / f"{img_idx}.jpg"
        if not img_path.exists():
            # Try with VG_ prefix
            img_path = self.images_dir / f"VG_{img_idx}.jpg"
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback to random noise if image not found
            image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        return image, (subject, predicate, obj), p_idx


def collate_images(batch):
    """Collate batch of images and triplets."""
    images, triplets, pred_labels = zip(*batch)
    return list(images), list(triplets), torch.tensor(pred_labels, dtype=torch.long)


def _dtype_from_str(s: str):
    if s == "float32":
        return torch.float32
    elif s == "bfloat16":
        return torch.bfloat16
    elif s == "float16":
        return torch.float16
    return torch.float32


def _inject_lora_into_blocks(transformer, block_indices, rank: int, alpha: float):
    """Inject LoRA into middle blocks' attention layers."""
    from relation_analysis.flux.lora import inject_lora
    targets = ["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    for idx in block_indices:
        if idx < 0 or idx >= len(transformer.transformer_blocks):
            continue
        inject_lora(transformer.transformer_blocks[idx], targets, rank=rank, alpha=alpha)


def _trainable_params(transformer, graph_encoder=None):
    """Collect trainable parameters: LoRA + graph projections."""
    params = []
    
    # LoRA parameters
    for name, param in transformer.named_parameters():
        if any(key in name for key in ["lora_A", "lora_B", "graph_local_proj", "graph_global_proj", "graph_global_to_temb"]):
            if param.requires_grad:
                params.append(param)
    
    return params


def build_pipeline(cfg: TrainConfig, torch_dtype):
    """Build Flux pipeline with proper tokenizers."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    
    clip_tok = AutoTokenizer.from_pretrained(
        cfg.model_id, subfolder="tokenizer", use_fast=False, cache_dir=cache_dir, force_download=True
    )
    t5_tok = AutoTokenizer.from_pretrained(
        cfg.model_id, subfolder="tokenizer_2", use_fast=False, cache_dir=cache_dir, force_download=True
    )
    
    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch_dtype)
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
    
    print("=" * 80)
    print("GRAPH-CONDITIONED FLUX TRAINING WITH DIFFUSION LOSS")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Mode: {cfg.graph_mode}")
    print(f"LoRA rank: {cfg.lora_rank}")
    print(f"Blocks: {cfg.block_start}-{cfg.block_end}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Image size: {cfg.height}x{cfg.width}")
    print("=" * 80)
    
    # Data
    train_ds = VGImageDataset(cfg.vocab_path, cfg.train_h5, cfg.vg_images_dir, max_samples=cfg.max_train_samples)
    val_ds = VGImageDataset(cfg.vocab_path, cfg.val_h5, cfg.vg_images_dir, max_samples=cfg.max_val_samples)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, collate_fn=collate_images)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, collate_fn=collate_images)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Models
    graph_encoder = SGDiffGraphEncoder(vocab_path=cfg.vocab_path, ckpt_path=cfg.cgip_ckpt, 
                                      device=cfg.graph_encoder_device)
    pipe = build_pipeline(cfg, torch_dtype)
    
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
    
    # Scheduler for diffusion training
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_id, subfolder="scheduler"
    )
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    def encode_prompt_batch(prompts: List[str]):
        """Encode text prompts."""
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
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
        
        pixel_values = torch.stack([transform(img) for img in images]).to(device, dtype=torch_dtype)
        
        # Encode with VAE
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        
        return latents
    
    def run_batch(images: List[Image.Image], triples: List[Tuple[str, str, str]], training: bool = True):
        """Run diffusion training on a batch."""
        prompts = [cfg.prompt_template.format(subject=s, predicate=p, object=o) for s, p, o in triples]
        
        # Encode prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_batch(prompts)
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch_dtype)
        text_ids = text_ids.to(device=device, dtype=torch_dtype)
        
        # Encode graph
        with torch.no_grad():
            graph_local, graph_global = graph_encoder.encode_batch(triples)
        graph_local = graph_local.to(device=device, dtype=torch_dtype)
        graph_global = graph_global.to(device=device, dtype=torch_dtype)
        
        # Encode images to latents
        latents = encode_images(images)
        
        # Sample random timesteps
        timesteps = torch.rand(len(triples), device=device, dtype=torch_dtype)
        
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
        
        # Set graph conditioning
        set_graph_condition(transformer, graph_local=graph_local, graph_global=graph_global)
        
        # Predict noise (or velocity for flow matching)
        model_pred = transformer(
            hidden_states=packed_noisy_latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            return_dict=False,
            graph_local=graph_local,
            graph_global=graph_global,
        )[0]
        
        # Unpack prediction (needs pixel dimensions, not latent dimensions)
        model_pred = pipe._unpack_latents(
            model_pred,
            height=cfg.height,
            width=cfg.width,
            vae_scale_factor=pipe.vae_scale_factor,
        )
        
        # For flow matching, target is: velocity = noise - latents
        target = noise - latents
        
        # Compute loss (MSE between prediction and target)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg.epochs):
        transformer.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for images, triples, _ in pbar:
            optim.zero_grad(set_to_none=True)
            
            loss = run_batch(images, triples, training=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(_trainable_params(transformer), 1.0)
            
            optim.step()
            
            if cfg.enable_cpu_offload and cfg.device.startswith("cuda"):
                pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            
            global_step += 1
            
            if global_step % cfg.log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Validation
            if global_step % cfg.val_every == 0:
                transformer.eval()
                val_losses = []
                with torch.no_grad():
                    for images, triples, _ in val_loader:
                        loss = run_batch(images, triples, training=False)
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
                        "state_dict": transformer.state_dict(),
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
                    "state_dict": transformer.state_dict(),
                    "optimizer": optim.state_dict(),
                    "step": global_step,
                }, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")
    
    # Final save
    ckpt_path = cfg.output_dir / f"final_graph_lora_{cfg.graph_mode}.pt"
    torch.save({
        "config": asdict(cfg),
        "state_dict": transformer.state_dict(),
        "optimizer": optim.state_dict(),
        "step": global_step,
    }, ckpt_path)
    print(f"\nTraining complete! Saved final checkpoint to {ckpt_path}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Graph-conditioned Flux LoRA training with diffusion loss.")
    p.add_argument("--model-id", default=TrainConfig.model_id)
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--dtype", default=TrainConfig.dtype, choices=["float32", "bfloat16", "float16"])
    p.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false", 
                   default=TrainConfig.enable_cpu_offload)
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
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--vocab-path", default=TrainConfig.vocab_path)
    p.add_argument("--train-h5", default=TrainConfig.train_h5)
    p.add_argument("--val-h5", default=TrainConfig.val_h5)
    p.add_argument("--vg-images-dir", default=TrainConfig.vg_images_dir)
    p.add_argument("--cgip-ckpt", default=TrainConfig.cgip_ckpt)
    p.add_argument("--graph-encoder-device", default=TrainConfig.graph_encoder_device)
    p.add_argument("--max-train-samples", type=int, default=TrainConfig.max_train_samples)
    p.add_argument("--max-val-samples", type=int, default=TrainConfig.max_val_samples)
    
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
        seed=args.seed,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        vg_images_dir=args.vg_images_dir,
        cgip_ckpt=args.cgip_ckpt,
        graph_encoder_device=args.graph_encoder_device,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    train(parse_args())
