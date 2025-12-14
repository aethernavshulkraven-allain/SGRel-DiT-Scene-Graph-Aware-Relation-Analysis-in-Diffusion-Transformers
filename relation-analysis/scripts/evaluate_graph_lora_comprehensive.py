#!/usr/bin/env python3
"""
Comprehensive evaluation for graph-conditioned FLUX LoRA.

Metrics:
1. CLIP Text Alignment (semantic consistency)
2. Object Detection + Spatial Reasoning (actual relationships)
3. Graph Sensitivity (MSE, LPIPS between g+ and g-)
4. Image Quality (FID optional)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
local_diffusers = REPO_ROOT / "diffusers" / "src"
if local_diffusers.exists() and str(local_diffusers) not in sys.path:
    sys.path.insert(0, str(local_diffusers))

from diffusers import FluxPipeline
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.flux.graph_conditioned_flux import patch_flux_for_graph, set_graph_condition
from relation_analysis.data.relations import default_predicate_map
from relation_analysis.prompt_builder import predicate_to_phrase


@dataclass
class EvalConfig:
    checkpoint: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    output_dir: Path = PROJECT_ROOT / "outputs" / "eval_comprehensive"
    
    # Test data
    val_jsonl: Path = PROJECT_ROOT / "scripts" / "splits" / "vg_quickwin_test.jsonl"
    max_samples: int = 160
    
    # Generation params
    height: int = 256
    width: int = 256
    num_steps: int = 4
    guidance: float = 0.0
    seeds: List[int] = None  # [0, 1, 2]
    
    # Model config
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    graph_mode: str = "token"
    block_start: int = 7
    block_end: int = 13
    
    # Evaluation flags
    compute_clip: bool = True
    compute_spatial: bool = True
    compute_lpips: bool = True
    save_images: bool = True
    
    # Graph encoder
    vocab_path: str = str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json")
    cgip_ckpt: str = str(REPO_ROOT / "SGDiff" / "pretrained" / "sip_vg.pt")
    

def make_negative_triple(triple: Tuple[str, str, str], all_predicates: List[str]) -> Tuple[str, str, str]:
    """Create negative triple by swapping or replacing predicate."""
    subject, predicate, obj = triple
    
    # Directional predicates: swap subject/object
    directional = {
        'left of': 'right of', 'right of': 'left of',
        'above': 'below', 'below': 'above',
        'in front of': 'behind', 'behind': 'in front of'
    }
    
    if predicate in directional:
        return (obj, directional[predicate], subject)
    
    # Non-directional: replace with different predicate
    import random
    neg_pred = random.choice([p for p in all_predicates if p != predicate])
    return (subject, neg_pred, obj)


def load_checkpoint(pipe, checkpoint_path: Path, graph_mode: str):
    """Load LoRA checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try different key formats
    if "lora_state" in state:
        lora_state = state["lora_state"]
    elif "state_dict" in state:
        lora_state = state["state_dict"]
    else:
        lora_state = state
    
    # Load into transformer
    transformer = pipe.transformer
    missing, unexpected = transformer.load_state_dict(lora_state, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    
    return pipe


def generate_images(
    pipe,
    graph_encoder,
    triples_pos: List[Tuple[str, str, str]],
    triples_neg: List[Tuple[str, str, str]],
    prompts: List[str],
    seeds: List[int],
    cfg: EvalConfig,
    torch_dtype,
) -> Dict[str, List[Image.Image]]:
    """Generate images for positive and negative graphs."""
    results = {"positive": [], "negative": [], "seeds": []}
    
    transformer = pipe.transformer
    device = torch.device(cfg.device)
    
    for seed in tqdm(seeds, desc="Seeds"):
        generator = torch.Generator(device=device).manual_seed(seed)
        
        for triple_pos, triple_neg, prompt in tqdm(
            zip(triples_pos, triples_neg, prompts),
            total=len(prompts),
            desc=f"Seed {seed}",
            leave=False
        ):
            # Encode graphs
            with torch.no_grad():
                graph_local_pos, graph_global_pos = graph_encoder.encode_batch([triple_pos])
                graph_local_neg, graph_global_neg = graph_encoder.encode_batch([triple_neg])
            
            graph_local_pos = graph_local_pos.to(device=device, dtype=torch_dtype)
            graph_global_pos = graph_global_pos.to(device=device, dtype=torch_dtype)
            graph_local_neg = graph_local_neg.to(device=device, dtype=torch_dtype)
            graph_global_neg = graph_global_neg.to(device=device, dtype=torch_dtype)
            
            # Generate with positive graph
            set_graph_condition(transformer, graph_local=graph_local_pos, graph_global=graph_global_pos)
            img_pos = pipe(
                prompt=prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance,
                generator=generator,
            ).images[0]
            
            # Generate with negative graph (same seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            set_graph_condition(transformer, graph_local=graph_local_neg, graph_global=graph_global_neg)
            img_neg = pipe(
                prompt=prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance,
                generator=generator,
            ).images[0]
            
            results["positive"].append(img_pos)
            results["negative"].append(img_neg)
            results["seeds"].append(seed)
    
    return results


def compute_clip_scores(
    images_pos: List[Image.Image],
    images_neg: List[Image.Image],
    prompts_pos: List[str],
    prompts_neg: List[str],
    device: str = "cuda:0"
) -> Dict[str, float]:
    """Compute CLIP text-image alignment scores."""
    from transformers import CLIPProcessor, CLIPModel
    
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    scores = []
    
    for img_pos, img_neg, prompt_pos, prompt_neg in tqdm(
        zip(images_pos, images_neg, prompts_pos, prompts_neg),
        desc="CLIP scoring"
    ):
        # Score positive image with both prompts
        inputs = processor(
            text=[prompt_pos, prompt_neg],
            images=[img_pos, img_pos],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # (2, 2)
            
        score_pos_correct = logits[0, 0].item()
        score_pos_wrong = logits[1, 1].item()
        
        # Score negative image with both prompts
        inputs = processor(
            text=[prompt_pos, prompt_neg],
            images=[img_neg, img_neg],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            
        score_neg_correct = logits[0, 0].item()
        score_neg_wrong = logits[1, 1].item()
        
        scores.append({
            "pos_correct": score_pos_correct,
            "pos_wrong": score_pos_wrong,
            "neg_correct": score_neg_correct,
            "neg_wrong": score_neg_wrong,
            "pos_margin": score_pos_correct - score_pos_wrong,
            "neg_margin": score_neg_correct - score_neg_wrong,
        })
    
    # Aggregate
    return {
        "clip_pos_correct": np.mean([s["pos_correct"] for s in scores]),
        "clip_pos_wrong": np.mean([s["pos_wrong"] for s in scores]),
        "clip_pos_margin": np.mean([s["pos_margin"] for s in scores]),
        "clip_neg_correct": np.mean([s["neg_correct"] for s in scores]),
        "clip_neg_wrong": np.mean([s["neg_wrong"] for s in scores]),
        "clip_neg_margin": np.mean([s["neg_margin"] for s in scores]),
        "clip_alignment_acc": np.mean([1.0 if s["pos_margin"] > 0 else 0.0 for s in scores]),
    }


def compute_lpips_scores(
    images_pos: List[Image.Image],
    images_neg: List[Image.Image],
    device: str = "cuda:0"
) -> Dict[str, float]:
    """Compute LPIPS perceptual distance between positive and negative images."""
    import lpips
    
    print("Loading LPIPS model...")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    distances = []
    
    for img_pos, img_neg in tqdm(zip(images_pos, images_neg), desc="LPIPS"):
        # Convert to tensors
        img_pos_t = torch.from_numpy(np.array(img_pos)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_neg_t = torch.from_numpy(np.array(img_neg)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        img_pos_t = img_pos_t.to(device)
        img_neg_t = img_neg_t.to(device)
        
        # Normalize to [-1, 1]
        img_pos_t = img_pos_t * 2 - 1
        img_neg_t = img_neg_t * 2 - 1
        
        with torch.no_grad():
            dist = loss_fn(img_pos_t, img_neg_t).item()
        
        distances.append(dist)
    
    return {
        "lpips_mean": np.mean(distances),
        "lpips_std": np.std(distances),
        "lpips_min": np.min(distances),
        "lpips_max": np.max(distances),
    }


def compute_mse_scores(
    images_pos: List[Image.Image],
    images_neg: List[Image.Image],
) -> Dict[str, float]:
    """Compute MSE between positive and negative images."""
    mses = []
    
    for img_pos, img_neg in zip(images_pos, images_neg):
        arr_pos = np.array(img_pos).astype(float)
        arr_neg = np.array(img_neg).astype(float)
        mse = np.mean((arr_pos - arr_neg) ** 2)
        mses.append(mse)
    
    return {
        "mse_mean": np.mean(mses),
        "mse_std": np.std(mses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    
    args = parser.parse_args()
    
    cfg = EvalConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir or PROJECT_ROOT / "outputs" / f"eval_{args.checkpoint.parent.name}",
        val_jsonl=args.val_jsonl or PROJECT_ROOT / "scripts" / "splits" / "vg_quickwin_test.jsonl",
        max_samples=args.max_samples,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        seeds=[int(s) for s in args.seeds.split(",")],
        compute_clip=not args.no_clip,
        compute_spatial=not args.no_spatial,
        compute_lpips=not args.no_lpips,
        save_images=not args.no_save_images,
    )
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation data
    print(f"Loading validation data from {cfg.val_jsonl}")
    with open(cfg.val_jsonl) as f:
        val_data = [json.loads(line) for line in f][:cfg.max_samples]
    
    triples_pos = [(item["triple"]["subject"], item["triple"]["predicate"], item["triple"]["object"]) 
                   for item in val_data]
    
    # Get all unique predicates from validation data
    all_predicates = sorted(set(t[1] for t in triples_pos))
    
    triples_neg = [make_negative_triple(t, all_predicates) for t in triples_pos]
    
    prompts_pos = [f"a photo of {s} {predicate_to_phrase(p)} {o}" for s, p, o in triples_pos]
    prompts_neg = [f"a photo of {s} {predicate_to_phrase(p)} {o}" for s, p, o in triples_neg]
    
    print(f"Loaded {len(val_data)} validation samples")
    print(f"Seeds: {cfg.seeds}")
    
    # Setup
    torch_dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    
    # Load pipeline
    print("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    
    # Patch for graph conditioning
    patch_flux_for_graph(
        pipe.transformer,
        mode=cfg.graph_mode,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {cfg.checkpoint}")
    load_checkpoint(pipe, cfg.checkpoint, cfg.graph_mode)
    
    # Load graph encoder
    print("Loading graph encoder...")
    graph_encoder = SGDiffGraphEncoder(cfg.vocab_path, cfg.cgip_ckpt, device="cpu")
    
    # Generate images
    print("\n" + "="*80)
    print("GENERATING IMAGES")
    print("="*80)
    results = generate_images(
        pipe, graph_encoder, triples_pos, triples_neg, prompts_pos,
        cfg.seeds, cfg, torch_dtype
    )
    
    images_pos = results["positive"]
    images_neg = results["negative"]
    
    print(f"Generated {len(images_pos)} positive images, {len(images_neg)} negative images")
    
    # Save images
    if cfg.save_images:
        print("Saving images...")
        img_dir = cfg.output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        
        for i, (img_pos, img_neg) in enumerate(zip(images_pos, images_neg)):
            img_pos.save(img_dir / f"sample_{i:04d}_pos.png")
            img_neg.save(img_dir / f"sample_{i:04d}_neg.png")
    
    # Compute metrics
    metrics = {}
    
    # MSE (always compute, cheap)
    print("\n" + "="*80)
    print("COMPUTING MSE")
    print("="*80)
    mse_scores = compute_mse_scores(images_pos, images_neg)
    metrics.update(mse_scores)
    
    # LPIPS
    if cfg.compute_lpips:
        print("\n" + "="*80)
        print("COMPUTING LPIPS")
        print("="*80)
        try:
            lpips_scores = compute_lpips_scores(images_pos, images_neg, cfg.device)
            metrics.update(lpips_scores)
        except Exception as e:
            print(f"LPIPS failed: {e}")
            print("Install lpips: pip install lpips")
    
    # CLIP
    if cfg.compute_clip:
        print("\n" + "="*80)
        print("COMPUTING CLIP SCORES")
        print("="*80)
        try:
            clip_scores = compute_clip_scores(
                images_pos, images_neg, prompts_pos, prompts_neg, cfg.device
            )
            metrics.update(clip_scores)
        except Exception as e:
            print(f"CLIP failed: {e}")
    
    # Save results
    results_path = cfg.output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for k, v in metrics.items():
        print(f"{k:30s}: {v:.4f}")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
