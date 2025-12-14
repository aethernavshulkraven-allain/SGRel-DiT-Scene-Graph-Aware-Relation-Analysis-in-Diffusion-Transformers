"""
Test graph-conditioned LoRA inference.
Generate images with different scene graphs to verify graph conditioning works.
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import sys
import gc
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import FluxPipeline
from relation_analysis.flux.graph_conditioned_flux import patch_flux_for_graph, set_graph_condition
from relation_analysis.flux.lora import LinearWithLoRA
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder


def _inject_lora_into_blocks(transformer, block_indices, rank: int, alpha: float):
    """Inject LoRA into FluxAttention projections (to_q/to_k/to_v/to_out[0])."""
    for idx in block_indices:
        if idx < 0 or idx >= len(transformer.transformer_blocks):
            continue
        block = transformer.transformer_blocks[idx]
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for name in ("to_q", "to_k", "to_v"):
            proj = getattr(attn, name, None)
            if isinstance(proj, torch.nn.Linear) and not isinstance(proj, LinearWithLoRA):
                setattr(attn, name, LinearWithLoRA(proj, rank=rank, alpha=alpha))
        to_out = getattr(attn, "to_out", None)
        if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0 and isinstance(to_out[0], torch.nn.Linear):
            if not isinstance(to_out[0], LinearWithLoRA):
                to_out[0] = LinearWithLoRA(to_out[0], rank=rank, alpha=alpha)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained LoRA checkpoint")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--graph-encoder-path", type=str, 
                       default="../../SGDiff/pretrained/sip_vg.pt")
    parser.add_argument("--vocab-path", type=str,
                       default="../../SGDiff/datasets/vg/vocab.json")
    parser.add_argument("--output-dir", type=str, default="./inference_results")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--graph-mode", type=str, default="token", choices=["token", "temb"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--block-start", type=int, default=7)
    parser.add_argument("--block-end", type=int, default=14)
    return parser.parse_args()


def load_graph_encoder(checkpoint_path, vocab_path, device):
    """Load pretrained SGDiff graph encoder."""
    encoder = SGDiffGraphEncoder(
        vocab_path=vocab_path,
        ckpt_path=checkpoint_path,
        device=device
    )
    return encoder


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    dtype = torch.bfloat16
    
    print("Loading Flux pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16  # Use fp16 to save memory
    )
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory optimizations (no CPU offload to avoid device conflicts)
    print("Enabling memory optimizations...")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Patch transformer with graph conditioning
    transformer = pipe.transformer
    patch_flux_for_graph(
        transformer, 
        mode=args.graph_mode,
        block_range=range(args.block_start, args.block_end)
    )
    
    # Inject LoRA
    print(f"Injecting LoRA (rank={args.lora_rank})...")
    _inject_lora_into_blocks(transformer, range(args.block_start, args.block_end), 
                            rank=args.lora_rank, alpha=16.0)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Debug: print available keys
    print(f"Available checkpoint keys: {list(checkpoint.keys())}")
    
    # Try to find the correct key
    if 'transformer_state_dict' in checkpoint:
        state_dict = checkpoint['transformer_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Checkpoint might be the state dict itself
        state_dict = checkpoint
    
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval()
    
    # Load graph encoder
    print("Loading graph encoder...")
    graph_encoder = load_graph_encoder(args.graph_encoder_path, args.vocab_path, device)
    
    # Test cases: same prompt, different graphs
    test_cases = [
        {
            "name": "person_riding_bike",
            "prompt": "a photo of a person riding a bike",
            "triplet": ("person", "riding", "bike"),
            "description": "Person actively riding a bike"
        },
        {
            "name": "person_next_to_bike",
            "prompt": "a photo of a person next to a bike",
            "triplet": ("person", "next to", "bike"),
            "description": "Person standing next to a bike"
        },
        {
            "name": "person_on_bike",
            "prompt": "a photo of a person on a bike",
            "triplet": ("person", "on", "bike"),
            "description": "Person sitting on a bike"
        },
        {
            "name": "dog_wearing_hat",
            "prompt": "a photo of a dog wearing a hat",
            "triplet": ("dog", "wearing", "hat"),
            "description": "Dog wearing a hat on its head"
        },
        {
            "name": "dog_next_to_hat",
            "prompt": "a photo of a dog next to a hat",
            "triplet": ("dog", "next to", "hat"),
            "description": "Dog standing beside a hat"
        },
        {
            "name": "person_holding_cup",
            "prompt": "a photo of a person holding a cup",
            "triplet": ("person", "holding", "cup"),
            "description": "Person holding a cup in their hand"
        },
        {
            "name": "person_drinking_from_cup",
            "prompt": "a photo of a person drinking from a cup",
            "triplet": ("person", "drinking from", "cup"),
            "description": "Person drinking from a cup"
        },
    ]
    
    print(f"\nGenerating {len(test_cases)} test images...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test_case['name']}")
        print(f"  Prompt: {test_case['prompt']}")
        print(f"  Graph: {test_case['triplet']}")
        print(f"  Description: {test_case['description']}")
        
        # Encode graph
        with torch.no_grad():
            # Encode the triplet using SGDiffGraphEncoder
            graph_local, graph_global = graph_encoder.encode_batch([test_case['triplet']])
            graph_local = graph_local.to(dtype=dtype)
            graph_global = graph_global.to(dtype=dtype)
            
            # Set graph conditioning
            set_graph_condition(transformer, graph_local=graph_local, graph_global=graph_global)
            
            # Generate image
            image = pipe(
                prompt=test_case['prompt'],
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=0.0,  # Schnell doesn't need guidance
            ).images[0]
            
            # Save image
            save_path = output_dir / f"{test_case['name']}.png"
            image.save(save_path)
            print(f"  Saved: {save_path}")
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n" + "=" * 80)
    print(f"✓ All images saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Compare images with similar prompts but different graphs")
    print("  2. Check if spatial arrangements match the relationships")
    print("  3. Example: 'riding' should show dynamic action vs 'next to' showing static placement")


if __name__ == "__main__":
    main()
