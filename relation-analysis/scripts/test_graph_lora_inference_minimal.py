"""
Minimal memory footprint inference for graph-conditioned LoRA.
Uses the same setup as training.
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
from relation_analysis.flux.lora import inject_lora
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder


def _inject_lora_into_blocks(transformer, block_indices, rank: int, alpha: float):
    """Inject LoRA into middle blocks' attention layers."""
    targets = ["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    for idx in block_indices:
        if idx < 0 or idx >= len(transformer.transformer_blocks):
            continue
        inject_lora(transformer.transformer_blocks[idx], targets, rank=rank, alpha=alpha)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--graph-encoder-path", type=str, default="../../SGDiff/pretrained/sip_vg.pt")
    parser.add_argument("--vocab-path", type=str, default="../../SGDiff/datasets/vg/vocab.json")
    parser.add_argument("--output-dir", type=str, default="./inference_results")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Check GPU memory before starting
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB\n")
    
    # Load with minimal memory - same as training
    print("Loading Flux pipeline (this may take a while)...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 like training
        low_cpu_mem_usage=True,      # Load weights progressively
    )
    
    print("Moving transformer to device...")
    pipe.transformer.to(device)
    pipe.vae.to(device) 
    pipe.text_encoder.to(device) if pipe.text_encoder else None
    pipe.text_encoder_2.to(device) if pipe.text_encoder_2 else None
    
    transformer = pipe.transformer
    
    # Patch and inject LoRA - same as training
    print("Patching transformer with graph conditioning...")
    patch_flux_for_graph(transformer, mode="token", block_range=range(7, 14))
    
    print(f"Injecting LoRA (rank={args.lora_rank})...")
    _inject_lora_into_blocks(transformer, range(7, 14), rank=args.lora_rank, alpha=16.0)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # Load to CPU first
    
    # Find state dict
    if 'transformer_state_dict' in checkpoint:
        state_dict = checkpoint['transformer_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    print("Loading state dict...")
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval()
    
    # Clear checkpoint from memory
    del checkpoint, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"After loading - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB\n")
    
    # Load graph encoder
    print("Loading graph encoder...")
    graph_encoder = SGDiffGraphEncoder(
        vocab_path=args.vocab_path,
        ckpt_path=args.graph_encoder_path,
        device=device
    )
    
    # Minimal test cases
    test_cases = [
        {
            "name": "person_riding_bike",
            "prompt": "a photo of a person riding a bike",
            "triplet": ("person", "riding", "bike"),
        },
        {
            "name": "person_next_to_bike",
            "prompt": "a photo of a person next to a bike",
            "triplet": ("person", "next to", "bike"),
        },
        {
            "name": "dog_wearing_hat",
            "prompt": "a photo of a dog wearing a hat",
            "triplet": ("dog", "wearing", "hat"),
        },
    ]
    
    print(f"\nGenerating {len(test_cases)} test images...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test_case['name']}")
        print(f"  Prompt: {test_case['prompt']}")
        
        with torch.no_grad():
            # Encode graph
            graph_local, graph_global = graph_encoder.encode_batch([test_case['triplet']])
            graph_local = graph_local.to(dtype=torch.bfloat16)
            graph_global = graph_global.to(dtype=torch.bfloat16)
            
            # Set graph conditioning
            set_graph_condition(transformer, graph_local=graph_local, graph_global=graph_global)
            
            print(f"  Generating {args.height}x{args.width} image...")
            print(f"  Memory before generation: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            
            # Generate
            image = pipe(
                prompt=test_case['prompt'],
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_steps,
                guidance_scale=0.0,
            ).images[0]
            
            # Save
            save_path = output_dir / f"{test_case['name']}.png"
            image.save(save_path)
            print(f"  ✓ Saved: {save_path}")
            
            # Aggressive memory cleanup
            del image
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  Memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 80)
    print(f"✓ All images saved to: {output_dir}")


if __name__ == "__main__":
    main()
