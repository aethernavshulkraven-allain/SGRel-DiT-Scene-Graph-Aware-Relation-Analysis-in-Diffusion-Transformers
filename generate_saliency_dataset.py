"""
Multi-GPU Saliency Map Dataset Generator

Generates 3 separate datasets with saliency maps from different layer groups:
- Early layers: [0, 1, 2, 3, 4, 5, 6]
- Middle layers: [7, 8, 9, 10, 11, 12]
- Late layers: [13, 14, 15, 16, 17, 18]

For 24 relationship classes with 2000 samples each.
"""

import json
import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse

# Add ConceptAttention to path
sys.path.insert(0, "/home/namanb/SBILab/CSE677/Project/ConceptAttention")
from concept_attention import ConceptAttentionFluxPipeline


# Layer configurations
LAYER_CONFIGS = {
    "early_layers": list(range(0, 7)),      # 0-6
    "middle_layers": list(range(7, 13)),    # 7-12
    "late_layers": list(range(13, 19))      # 13-18
}

SAMPLES_PER_CLASS = 2000
NUM_GPUS = 4
SAMPLES_PER_GPU = SAMPLES_PER_CLASS // NUM_GPUS  # 500 per GPU


def load_and_prepare_data(jsonl_path):
    """Load JSONL and organize by predicate."""
    print(f"Loading data from {jsonl_path}...")
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # Organize by predicate
    predicate_samples = defaultdict(list)
    for item in tqdm(data, desc="Organizing by predicate"):
        predicate = item["triple"]["predicate"]
        predicate_samples[predicate].append(item)
    
    # Get the 24 predicates
    predicates = sorted(predicate_samples.keys())
    assert len(predicates) == 24, f"Expected 24 predicates, got {len(predicates)}"
    
    print(f"\nFound {len(predicates)} predicates:")
    small_classes = []
    for pred in predicates:
        count = len(predicate_samples[pred])
        status = ""
        if count < SAMPLES_PER_CLASS:
            status = f" ⚠️  (will oversample {count} samples to reach {SAMPLES_PER_CLASS})"
            small_classes.append((pred, count))
        print(f"  {pred:20s}: {count:6d} samples{status}")
    
    if small_classes:
        print(f"\n⚠️  WARNING: {len(small_classes)} classes have fewer than {SAMPLES_PER_CLASS} samples.")
        print(f"   These classes will be oversampled (with shuffling) to reach the target count.")
        print(f"   Small classes: {', '.join([f'{p} ({c})' for p, c in small_classes])}")
    
    return predicate_samples, predicates


def sample_data_for_gpu(predicate_samples, predicates, gpu_id, samples_per_gpu, num_gpus):
    """
    Sample data for a specific GPU.
    Each GPU gets a different subset of the samples per class.

    This function ensures the per-GPU sample counts sum exactly to SAMPLES_PER_CLASS
    (distributes the remainder across the first few GPUs).
    For classes with >= SAMPLES_PER_CLASS: Each GPU gets a unique slice.
    For classes with < SAMPLES_PER_CLASS: Oversample the available samples and
    take the appropriate slice for this GPU.
    """
    gpu_data = {}

    # Compute per-GPU allocation so total equals SAMPLES_PER_CLASS
    base = SAMPLES_PER_CLASS // num_gpus
    rem = SAMPLES_PER_CLASS % num_gpus
    # Number of samples for this GPU
    samples_for_this_gpu = base + (1 if gpu_id < rem else 0)
    # Start index is sum of allocations of previous GPUs
    start_idx = base * gpu_id + min(gpu_id, rem)
    end_idx = start_idx + samples_for_this_gpu

    for pred in predicates:
        available = predicate_samples[pred]
        num_available = len(available)

        if num_available >= SAMPLES_PER_CLASS:
            # Enough samples: take a unique slice for this GPU
            gpu_data[pred] = available[start_idx:end_idx]
        else:
            # Not enough samples: oversample by repeating the dataset
            import random

            # Create an oversampled list by repeating available samples
            repetitions_needed = (SAMPLES_PER_CLASS + num_available - 1) // num_available
            oversampled = available * repetitions_needed

            # Shuffle to avoid order bias (different seed per GPU)
            random.seed(42 + gpu_id)
            random.shuffle(oversampled)

            # Take this GPU's slice from the oversampled data
            gpu_data[pred] = oversampled[start_idx:end_idx]

    return gpu_data


def generate_saliency_maps_worker(gpu_id, gpu_data, predicates, output_dir, layer_config_name, layer_indices, all_predicates):
    """
    Worker function that runs on a single GPU.
    
    Args:
        gpu_id: GPU device ID (0-3)
        gpu_data: Dictionary mapping predicate -> list of samples for this GPU
        predicates: List of predicate names to generate (filtered)
        output_dir: Base output directory
        layer_config_name: Name of layer config (early/middle/late)
        layer_indices: List of layer indices to use
        all_predicates: Full list of 24 predicates (for global class ID mapping)
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Starting worker on {device}")
    
    # Initialize pipeline on this GPU
    print(f"[GPU {gpu_id}] Loading ConceptAttentionFluxPipeline...")
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",
        device=device
    )
    
    # Create output directories
    layer_output_dir = Path(output_dir) / layer_config_name
    
    # Process each predicate
    for predicate in predicates:
        # Use global class ID from full 24-predicate list
        pred_idx = all_predicates.index(predicate)
        samples = gpu_data[predicate]
        
        if len(samples) == 0:
            print(f"[GPU {gpu_id}] Skipping {predicate} - no samples")
            continue
        
        # Create predicate directory
        pred_dir = layer_output_dir / f"class_{pred_idx:02d}_{predicate.replace('/', '_')}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save class metadata
        class_info = {
            "class_id": pred_idx,
            "predicate": predicate,
            "num_samples": len(samples),
            "layer_config": layer_config_name,
            "layer_indices": layer_indices
        }
        with open(pred_dir / "class_info.json", "w") as f:
            json.dump(class_info, f, indent=2)
        
        # Process each sample
        pbar = tqdm(
            samples, 
            desc=f"[GPU {gpu_id}] {predicate:15s}", 
            position=gpu_id,
            leave=True
        )
        
        for sample_idx, item in enumerate(pbar):
            try:
                prompt = item["prompt"]
                concepts = item["concepts"]
                
                # Calculate sample ID (offset by GPU to avoid collisions)
                global_sample_id = gpu_id * SAMPLES_PER_GPU + sample_idx
                
                # Generate saliency maps
                pipeline_output = pipeline.generate_image(
                    prompt=prompt,
                    concepts=concepts,
                    width=512,
                    height=512,
                    return_pil_heatmaps=False,  # Get numpy arrays
                    softmax=True,
                    layer_indices=layer_indices,
                    timesteps=list(range(4)),  # All timesteps
                    num_inference_steps=4,
                    seed=global_sample_id  # Use sample ID as seed for reproducibility
                )
                
                # Prepare data to save
                save_data = {
                    "saliency_maps": torch.from_numpy(pipeline_output.concept_heatmaps),  # (num_concepts, H, W)
                    "cross_attention_maps": torch.from_numpy(pipeline_output.cross_attention_maps),
                    "prompt": prompt,
                    "concepts": concepts,
                    "class_id": pred_idx,
                    "predicate": predicate,
                    "sample_id": global_sample_id,
                    "layer_config": layer_config_name,
                    "layer_indices": layer_indices,
                    "metadata": item["triple"]
                }
                
                # Save to disk
                save_path = pred_dir / f"sample_{global_sample_id:04d}.pt"
                torch.save(save_data, save_path)
                
                # Update progress bar
                pbar.set_postfix({"saved": str(save_path.name)})
                
            except Exception as e:
                print(f"\n[GPU {gpu_id}] Error processing sample {sample_idx} for {predicate}: {e}")
                continue
    
    print(f"[GPU {gpu_id}] Completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate saliency map dataset")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="/home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers/relation-analysis/outputs/stage_a/vg_stage_a_full.jsonl",
        help="Path to JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/namanb/SBILab/CSE677/Project/saliency_datasets",
        help="Base output directory"
    )
    parser.add_argument(
        "--layer-config",
        type=str,
        choices=["early_layers", "middle_layers", "late_layers", "all"],
        default="all",
        help="Which layer configuration to generate"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2000,
        help="Number of samples per class"
    )
    parser.add_argument(
        "--predicates",
        type=str,
        nargs="+",
        default=None,
        help="Specific predicates to generate (e.g., 'on' 'in' 'wearing'). If not specified, generates all 24 classes."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip classes that already have the target number of samples"
    )
    
    args = parser.parse_args()
    
    # Update global variables
    global NUM_GPUS, SAMPLES_PER_CLASS, SAMPLES_PER_GPU
    NUM_GPUS = args.num_gpus
    SAMPLES_PER_CLASS = args.samples_per_class
    SAMPLES_PER_GPU = SAMPLES_PER_CLASS // NUM_GPUS
    
    # Load and organize data
    predicate_samples, predicates = load_and_prepare_data(args.jsonl_path)
    
    # Filter predicates if specified
    if args.predicates:
        # Validate predicates
        invalid = [p for p in args.predicates if p not in predicates]
        if invalid:
            print(f"ERROR: Invalid predicates: {invalid}")
            print(f"Valid predicates: {predicates}")
            return
        predicates_to_generate = args.predicates
        print(f"\n🎯 Generating only {len(predicates_to_generate)} classes: {predicates_to_generate}")
    else:
        predicates_to_generate = predicates
        print(f"\n🎯 Generating all {len(predicates_to_generate)} classes")
    
    # Determine which layer configs to run
    if args.layer_config == "all":
        configs_to_run = list(LAYER_CONFIGS.keys())
    else:
        configs_to_run = [args.layer_config]
    
    print(f"\nGenerating datasets for: {configs_to_run}")
    print(f"Using {NUM_GPUS} GPUs with {SAMPLES_PER_GPU} samples per GPU per class")
    print(f"Output directory: {args.output_dir}\n")
    
    # Process each layer configuration
    for layer_config_name in configs_to_run:
        layer_indices = LAYER_CONFIGS[layer_config_name]
        print(f"\n{'='*80}")
        print(f"Processing {layer_config_name}: layers {layer_indices}")
        print(f"{'='*80}\n")
        
        # Check for resume mode - determine which predicates to actually generate for this layer
        predicates_for_this_layer = predicates_to_generate.copy()
        
        if args.resume:
            from pathlib import Path
            layer_output_dir = Path(args.output_dir) / layer_config_name
            completed_predicates = []
            
            for pred in predicates_for_this_layer:
                pred_idx = predicates.index(pred)
                pred_dir = layer_output_dir / f"class_{pred_idx:02d}_{pred.replace('/', '_')}"
                if pred_dir.exists():
                    num_samples = len(list(pred_dir.glob("sample_*.pt")))
                    if num_samples >= SAMPLES_PER_CLASS:
                        completed_predicates.append(pred)
                        print(f"✓ Skipping {pred} - already has {num_samples} samples")
            
            # Remove completed predicates for this layer
            predicates_for_this_layer = [p for p in predicates_for_this_layer if p not in completed_predicates]
            
            if not predicates_for_this_layer:
                print(f"\n✓ All classes already completed for {layer_config_name}!")
                continue
            
            print(f"\n📝 Resuming generation for {len(predicates_for_this_layer)} remaining classes")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Create processes for each GPU
        processes = []
        for gpu_id in range(NUM_GPUS):
            # Sample data for this GPU (only for predicates we're generating for this layer)
            gpu_data = sample_data_for_gpu(predicate_samples, predicates_for_this_layer, gpu_id, SAMPLES_PER_GPU, NUM_GPUS)
            
            # Create process
            p = mp.Process(
                target=generate_saliency_maps_worker,
                args=(gpu_id, gpu_data, predicates_for_this_layer, args.output_dir, layer_config_name, layer_indices, predicates)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"\n{layer_config_name} dataset generation complete!")
    
    print("\n" + "="*80)
    print("All datasets generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
