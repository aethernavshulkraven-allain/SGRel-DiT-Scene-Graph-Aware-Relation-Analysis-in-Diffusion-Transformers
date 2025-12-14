"""
Multi-GPU Saliency Map Dataset Generator with SMART RESUME

This version properly handles partial class completion:
- Counts existing samples per class
- Generates ONLY the remaining samples needed
- Avoids regenerating already-completed samples

For 24 relationship classes with configurable samples per class.
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


def get_existing_sample_counts(output_dir, layer_config_name, predicates, all_predicates):
    """
    Count existing samples for each predicate in a layer config.
    Also extracts used prompts/metadata to prevent duplicates.
    
    Returns:
        dict: {predicate: {
            'existing': count, 
            'existing_ids': set, 
            'needed': remaining, 
            'safe_start_id': id,
            'used_prompts': set,
            'used_metadata_hashes': set
        }}
    """
    layer_output_dir = Path(output_dir) / layer_config_name
    sample_counts = {}
    
    for pred in predicates:
        pred_idx = all_predicates.index(pred)
        pred_dir = layer_output_dir / f"class_{pred_idx:02d}_{pred.replace('/', '_')}"
        
        if pred_dir.exists():
            existing_files = list(pred_dir.glob("sample_*.pt"))
            existing_count = len(existing_files)
            
            # Collect ALL existing sample IDs to avoid collisions
            existing_ids = set()
            used_prompts = set()
            used_metadata_hashes = set()
            max_id = -1
            
            print(f"  Loading {existing_count} existing samples for {pred} to detect duplicates...")
            for f in existing_files:
                try:
                    sample_id = int(f.stem.replace("sample_", ""))
                    existing_ids.add(sample_id)
                    max_id = max(max_id, sample_id)
                    
                    # Load the file to extract prompt/metadata
                    try:
                        data = torch.load(f, map_location='cpu')
                        if 'prompt' in data:
                            used_prompts.add(data['prompt'])
                        if 'metadata' in data:
                            # Create a hash of the metadata to identify unique triplets
                            metadata = data['metadata']
                            metadata_hash = f"{metadata.get('subject_id', '')}_{metadata.get('predicate_id', '')}_{metadata.get('object_id', '')}"
                            used_metadata_hashes.add(metadata_hash)
                    except Exception as e:
                        # If we can't load the file, just track the ID
                        pass
                except:
                    continue
            
            # Use a safe starting ID: max_id + 1000 to guarantee no collision
            safe_start_id = max_id + 1000 if max_id >= 0 else 0
            
            sample_counts[pred] = {
                'existing': existing_count,
                'existing_ids': existing_ids,
                'needed': max(0, SAMPLES_PER_CLASS - existing_count),
                'safe_start_id': safe_start_id,
                'used_prompts': used_prompts,
                'used_metadata_hashes': used_metadata_hashes
            }
        else:
            sample_counts[pred] = {
                'existing': 0,
                'existing_ids': set(),
                'needed': SAMPLES_PER_CLASS,
                'safe_start_id': 0,
                'used_prompts': set(),
                'used_metadata_hashes': set()
            }
    
    return sample_counts


def sample_data_for_gpu_resume(predicate_samples, pred, gpu_id, samples_needed_total, num_gpus, used_prompts, used_metadata_hashes):
    """
    Sample data for a specific GPU when resuming.
    FILTERS OUT already-used prompts/metadata to prevent duplicates.
    If not enough unused samples, OVERSAMPLES with different seeds.
    
    Args:
        predicate_samples: All available samples for all predicates
        pred: The specific predicate to sample for
        gpu_id: GPU device ID
        samples_needed_total: Total remaining samples needed for this predicate
        num_gpus: Total number of GPUs
        used_prompts: Set of prompts already used
        used_metadata_hashes: Set of metadata hashes already used
    """
    available = predicate_samples[pred]
    
    # FILTER OUT already-used samples to prevent duplicates
    filtered_available = []
    for item in available:
        prompt = item["prompt"]
        metadata = item["triple"]
        metadata_hash = f"{metadata.get('subject_id', '')}_{metadata.get('predicate_id', '')}_{metadata.get('object_id', '')}"
        
        # Skip if we've already generated this prompt or metadata
        if prompt not in used_prompts and metadata_hash not in used_metadata_hashes:
            filtered_available.append(item)
    
    num_available = len(filtered_available)
    
    # Handle case where ALL prompts have been used (no unused samples)
    if num_available == 0:
        print(f"  ⚠️  WARNING: {pred} has 0 unused samples - all prompts already used!")
        print(f"     Will RE-USE all {len(available)} available prompts with DIFFERENT SEEDS to generate {samples_needed_total} more samples.")
        # Fall back to using ALL available prompts (don't filter)
        filtered_available = available
        num_available = len(filtered_available)
    
    if num_available < samples_needed_total:
        print(f"  ⚠️  WARNING: {pred} only has {num_available} unused samples but needs {samples_needed_total}")
        print(f"     Will OVERSAMPLE the {num_available} samples with different seeds to reach target.")
        
        # OVERSAMPLE: Repeat the filtered samples with shuffling
        import random
        repetitions_needed = (samples_needed_total + num_available - 1) // num_available
        oversampled = filtered_available * repetitions_needed
        
        # Shuffle with GPU-specific seed for variety
        random.seed(42 + gpu_id)
        random.shuffle(oversampled)
        
        # Use oversampled list
        filtered_available = oversampled
        num_available = len(filtered_available)
    
    # Distribute remaining samples across GPUs
    base = samples_needed_total // num_gpus
    rem = samples_needed_total % num_gpus
    samples_for_this_gpu = base + (1 if gpu_id < rem else 0)
    start_idx = base * gpu_id + min(gpu_id, rem)
    end_idx = start_idx + samples_for_this_gpu
    
    # Take clean slice from filtered (possibly oversampled) data
    return filtered_available[start_idx:end_idx]


def generate_saliency_maps_worker(gpu_id, gpu_data, predicates_to_process, pred_info, output_dir, layer_config_name, layer_indices, all_predicates):
    """
    Worker function that runs on a single GPU with SMART RESUME support.
    
    Args:
        gpu_id: GPU device ID (0-3)
        gpu_data: Dictionary mapping predicate -> list of samples for this GPU
        predicates_to_process: Ordered list of predicates to process (determines generation order)
        pred_info: Dictionary mapping predicate -> {existing, max_sample_id, needed}
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
    
    # Process each predicate IN ORDER (reversed: class_23 -> class_00)
    for predicate in predicates_to_process:
        if predicate not in gpu_data:
            continue
        samples = gpu_data[predicate]
        pred_idx = all_predicates.index(predicate)
        
        if len(samples) == 0:
            print(f"[GPU {gpu_id}] Skipping {predicate} - no samples to generate")
            continue
        
        # Get resume info
        info = pred_info[predicate]
        existing_count = info['existing']
        existing_ids = info['existing_ids']
        safe_start_id = info['safe_start_id']
        
        # Create predicate directory
        pred_dir = layer_output_dir / f"class_{pred_idx:02d}_{predicate.replace('/', '_')}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save/update class metadata
        class_info = {
            "class_id": pred_idx,
            "predicate": predicate,
            "num_samples": existing_count + len(samples),
            "layer_config": layer_config_name,
            "layer_indices": layer_indices
        }
        with open(pred_dir / "class_info.json", "w") as f:
            json.dump(class_info, f, indent=2)
        
        # Calculate starting sample ID using SAFE offset to avoid collisions
        # Use safe_start_id (max_existing + 1000) and distribute across GPUs
        samples_per_gpu_for_pred = (info['needed'] + NUM_GPUS - 1) // NUM_GPUS
        gpu_start_id = safe_start_id + (gpu_id * samples_per_gpu_for_pred)
        
        # Track seen prompts within this batch for duplicate detection
        seen_prompts_this_batch = {}
        
        # Process each sample
        pbar = tqdm(
            samples, 
            desc=f"[GPU {gpu_id}] {predicate:15s} (resume from {gpu_start_id})", 
            position=gpu_id,
            leave=True
        )
        
        for local_idx, item in enumerate(pbar):
            try:
                prompt = item["prompt"]
                concepts = item["concepts"]
                
                # Calculate global sample ID (continue from existing)
                global_sample_id = gpu_start_id + local_idx
                
                # DOUBLE SAFETY CHECK: Skip if ID already exists or file exists
                while global_sample_id in existing_ids:
                    global_sample_id += NUM_GPUS * samples_per_gpu_for_pred
                
                save_path = pred_dir / f"sample_{global_sample_id:04d}.pt"
                if save_path.exists():
                    pbar.set_postfix({"status": "skip-exists"})
                    continue
                
                # Add to existing_ids to prevent collision within this run
                existing_ids.add(global_sample_id)
                
                # For oversampled prompts: use a DIFFERENT seed each time
                # Track how many times we've seen this prompt in this batch
                if prompt in seen_prompts_this_batch:
                    seen_prompts_this_batch[prompt] += 1
                    # Use different seed: base_id + repetition_count * large_offset
                    seed = global_sample_id + (seen_prompts_this_batch[prompt] * 10000)
                else:
                    seen_prompts_this_batch[prompt] = 0
                    seed = global_sample_id
                
                # Generate saliency maps
                pipeline_output = pipeline.generate_image(
                    prompt=prompt,
                    concepts=concepts,
                    width=512,
                    height=512,
                    return_pil_heatmaps=False,
                    softmax=True,
                    layer_indices=layer_indices,
                    timesteps=list(range(4)),
                    num_inference_steps=4,
                    seed=seed  # Use varied seed for oversampled prompts
                )
                
                # Prepare data to save
                save_data = {
                    "saliency_maps": torch.from_numpy(pipeline_output.concept_heatmaps),
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
                torch.save(save_data, save_path)
                
                # Update progress bar
                pbar.set_postfix({"saved": str(save_path.name)})
                
            except Exception as e:
                print(f"\n[GPU {gpu_id}] Error processing sample for {predicate}: {e}")
                continue
    
    print(f"[GPU {gpu_id}] Completed!")


def main():
    parser = argparse.ArgumentParser(description="Generate saliency map dataset with smart resume")
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
        help="Specific predicates to generate"
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
        invalid = [p for p in args.predicates if p not in predicates]
        if invalid:
            print(f"ERROR: Invalid predicates: {invalid}")
            return
        predicates_to_generate = args.predicates
        print(f"\n🎯 Generating only {len(predicates_to_generate)} classes: {predicates_to_generate}")
    else:
        predicates_to_generate = predicates
    
    # Determine which layer configs to run
    if args.layer_config == "all":
        configs_to_run = list(LAYER_CONFIGS.keys())
    else:
        configs_to_run = [args.layer_config]
    
    print(f"\nGenerating datasets for: {configs_to_run}")
    print(f"Using {NUM_GPUS} GPUs")
    print(f"Output directory: {args.output_dir}\n")
    
    # Process each layer configuration
    for layer_config_name in configs_to_run:
        layer_indices = LAYER_CONFIGS[layer_config_name]
        print(f"\n{'='*80}")
        print(f"Processing {layer_config_name}: layers {layer_indices}")
        print(f"{'='*80}\n")
        
        # SMART RESUME: Count existing samples
        sample_counts = get_existing_sample_counts(
            args.output_dir, 
            layer_config_name, 
            predicates_to_generate,
            predicates
        )
        
        print("Current status:")
        predicates_needing_samples = []
        for pred in predicates_to_generate:
            info = sample_counts[pred]
            status = "✓" if info['needed'] == 0 else f"needs {info['needed']} more"
            print(f"  {pred:20s}: {info['existing']:4d}/{SAMPLES_PER_CLASS} ({status})")
            if info['needed'] > 0:
                predicates_needing_samples.append(pred)
        
        if not predicates_needing_samples:
            print(f"\n✓ All classes complete for {layer_config_name}!")
            continue
        
        # REVERSE ORDER: Start from highest class number (class_23) and work backwards
        predicates_needing_samples_reversed = sorted(
            predicates_needing_samples,
            key=lambda p: predicates.index(p),
            reverse=True
        )
        
        print(f"\n📝 Generating remaining samples for {len(predicates_needing_samples_reversed)} classes")
        print(f"   Generation order: {' → '.join(predicates_needing_samples_reversed[:5])}{'...' if len(predicates_needing_samples_reversed) > 5 else ''}")
        total_remaining = sum(sample_counts[p]['needed'] for p in predicates_needing_samples_reversed)
        print(f"   Total samples to generate: {total_remaining}")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Create processes for each GPU
        processes = []
        for gpu_id in range(NUM_GPUS):
            # Sample data for this GPU (only remaining samples, filtered for duplicates)
            gpu_data = {}
            for pred in predicates_needing_samples_reversed:
                info = sample_counts[pred]
                samples = sample_data_for_gpu_resume(
                    predicate_samples, 
                    pred, 
                    gpu_id, 
                    info['needed'],
                    NUM_GPUS,
                    info['used_prompts'],
                    info['used_metadata_hashes']
                )
                gpu_data[pred] = samples
            
            # Create process (pass reversed list to maintain order in worker)
            p = mp.Process(
                target=generate_saliency_maps_worker,
                args=(gpu_id, gpu_data, predicates_needing_samples_reversed, sample_counts, args.output_dir, layer_config_name, layer_indices, predicates)
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
