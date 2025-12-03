"""
Dataset loader and utilities for the generated saliency map datasets.
"""

import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class SaliencyMapDataset(Dataset):
    """
    PyTorch Dataset for loading saliency maps.
    
    Args:
        dataset_path: Path to dataset (e.g., 'saliency_datasets/early_layers')
        predicates: Optional list of specific predicates to load. If None, loads all.
        transform: Optional transform to apply to saliency maps
    """
    
    def __init__(self, dataset_path: str, predicates: List[str] = None, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load all samples
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Iterate through class directories
        class_dirs = sorted(self.dataset_path.glob("class_*"))
        
        for class_dir in class_dirs:
            # Load class info
            class_info_path = class_dir / "class_info.json"
            if not class_info_path.exists():
                continue
            
            with open(class_info_path, "r") as f:
                class_info = json.load(f)
            
            predicate = class_info["predicate"]
            class_id = class_info["class_id"]
            
            # Skip if predicate filtering is enabled and this predicate is not in the list
            if predicates is not None and predicate not in predicates:
                continue
            
            self.class_to_idx[predicate] = class_id
            self.idx_to_class[class_id] = predicate
            
            # Load all samples for this class
            sample_files = sorted(class_dir.glob("sample_*.pt"))
            for sample_file in sample_files:
                self.samples.append({
                    "path": sample_file,
                    "class_id": class_id,
                    "predicate": predicate
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load the saved data
        data = torch.load(sample_info["path"])
        
        # Extract saliency maps and label
        saliency_maps = data["saliency_maps"]  # (num_concepts, H, W)
        label = data["class_id"]
        
        if self.transform:
            saliency_maps = self.transform(saliency_maps)
        
        return {
            "saliency_maps": saliency_maps,
            "cross_attention_maps": data["cross_attention_maps"],
            "label": label,
            "predicate": data["predicate"],
            "prompt": data["prompt"],
            "concepts": data["concepts"],
            "metadata": data["metadata"]
        }
    
    def get_class_distribution(self):
        """Get the distribution of samples per class."""
        distribution = {}
        for sample in self.samples:
            pred = sample["predicate"]
            distribution[pred] = distribution.get(pred, 0) + 1
        return distribution


def verify_dataset(dataset_path: str):
    """
    Verify the integrity of a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
    """
    dataset_path = Path(dataset_path)
    
    print(f"\nVerifying dataset: {dataset_path}")
    print("="*80)
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    class_dirs = sorted(dataset_path.glob("class_*"))
    
    if len(class_dirs) == 0:
        print("ERROR: No class directories found")
        return False
    
    print(f"Found {len(class_dirs)} class directories\n")
    
    total_samples = 0
    all_valid = True
    
    for class_dir in class_dirs:
        class_info_path = class_dir / "class_info.json"
        
        if not class_info_path.exists():
            print(f"WARNING: Missing class_info.json in {class_dir.name}")
            all_valid = False
            continue
        
        with open(class_info_path, "r") as f:
            class_info = json.load(f)
        
        predicate = class_info["predicate"]
        class_id = class_info["class_id"]
        layer_config = class_info.get("layer_config", "unknown")
        
        sample_files = sorted(class_dir.glob("sample_*.pt"))
        num_samples = len(sample_files)
        total_samples += num_samples
        
        # Verify a sample file
        if num_samples > 0:
            try:
                sample_data = torch.load(sample_files[0])
                saliency_shape = sample_data["saliency_maps"].shape
                
                print(f"Class {class_id:2d} | {predicate:20s} | {num_samples:4d} samples | Shape: {saliency_shape}")
            except Exception as e:
                print(f"ERROR loading sample from {class_dir.name}: {e}")
                all_valid = False
        else:
            print(f"Class {class_id:2d} | {predicate:20s} | {num_samples:4d} samples | WARNING: No samples!")
            all_valid = False
    
    print("\n" + "="*80)
    print(f"Total samples: {total_samples}")
    print(f"Status: {'✓ VALID' if all_valid else '✗ ISSUES FOUND'}")
    print("="*80)
    
    return all_valid


def load_sample(sample_path: str):
    """Load and display information about a single sample."""
    data = torch.load(sample_path)
    
    print(f"\nSample: {sample_path}")
    print("="*80)
    print(f"Predicate: {data['predicate']}")
    print(f"Class ID: {data['class_id']}")
    print(f"Prompt: {data['prompt']}")
    print(f"Concepts: {data['concepts']}")
    print(f"Layer config: {data['layer_config']}")
    print(f"Layer indices: {data['layer_indices']}")
    print(f"Saliency maps shape: {data['saliency_maps'].shape}")
    print(f"Cross attention maps shape: {data['cross_attention_maps'].shape}")
    print(f"Metadata: {data['metadata']}")
    print("="*80)
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify and explore saliency map datasets")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., saliency_datasets/early_layers)"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["verify", "load", "stats"],
        default="verify",
        help="Action to perform"
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        help="Path to specific sample (for 'load' action)"
    )
    
    args = parser.parse_args()
    
    if args.action == "verify":
        verify_dataset(args.dataset_path)
    
    elif args.action == "stats":
        dataset = SaliencyMapDataset(args.dataset_path)
        distribution = dataset.get_class_distribution()
        
        print("\nClass Distribution:")
        print("="*80)
        for predicate, count in sorted(distribution.items()):
            print(f"{predicate:20s}: {count:4d} samples")
        print("="*80)
    
    elif args.action == "load":
        if args.sample_path:
            load_sample(args.sample_path)
        else:
            print("ERROR: --sample-path required for 'load' action")
