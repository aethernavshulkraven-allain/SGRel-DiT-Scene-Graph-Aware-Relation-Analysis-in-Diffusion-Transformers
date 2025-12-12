"""
Quick test script to verify dataset loading and model creation
"""

import torch
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, '/home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers/relation-analysis/')

from train_wideresnet_saliency import (
    load_dataset, 
    SaliencyRelationDataset, 
    WideResNet
)

def test_dataset_loading():
    """Test dataset loading"""
    print("="*60)
    print("Testing Dataset Loading")
    print("="*60)
    
    data_dir = '../../../saliency_datasets/early_layers'
    
    try:
        train_files, val_files, test_files, train_labels, val_labels, test_labels, class_names = load_dataset(
            data_dir, test_size=0.15, val_size=0.15, random_state=42
        )
        
        print(f"\n✓ Successfully loaded dataset")
        print(f"  - {len(class_names)} classes")
        print(f"  - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return train_files, train_labels, class_names
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return None, None, None


def test_fusion_modes(train_files, train_labels):
    """Test all fusion modes"""
    print("\n" + "="*60)
    print("Testing Fusion Modes")
    print("="*60)
    
    fusion_modes = ['concat', 'saliency_only', 'attention_only', 'add', 'multiply', 'max', 'weighted']
    
    for mode in fusion_modes:
        try:
            dataset = SaliencyRelationDataset(
                train_files[:10],  # Test with first 10 files
                train_labels[:10],
                fusion_mode=mode
            )
            
            sample, label = dataset[0]
            print(f"  {mode:20s} → shape: {sample.shape}, label: {label}")
            
        except Exception as e:
            print(f"  {mode:20s} → ✗ Error: {e}")


def test_model_creation():
    """Test WideResNet model creation"""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)
    
    configs = [
        (3, 28, 10, "WideResNet-28-10 (3 channels)"),
        (6, 28, 10, "WideResNet-28-10 (6 channels)"),
        (3, 40, 10, "WideResNet-40-10 (3 channels)"),
        (3, 28, 2, "WideResNet-28-2 (3 channels)"),
    ]
    
    for in_channels, depth, widen_factor, desc in configs:
        try:
            model = WideResNet(
                depth=depth,
                num_classes=24,
                widen_factor=widen_factor,
                dropRate=0.3,
                in_channels=in_channels
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            dummy_input = torch.randn(2, in_channels, 32, 32)
            output = model(dummy_input)
            
            print(f"  ✓ {desc}")
            print(f"    Parameters: {num_params:,}")
            print(f"    Output shape: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ {desc} - Error: {e}")


def test_dataloader():
    """Test DataLoader"""
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    data_dir = 'saliency_datasets/early_layers'
    
    try:
        train_files, val_files, test_files, train_labels, val_labels, test_labels, class_names = load_dataset(
            data_dir, test_size=0.15, val_size=0.15, random_state=42
        )
        
        # Create small dataset for testing
        dataset = SaliencyRelationDataset(
            train_files[:100],
            train_labels[:100],
            fusion_mode='concat'
        )
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        
        # Get one batch
        batch_data, batch_labels = next(iter(loader))
        
        print(f"  ✓ DataLoader working")
        print(f"    Batch data shape: {batch_data.shape}")
        print(f"    Batch labels shape: {batch_labels.shape}")
        print(f"    Label range: {batch_labels.min()} - {batch_labels.max()}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("WideResNet Training - System Test")
    print("="*60 + "\n")
    
    # Test 1: Dataset loading
    train_files, train_labels, class_names = test_dataset_loading()
    
    if train_files is not None:
        # Test 2: Fusion modes
        test_fusion_modes(train_files, train_labels)
        
        # Test 3: Model creation
        test_model_creation()
        
        # Test 4: DataLoader
        test_dataloader()
        
        print("\n" + "="*60)
        print("All Tests Completed!")
        print("="*60)
        print("\nYou can now run:")
        print("  python3 train_wideresnet_saliency.py --fusion_mode concat")
        print("or")
        print("  ./run_all_fusion_experiments.sh")
        print("="*60 + "\n")
    else:
        print("\n✗ Dataset loading failed. Please check your data directory.")
