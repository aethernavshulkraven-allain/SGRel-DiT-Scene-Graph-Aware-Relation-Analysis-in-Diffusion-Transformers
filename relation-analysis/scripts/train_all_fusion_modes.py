#!/usr/bin/env python3
"""
Train WideResNet with all fusion modes - suitable for nohup execution
Run with: nohup python3 train_all_fusion_modes.py > training_all.log 2>&1 </dev/null & disown
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import argparse
import torch
import numpy as np

# Import the main training function and components
from train_wideresnet_saliency import main as train_main

# Configuration
DATA_DIR = "/home/namanb/SBILab/CSE677/Project/saliency_datasets/early_layers"
OUTPUT_DIR = "wideresnet_experiments"
EPOCHS = 50
BATCH_SIZE = 64
DEPTH = 28
WIDEN_FACTOR = 10
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3
NUM_WORKERS = 4
SEED = 42

# All fusion modes to test
FUSION_MODES = [
    "concat",
    "saliency_only", 
    "attention_only",
    "add",
    "multiply",
    "max",
    "weighted"
]


def run_training(fusion_mode):
    """Run training for a specific fusion mode"""
    print("=" * 80)
    print(f"Starting training: {fusion_mode}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        # Create args object mimicking argparse
        args = argparse.Namespace(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            test_size=0.15,
            val_size=0.15,
            fusion_mode=fusion_mode,
            depth=DEPTH,
            widen_factor=WIDEN_FACTOR,
            dropout=DROPOUT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            num_workers=NUM_WORKERS,
            seed=SEED
        )
        
        # Set random seeds
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
        
        # Run training
        train_main(args)
        
        print("\n" + "=" * 80)
        print(f"✓ Completed: {fusion_mode}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        sys.stdout.flush()
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Failed: {fusion_mode}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")
        sys.stdout.flush()
        
        return False


def main():
    """Main function to run all experiments"""
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("WideResNet Training - All Fusion Modes")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total fusion modes: {len(FUSION_MODES)}")
    print(f"Epochs per mode: {EPOCHS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")
    sys.stdout.flush()
    
    results = {}
    
    # Run training for each fusion mode
    for i, fusion_mode in enumerate(FUSION_MODES, 1):
        print(f"\n[{i}/{len(FUSION_MODES)}] Processing: {fusion_mode}")
        sys.stdout.flush()
        
        success = run_training(fusion_mode)
        results[fusion_mode] = "SUCCESS" if success else "FAILED"
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Start time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print("\nResults Summary:")
    print("-" * 80)
    
    for fusion_mode, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_symbol} {fusion_mode:20s} - {status}")
    
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nTo compare results, run:")
    print(f"  python3 compare_fusion_results.py --output_dir {OUTPUT_DIR}")
    print("=" * 80 + "\n")
    sys.stdout.flush()
    
    # Return non-zero exit code if any experiments failed
    if "FAILED" in results.values():
        sys.exit(1)


if __name__ == "__main__":
    main()
