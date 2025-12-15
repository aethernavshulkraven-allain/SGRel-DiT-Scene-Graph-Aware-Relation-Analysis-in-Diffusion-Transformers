"""
Test the oversampling logic for classes with fewer samples.
"""

def test_oversampling():
    """Test that oversampling distributes samples correctly."""
    
    # Simulate the sampling logic
    SAMPLES_PER_CLASS = 2000
    NUM_GPUS = 4
    SAMPLES_PER_GPU = SAMPLES_PER_CLASS // NUM_GPUS  # 500
    
    # Test case 1: Class with plenty of samples (e.g., "on" with 668k samples)
    print("Test 1: Class with 668,000 samples")
    print("="*60)
    available_large = list(range(668000))
    
    for gpu_id in range(NUM_GPUS):
        start_idx = gpu_id * SAMPLES_PER_GPU
        end_idx = start_idx + SAMPLES_PER_GPU
        gpu_samples = available_large[start_idx:end_idx]
        print(f"GPU {gpu_id}: {len(gpu_samples)} samples (indices {start_idx}-{end_idx-1})")
    
    # Test case 2: Class with few samples (e.g., "right of" with 401 samples)
    print("\nTest 2: Class with 401 samples (needs oversampling)")
    print("="*60)
    available_small = list(range(401))
    
    import random
    for gpu_id in range(NUM_GPUS):
        start_idx = gpu_id * SAMPLES_PER_GPU
        end_idx = start_idx + SAMPLES_PER_GPU
        
        # Oversample
        repetitions_needed = (SAMPLES_PER_CLASS + len(available_small) - 1) // len(available_small)
        oversampled = available_small * repetitions_needed
        
        # Shuffle
        random.seed(42 + gpu_id)
        random.shuffle(oversampled)
        
        # Take slice
        gpu_samples = oversampled[start_idx:end_idx]
        
        print(f"GPU {gpu_id}: {len(gpu_samples)} samples")
        print(f"  Repetitions needed: {repetitions_needed}")
        print(f"  Oversampled pool size: {len(oversampled)}")
        print(f"  Unique samples in GPU slice: {len(set(gpu_samples))}")
        print(f"  Sample indices (first 10): {gpu_samples[:10]}")
    
    # Test case 3: Verify all GPUs get exactly 500 samples
    print("\nTest 3: Verify sample counts")
    print("="*60)
    
    test_cases = [
        ("Large class", 668000),
        ("Medium class", 5000),
        ("Small class", 401),
        ("Tiny class", 150),
    ]
    
    for name, num_available in test_cases:
        print(f"\n{name} ({num_available} samples):")
        total_assigned = 0
        
        for gpu_id in range(NUM_GPUS):
            start_idx = gpu_id * SAMPLES_PER_GPU
            end_idx = start_idx + SAMPLES_PER_GPU
            
            if num_available >= SAMPLES_PER_CLASS:
                # Direct slicing
                count = end_idx - start_idx
            else:
                # Oversampling
                available = list(range(num_available))
                repetitions_needed = (SAMPLES_PER_CLASS + num_available - 1) // num_available
                oversampled = available * repetitions_needed
                random.seed(42 + gpu_id)
                random.shuffle(oversampled)
                gpu_samples = oversampled[start_idx:end_idx]
                count = len(gpu_samples)
            
            total_assigned += count
            print(f"  GPU {gpu_id}: {count} samples")
        
        print(f"  Total: {total_assigned} / {SAMPLES_PER_CLASS} (target)")
        assert total_assigned == SAMPLES_PER_CLASS, f"Sample count mismatch!"
        print(f"  ✓ Correct!")


if __name__ == "__main__":
    test_oversampling()
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
