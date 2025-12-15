"""
Quick test script to verify the setup before running full generation.
Tests on a single GPU with a few samples.
"""

import sys
sys.path.insert(0, "/home/namanb/SBILab/CSE677/Project/ConceptAttention")

import torch
from ConceptAttention_.concept_attention import ConceptAttentionFluxPipeline
import json
from pathlib import Path

def test_pipeline():
    """Test that the ConceptAttention pipeline works."""
    print("Testing ConceptAttention pipeline...")
    
    try:
        pipeline = ConceptAttentionFluxPipeline(
            model_name="flux-schnell",
            device="cuda:0"
        )
        print("✓ Pipeline loaded successfully")
        
        # Test generation
        prompt = "a photo of shade on sidewalk"
        concepts = ["shade", "on", "sidewalk"]
        
        print(f"Testing generation with prompt: '{prompt}'")
        
        pipeline_output = pipeline.generate_image(
            prompt=prompt,
            concepts=concepts,
            width=512,
            height=512,
            return_pil_heatmaps=False,
            softmax=True,
            layer_indices=list(range(0, 7)),  # Early layers
            timesteps=list(range(4)),
            num_inference_steps=4,
            seed=42
        )
        
        print(f"✓ Generation successful!")
        print(f"  Saliency maps shape: {pipeline_output.concept_heatmaps.shape}")
        print(f"  Expected: (3, 32, 32) for 3 concepts at 512x512 image")
        
        if pipeline_output.concept_heatmaps.shape == (3, 32, 32):
            print("✓ Shape matches expected!")
            return True
        else:
            print("✗ Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test that we can load the JSONL data."""
    print("\nTesting data loading...")
    
    jsonl_path = "/home/namanb/SBILab/CSE677/Project/SGRel-DiT-Scene-Graph-Aware-Relation-Analysis-in-Diffusion-Transformers/relation-analysis/outputs/stage_a/vg_stage_a_full.jsonl"
    
    try:
        with open(jsonl_path, "r") as f:
            first_line = f.readline()
            data = json.loads(first_line)
        
        print(f"✓ Data loaded successfully")
        print(f"  Sample data: {data}")
        
        required_keys = ["triple", "prompt", "concepts"]
        if all(key in data for key in required_keys):
            print(f"✓ All required keys present")
            return True
        else:
            print(f"✗ Missing keys!")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_save_load():
    """Test saving and loading a sample."""
    print("\nTesting save/load...")
    
    test_dir = Path("/tmp/saliency_test")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create dummy data
        test_data = {
            "saliency_maps": torch.randn(3, 32, 32),
            "cross_attention_maps": torch.randn(3, 32, 32),
            "prompt": "test prompt",
            "concepts": ["a", "b", "c"],
            "class_id": 0,
            "predicate": "on",
            "sample_id": 0,
            "layer_config": "early_layers",
            "layer_indices": [0, 1, 2],
            "metadata": {}
        }
        
        # Save
        save_path = test_dir / "test_sample.pt"
        torch.save(test_data, save_path)
        print(f"✓ Saved test sample to {save_path}")
        
        # Load
        loaded_data = torch.load(save_path)
        print(f"✓ Loaded test sample")
        
        # Verify
        if loaded_data["saliency_maps"].shape == (3, 32, 32):
            print(f"✓ Data integrity verified")
            return True
        else:
            print(f"✗ Data mismatch!")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*80)
    print("Running pre-flight checks for saliency dataset generation")
    print("="*80)
    
    results = {
        "Data Loading": test_data_loading(),
        "Pipeline": test_pipeline(),
        "Save/Load": test_save_load()
    }
    
    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to generate datasets.")
        print("\nRun:")
        print("  python generate_saliency_dataset.py --layer-config all")
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
