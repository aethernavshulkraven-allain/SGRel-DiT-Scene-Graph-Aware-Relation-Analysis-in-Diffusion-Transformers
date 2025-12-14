"""
Generate images WITHOUT graph conditioning to see baseline behavior.
If these look the same as graph-conditioned ones, the graph isn't helping.
"""

import torch
from pathlib import Path
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import FluxPipeline

def main():
    device = torch.device("cuda:3")
    output_dir = Path("../inference_results_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading baseline FLUX model (no graph conditioning)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer.to(device)
    pipe.vae.to(device)
    pipe.text_encoder.to(device) if pipe.text_encoder else None
    pipe.text_encoder_2.to(device) if pipe.text_encoder_2 else None
    
    test_cases = [
        ("person_riding_bike.png", "a photo of a person riding a bike"),
        ("person_next_to_bike.png", "a photo of a person next to a bike"),
        ("dog_wearing_hat.png", "a photo of a dog wearing a hat"),
    ]
    
    print("\nGenerating baseline images (NO graph conditioning)...")
    print("="*80)
    
    for filename, prompt in test_cases:
        print(f"\nPrompt: {prompt}")
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=256,
                width=256,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]
            
        save_path = output_dir / filename
        image.save(save_path)
        print(f"✓ Saved: {save_path}")
    
    print("\n" + "="*80)
    print(f"✓ Baseline images saved to: {output_dir}")
    print("\nNow compare:")
    print(f"  - Baseline: {output_dir}")
    print(f"  - Graph-conditioned: ../inference_results_quickwin")
    print("\nIf they look very similar, the graph conditioning isn't working.")

if __name__ == "__main__":
    main()
