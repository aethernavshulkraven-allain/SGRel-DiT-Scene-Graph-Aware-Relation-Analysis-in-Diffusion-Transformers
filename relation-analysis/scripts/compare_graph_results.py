"""
Quick visual comparison of graph-conditioned images.
Shows if different scene graphs produce different spatial arrangements.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Load images
results_dir = Path("../inference_results_quickwin")

images = [
    ("person_riding_bike.png", "person RIDING bike"),
    ("person_next_to_bike.png", "person NEXT TO bike"),
    ("dog_wearing_hat.png", "dog WEARING hat"),
]

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Graph-Conditioned Image Generation - Different Relations", fontsize=16, fontweight='bold')

for ax, (filename, title) in zip(axes, images):
    img_path = results_dir / filename
    if img_path.exists():
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{filename}", 
                ha='center', va='center', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig(results_dir / "comparison.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison to: {results_dir / 'comparison.png'}")
plt.show()

print("\n" + "="*80)
print("WHAT TO LOOK FOR:")
print("="*80)
print("1. RIDING vs NEXT TO:")
print("   - RIDING: Person should be ON TOP of the bike (vertical relationship)")
print("   - NEXT TO: Person and bike should be SIDE BY SIDE (horizontal)")
print()
print("2. WEARING:")
print("   - Dog and hat should have direct contact/overlap")
print("   - Hat should be positioned ON or ABOVE the dog")
print()
print("3. If the images look IDENTICAL or RANDOM:")
print("   → Graph conditioning is NOT working effectively")
print("   → The model didn't learn to distinguish spatial relationships")
print()
print("4. If the images show CLEAR SPATIAL DIFFERENCES:")
print("   → Graph conditioning IS working!")
print("   → Different relations produce different layouts")
print("="*80)
