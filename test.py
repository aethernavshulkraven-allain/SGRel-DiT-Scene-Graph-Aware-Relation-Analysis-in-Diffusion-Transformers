"""
Load a Stage B .pt file and save saliency heatmaps + raw matrices per concept per group/layer.

Usage (from repo root):
  PYTHONPATH=relation-analysis python test.py \
    --pt relation-analysis/outputs/stage_b/runs/sample10/example_00000.pt \
    --out-dir relation-analysis/outputs/stage_b/runs/sample10/vis \
    --mode group    # or layer
    --save-npy      # also dump .npy arrays
"""

import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# allowlist dataclasses stored in the checkpoint
from torch.serialization import add_safe_globals

from relation_analysis.schema import RelationTriple, StageAExample


add_safe_globals([RelationTriple, StageAExample])


def normalize_and_reshape(vec: torch.Tensor):
    vec = vec.float()
    vmin, vmax = vec.min(), vec.max()
    if (vmax - vmin) > 0:
        vec = (vec - vmin) / (vmax - vmin)
    n = vec.numel()
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"Token count {n} is not a perfect square; cannot reshape to grid.")
    return vec.view(side, side)


def save_heatmap(grid: torch.Tensor, title: str, out_path: Path):
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.cpu().numpy(), cmap="plasma")
    plt.axis("off")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to Stage B .pt output")
    parser.add_argument("--out-dir", required=True, help="Directory to write heatmaps")
    parser.add_argument("--mode", choices=["group", "layer"], default="group", help="Plot grouped saliency or per-layer saliency")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw saliency vectors as .npy")
    args = parser.parse_args()

    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    concepts = payload["meta"]["concepts"]
    out_dir = Path(args.out_dir)

    if args.mode == "group":
        group_sal = payload.get("group_saliency")
        if not group_sal:
            raise SystemExit("No group_saliency found in payload; rerun Stage B without --no-average-layer-groups.")
        for gname, tensor in group_sal.items():
            for ci, concept in enumerate(concepts):
                grid = normalize_and_reshape(tensor[ci])
                out_path = out_dir / f"{gname}_{concept}.png"
                save_heatmap(grid, f"{gname} / {concept}", out_path)
                if args.save_npy:
                    vec = tensor[ci].cpu().float().numpy()
                    np.save(out_dir / f"{gname}_{concept}.npy", vec)
                    np.save(out_dir / f"{gname}_{concept}_grid.npy", grid.cpu().numpy())
    else:
        layer_sal = payload.get("layer_saliency")
        if not layer_sal:
            raise SystemExit("No layer_saliency found in payload; rerun Stage B with --no-average-layer-groups.")
        for layer_idx, tensor in layer_sal.items():
            for ci, concept in enumerate(concepts):
                grid = normalize_and_reshape(tensor[ci])
                out_path = out_dir / f"layer{layer_idx:02d}_{concept}.png"
                save_heatmap(grid, f"layer {layer_idx} / {concept}", out_path)
                if args.save_npy:
                    vec = tensor[ci].cpu().float().numpy()
                    np.save(out_dir / f"layer{layer_idx:02d}_{concept}.npy", vec)
                    np.save(out_dir / f"layer{layer_idx:02d}_{concept}_grid.npy", grid.cpu().numpy())

    print(f"Saved heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
