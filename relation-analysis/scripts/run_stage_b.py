#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Make local diffusers importable if not installed
local_diffusers = REPO_ROOT / "diffusers" / "src"
if local_diffusers.exists() and str(local_diffusers) not in sys.path:
    sys.path.insert(0, str(local_diffusers))

from relation_analysis.stage_b.config import StageBConfig
from relation_analysis.stage_b.runner import StageBRunner


def parse_args():
    p = argparse.ArgumentParser(description="Stage B: Flux-small + ConceptAttention saliency tracer.")
    p.add_argument("--input", type=str, default="outputs/stage_a/vg_stage_a.jsonl", help="Stage A JSONL path")
    p.add_argument("--output-dir", type=str, default="outputs/stage_b/runs/test_run", help="Where to write .pt traces")
    p.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-schnell", help="Flux model id")
    p.add_argument("--device", type=str, default="cuda", help="Device for pipeline")
    p.add_argument("--dtype", type=str, default="bfloat16", help="torch dtype (bfloat16|float16|float32)")
    p.add_argument("--max-examples", type=int, default=4, help="Number of prompts to process")
    p.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    p.add_argument("--guidance", type=float, default=3.0, help="Guidance scale")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--store-concept-states", action="store_true", help="Persist concept states per layer (larger files)")
    p.add_argument("--downsample-saliency", type=int, default=None, help="Keep every Nth token in saliency to shrink output size")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = StageBConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        max_examples=args.max_examples,
        stage_a_jsonl=Path(args.input),
        output_dir=Path(args.output_dir),
        store_concept_states=args.store_concept_states,
        downsample_saliency=args.downsample_saliency,
    )
    runner = StageBRunner(cfg)
    runner.run()
    print(f"Wrote traces to {cfg.output_dir}")


if __name__ == "__main__":
    main()
