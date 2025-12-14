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
    p.add_argument("--no-average-timesteps", action="store_true", help="Do not average saliency across timesteps")
    p.add_argument("--no-average-layer-groups", action="store_true", help="Do not average saliency into layer groups")
    p.add_argument("--height", type=int, default=1024, help="Output image height")
    p.add_argument("--width", type=int, default=1024, help="Output image width")
    p.add_argument("--cpu-offload", action="store_true", help="Enable sequential CPU offload to save GPU memory")
    p.add_argument("--lora-checkpoint", type=str, default=None, help="Optional LoRA checkpoint (.pt) to load for evaluation")
    p.add_argument("--graph-mode", type=str, default=None, choices=["token", "temb"], help="Enable graph conditioning in this mode")
    p.add_argument("--block-start", type=int, default=7, help="First Flux double-block index (inclusive) for graph+LoRA")
    p.add_argument("--block-end", type=int, default=13, help="Last Flux double-block index (exclusive) for graph+LoRA")
    p.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (if not stored in checkpoint)")
    p.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha (if not stored in checkpoint)")
    p.add_argument(
        "--vocab-path",
        type=str,
        default=str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json"),
        help="SGDiff VG vocab.json path (for graph encoder)",
    )
    p.add_argument(
        "--cgip-ckpt",
        type=str,
        default=str(REPO_ROOT / "SGDiff" / "pretrained" / "sip_vg.pt"),
        help="SGDiff CGIP checkpoint path (for graph encoder)",
    )
    p.add_argument("--graph-encoder-device", type=str, default="cpu", help="Device for SGDiff graph encoder (cpu|cuda)")
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
        average_timesteps=not args.no_average_timesteps,
        average_layer_groups=not args.no_average_layer_groups,
        height=args.height,
        width=args.width,
        enable_cpu_offload=args.cpu_offload,
        lora_checkpoint=Path(args.lora_checkpoint) if args.lora_checkpoint else None,
        graph_mode=args.graph_mode,
        block_start=args.block_start,
        block_end=args.block_end,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        vocab_path=Path(args.vocab_path) if (args.graph_mode or args.lora_checkpoint) else None,
        cgip_ckpt=Path(args.cgip_ckpt) if (args.graph_mode or args.lora_checkpoint) else None,
        graph_encoder_device=args.graph_encoder_device,
    )
    runner = StageBRunner(cfg)
    runner.run()
    print(f"Wrote traces to {cfg.output_dir}")


if __name__ == "__main__":
    main()
