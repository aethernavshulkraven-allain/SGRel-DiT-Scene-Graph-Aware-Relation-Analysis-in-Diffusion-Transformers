from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StageBConfig:
    """Configuration for Stage B runs (ConceptAttention + saliency)."""

    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int = 4
    guidance_scale: float = 3.0
    seed: int = 0
    max_examples: int = 8
    stage_a_jsonl: Path = Path("outputs/stage_a/vg_stage_a.jsonl")
    output_dir: Path = Path("outputs/stage_b/runs/default")
    store_saliency: bool = True
    store_concept_states: bool = False  # set True if you want to keep per-layer concept outputs
    downsample_saliency: Optional[int] = None  # e.g., 8 to shrink maps spatially
    average_timesteps: bool = True  # average saliency across diffusion steps
    layer_groups: dict = None  # e.g., {"early": [0,1,2,3,4,5], "mid": [...], "late": [...]}
    average_layer_groups: bool = True  # average per-layer saliency into group maps
    height: int = 1024
    width: int = 1024
    enable_cpu_offload: bool = False

    def resolved_layer_groups(self, num_layers: int = 19) -> dict:
        if self.layer_groups is not None:
            return self.layer_groups
        return {
            "early": list(range(0, max(1, num_layers // 3))),
            "mid": list(range(num_layers // 3, 2 * num_layers // 3)),
            "late": list(range(2 * num_layers // 3, num_layers)),
        }
