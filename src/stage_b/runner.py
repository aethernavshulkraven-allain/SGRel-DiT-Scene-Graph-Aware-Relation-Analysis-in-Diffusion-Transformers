import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import FluxPipeline

from relation_analysis.flux.graph_conditioned_flux import clear_graph_condition, patch_flux_for_graph, set_graph_condition
from relation_analysis.flux.lora import LinearWithLoRA
from relation_analysis.graph.sgdiff_encoder import SGDiffGraphEncoder
from relation_analysis.schema import RelationTriple, StageAExample
from relation_analysis.stage_b.config import StageBConfig
from relation_analysis.stage_b.concepts import build_concept_inputs
from relation_analysis.stage_b.tracer import ConceptAttentionTracer


def _dtype_from_str(name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_stage_a_examples(jsonl_path: Path, max_examples: int) -> List[StageAExample]:
    examples: List[StageAExample] = []
    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            if max_examples and idx >= max_examples:
                break
            obj = json.loads(line)
            triple_obj = obj["triple"]
            triple = RelationTriple(
                subject=triple_obj["subject"],
                predicate=triple_obj["predicate"],
                object=triple_obj["object"],
                relation_type=triple_obj["relation_type"],
                source_image_id=triple_obj.get("source_image_id"),
                source_relationship_id=triple_obj.get("source_relationship_id"),
            )
            ex = StageAExample(triple=triple, prompt=obj["prompt"], concepts=obj["concepts"], template_id=obj["template_id"])
            examples.append(ex)
    return examples


class StageBRunner:
    """Runs Flux-small with ConceptAttention tracer to emit saliency per block."""

    def __init__(self, config: StageBConfig):
        self.config = config
        self.dtype = _dtype_from_str(config.dtype)
        self.device = torch.device(config.device)
        self._graph_encoder: Optional[SGDiffGraphEncoder] = None

    @staticmethod
    def _inject_lora_into_blocks(transformer, block_indices, rank: int, alpha: float):
        for idx in block_indices:
            if idx < 0 or idx >= len(transformer.transformer_blocks):
                continue
            block = transformer.transformer_blocks[idx]
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            for name in ("to_q", "to_k", "to_v"):
                proj = getattr(attn, name, None)
                if isinstance(proj, torch.nn.Linear) and not isinstance(proj, LinearWithLoRA):
                    setattr(attn, name, LinearWithLoRA(proj, rank=rank, alpha=alpha))
            to_out = getattr(attn, "to_out", None)
            if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0 and isinstance(to_out[0], torch.nn.Linear):
                if not isinstance(to_out[0], LinearWithLoRA):
                    to_out[0] = LinearWithLoRA(to_out[0], rank=rank, alpha=alpha)

    @staticmethod
    def _extract_checkpoint_state_dict(obj) -> dict:
        if isinstance(obj, dict):
            for key in ("state_dict", "transformer_state_dict", "model_state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    obj = obj[key]
                    break
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
        # Filter to tensor values only (some checkpoints may include nested dicts like 'classifier').
        return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _extract_checkpoint_config(obj) -> dict:
        if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
            return obj["config"]
        return {}

    def _load_pipeline(self) -> FluxPipeline:
        pipe = FluxPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
        )

        # Optional: load graph-conditioned LoRA checkpoint for evaluation.
        cfg = self.config
        ckpt_obj = None
        ckpt_cfg: dict = {}
        if cfg.lora_checkpoint is not None:
            ckpt_obj = torch.load(cfg.lora_checkpoint, map_location="cpu", weights_only=False)
            ckpt_cfg = self._extract_checkpoint_config(ckpt_obj)

        graph_mode = cfg.graph_mode or ckpt_cfg.get("graph_mode")
        block_start = int(ckpt_cfg.get("block_start", cfg.block_start))
        block_end = int(ckpt_cfg.get("block_end", cfg.block_end))
        lora_rank = cfg.lora_rank if cfg.lora_rank is not None else ckpt_cfg.get("lora_rank")
        lora_alpha = cfg.lora_alpha if cfg.lora_alpha is not None else ckpt_cfg.get("lora_alpha")

        if graph_mode is not None:
            patch_flux_for_graph(pipe.transformer, mode=str(graph_mode), block_range=range(block_start, block_end))

        if ckpt_obj is not None:
            if lora_rank is None:
                raise ValueError("LoRA checkpoint provided but lora_rank is unknown (pass --lora-rank or save config in checkpoint).")
            if lora_alpha is None:
                lora_alpha = float(lora_rank)
            self._inject_lora_into_blocks(pipe.transformer, range(block_start, block_end), rank=int(lora_rank), alpha=float(lora_alpha))
            state_dict = self._extract_checkpoint_state_dict(ckpt_obj)
            missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[stage_b] Loaded LoRA checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

        if self.config.enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run(self):
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        examples = load_stage_a_examples(cfg.stage_a_jsonl, cfg.max_examples)
        pipe = self._load_pipeline()
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        if (cfg.graph_mode is not None or cfg.lora_checkpoint is not None) and cfg.vocab_path is not None and cfg.cgip_ckpt is not None:
            self._graph_encoder = SGDiffGraphEncoder(
                vocab_path=str(cfg.vocab_path),
                ckpt_path=str(cfg.cgip_ckpt),
                device=cfg.graph_encoder_device,
            )
        elif cfg.graph_mode is not None or cfg.lora_checkpoint is not None:
            raise ValueError("Graph evaluation requested but vocab_path/cgip_ckpt not provided.")

        for idx, ex in enumerate(examples):
            self._run_one(pipe, ex, idx, gen)

    def _run_one(self, pipe: FluxPipeline, ex: StageAExample, idx: int, gen: torch.Generator):
        # Encode concept strings -> inner_dim
        concept_embeds = build_concept_inputs(
            tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, concepts=ex.concepts, device=self.device, dtype=self.dtype
        )
        concept_states = pipe.transformer.context_embedder(concept_embeds)

        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "index": idx,
            "prompt": ex.prompt,
            "concepts": ex.concepts,
            "triple": ex.triple,
        }

        if self._graph_encoder is not None:
            triplet: Tuple[str, str, str] = (ex.triple.subject, ex.triple.predicate, ex.triple.object)
            with torch.no_grad():
                graph_local, graph_global = self._graph_encoder.encode_batch([triplet])
            graph_local = graph_local.to(device=self.device, dtype=self.dtype)
            graph_global = graph_global.to(device=self.device, dtype=self.dtype)
            set_graph_condition(pipe.transformer, graph_local=graph_local, graph_global=graph_global)

        with torch.no_grad():
            with ConceptAttentionTracer(
                pipe.transformer,
                concept_states=concept_states,
                record_concepts=self.config.store_concept_states,
                downsample=self.config.downsample_saliency,
                pipeline=pipe,
            ) as tracer:
                _ = pipe(
                    prompt=ex.prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=gen,
                    output_type="latent",
                    height=self.config.height,
                    width=self.config.width,
                )

        if self._graph_encoder is not None:
            clear_graph_condition(pipe.transformer)

        records = tracer.records
        payload = {"meta": meta}
        payload["layers"] = [
            {
                "layer": rec.layer,
                "timestep": rec.timestep,
                "call_index": rec.call_index,
                "saliency": rec.saliency,
                **({"concept_states": rec.concept_states} if self.config.store_concept_states else {}),
            }
            for rec in records
        ]

        # Aggregate over timesteps and layer groups if configured.
        if self.config.average_timesteps or self.config.average_layer_groups:
            per_layer: Dict[int, List[torch.Tensor]] = {}
            for rec in records:
                per_layer.setdefault(rec.layer, []).append(rec.saliency)
            layer_means: Dict[int, torch.Tensor] = {}
            for layer, items in per_layer.items():
                stack = torch.stack(items)
                if self.config.average_timesteps:
                    stack = stack.mean(dim=0)
                layer_means[layer] = stack
            if self.config.average_layer_groups:
                groups = self.config.resolved_layer_groups(num_layers=len(per_layer))
                group_maps = {}
                for name, layers in groups.items():
                    selected = [layer_means[l] for l in layers if l in layer_means]
                    if selected:
                        group_maps[name] = torch.stack(selected).mean(dim=0)
                payload["group_saliency"] = group_maps
            else:
                payload["layer_saliency"] = layer_means

        torch.save(payload, out_dir / f"example_{idx:05d}.pt")
