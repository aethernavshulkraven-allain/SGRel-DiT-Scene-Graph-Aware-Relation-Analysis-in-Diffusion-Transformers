import json
from pathlib import Path
from typing import List

import torch
from diffusers import FluxPipeline

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

    def _load_pipeline(self) -> FluxPipeline:
        pipe = FluxPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
        )
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run(self):
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        examples = load_stage_a_examples(cfg.stage_a_jsonl, cfg.max_examples)
        pipe = self._load_pipeline()
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

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

        with torch.no_grad():
            with ConceptAttentionTracer(
                pipe.transformer,
                concept_states=concept_states,
                record_concepts=self.config.store_concept_states,
                downsample=self.config.downsample_saliency,
            ) as tracer:
                _ = pipe(
                    prompt=ex.prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=gen,
                    output_type="latent",
                )

        records = []
        for rec in tracer.records:
            item = {"layer": rec.layer, "saliency": rec.saliency}
            if self.config.store_concept_states:
                item["concept_states"] = rec.concept_states
            records.append(item)

        payload = {"meta": meta, "layers": records}
        torch.save(payload, out_dir / f"example_{idx:05d}.pt")
