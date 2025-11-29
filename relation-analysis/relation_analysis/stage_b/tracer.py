import contextlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock


def _split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    b, t, dim = x.shape
    return x.view(b, t, heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def concept_attention_step(block: FluxTransformerBlock, concepts: torch.Tensor, image_states: torch.Tensor) -> torch.Tensor:
    """
    Lightweight ConceptAttention:
    - queries = concept tokens
    - keys/values = [image tokens, concept tokens]
    Uses the text projections (block.attn) for all streams.
    """
    heads = block.attn.heads
    head_dim = block.attn.head_dim

    q = _split_heads(block.attn.to_q(concepts), heads, head_dim)
    k_img = _split_heads(block.attn.to_k(image_states), heads, head_dim)
    v_img = _split_heads(block.attn.to_v(image_states), heads, head_dim)
    k_con = _split_heads(block.attn.to_k(concepts), heads, head_dim)
    v_con = _split_heads(block.attn.to_v(concepts), heads, head_dim)

    k = torch.cat([k_img, k_con], dim=2)
    v = torch.cat([v_img, v_con], dim=2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out = _merge_heads(out)
    return out


def compute_saliency(image_states: torch.Tensor, concept_states: torch.Tensor) -> torch.Tensor:
    """Dot-product saliency map per concept: (batch, concepts, img_tokens)."""
    # image_states: (b, img_tokens, d), concept_states: (b, concepts, d)
    return torch.einsum("bid,bjd->bji", image_states, concept_states)


@dataclass
class LayerRecord:
    layer: int
    saliency: torch.Tensor  # (num_concepts, img_tokens)
    concept_states: Optional[torch.Tensor] = None


class ConceptAttentionTracer(contextlib.AbstractContextManager):
    """
    Monkey-patches Flux transformer blocks to:
    - keep a running concept stream,
    - record saliency per block.
    """

    def __init__(self, transformer, concept_states: torch.Tensor, record_concepts: bool = False, downsample: Optional[int] = None):
        self.transformer = transformer
        self.concept_states = concept_states
        self.record_concepts = record_concepts
        self.downsample = downsample
        self.records: List[LayerRecord] = []
        self._orig_forwards: List[Any] = []

    def __enter__(self):
        for idx, block in enumerate(self.transformer.transformer_blocks):
            orig = block.forward
            self._orig_forwards.append(orig)

            def wrapper(b=block, i=idx, orig_forward=orig):
                def _wrapped(*args, **kwargs):
                    enc_out, img_out = orig_forward(*args, **kwargs)
                    concept_out = concept_attention_step(b, self.concept_states, img_out)
                    sal = compute_saliency(img_out, concept_out)
                    if self.downsample and self.downsample > 1:
                        sal = sal[..., :: self.downsample]
                    sal = sal.detach().cpu()
                    rec = LayerRecord(layer=i, saliency=sal.squeeze(0))
                    if self.record_concepts:
                        rec.concept_states = concept_out.detach().cpu()
                    self.records.append(rec)
                    self.concept_states = concept_out
                    return enc_out, img_out

                return _wrapped

            block.forward = wrapper()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original forwards
        for block, orig in zip(self.transformer.transformer_blocks, self._orig_forwards):
            block.forward = orig
        return False
