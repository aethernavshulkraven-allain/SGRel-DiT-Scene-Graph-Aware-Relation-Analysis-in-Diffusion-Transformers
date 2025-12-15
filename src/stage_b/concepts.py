from typing import Iterable, List, Tuple

import torch
from transformers import T5EncoderModel, T5TokenizerFast


def encode_concepts(
    tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, concepts: Iterable[str], device, dtype
) -> torch.Tensor:
    """Encode concept strings to embeddings (mean-pooled over tokens)."""
    inputs = tokenizer(
        list(concepts),
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = text_encoder(input_ids)
    hidden = outputs.last_hidden_state.to(dtype=dtype)
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1)
    pooled = summed / lengths
    return pooled.unsqueeze(0)  # (1, num_concepts, dim)


def build_concept_inputs(
    tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, concepts: List[str], device, dtype
) -> torch.Tensor:
    """Convenience wrapper."""
    return encode_concepts(tokenizer, text_encoder, concepts, device, dtype)
