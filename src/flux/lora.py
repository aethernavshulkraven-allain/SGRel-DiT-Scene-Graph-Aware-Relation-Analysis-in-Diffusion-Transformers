import math
import torch
import torch.nn as nn
from typing import List


class LinearWithLoRA(nn.Linear):
    """Wraps a Linear layer with a rank-r LoRA adapter."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        # copy base weights/bias
        self.weight = base.weight
        self.bias = base.bias
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Keep LoRA weights on the same dtype/device as the wrapped linear to avoid mixed-dtype matmuls
        # (e.g., bf16 activations with fp32 LoRA weights).
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # freeze base params
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def inject_lora(module: nn.Module, target_names: List[str], rank: int = 8, alpha: float = 8.0):
    """Recursively wrap specified Linear submodules with LoRA."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            wrapped = LinearWithLoRA(child, rank=rank, alpha=alpha)
            setattr(module, name, wrapped)
        else:
            inject_lora(child, target_names, rank=rank, alpha=alpha)
