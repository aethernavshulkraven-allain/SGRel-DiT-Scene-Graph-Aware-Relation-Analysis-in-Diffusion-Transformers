import torch
import torch.nn as nn
from typing import Iterable


def _project_graph_tokens(transformer, graph_local, graph_global):
    """Returns projected graph tokens (B, 2, inner_dim)."""
    proj_local = transformer.graph_local_proj(graph_local)  # (B, K, inner_dim)
    proj_global = transformer.graph_global_proj(graph_global.unsqueeze(1))  # (B,1,inner_dim)
    return torch.cat([proj_global, proj_local], dim=1)


def patch_flux_for_graph(transformer, mode: str = "token", block_range: Iterable[int] = range(7, 13)):
    """
    Patch FluxTransformer2DModel to accept graph embeddings and inject them into middle blocks.
    mode: 'token' (concat graph tokens) or 'temb' (add projected global to temb).
    block_range: indices of transformer_blocks to patch (e.g., 7..12).
    """
    assert mode in {"token", "temb"}
    inner_dim = transformer.inner_dim
    ref_param = next(iter(transformer.parameters()), None)
    ref_device = ref_param.device if ref_param is not None else None
    ref_dtype = ref_param.dtype if ref_param is not None else None

    def _make_linear(in_features: int, out_features: int) -> nn.Linear:
        # Keep new graph projection layers on the same device/dtype as the base transformer weights so that
        # CPU-offload hooks don't create mixed-dtype matmuls at runtime.
        return nn.Linear(in_features, out_features, device=ref_device, dtype=ref_dtype)

    if not hasattr(transformer, "graph_local_proj"):
        transformer.graph_local_proj = _make_linear(1536, inner_dim)
    if not hasattr(transformer, "graph_global_proj"):
        transformer.graph_global_proj = _make_linear(512, inner_dim)
    if not hasattr(transformer, "graph_global_to_temb"):
        transformer.graph_global_to_temb = _make_linear(512, inner_dim)

    transformer._graph_mode = mode
    transformer._graph_local_cache = None
    transformer._graph_global_cache = None
    if mode == "token":
        # Token concatenation changes the rotary-embedding sequence length, so we need the extra tokens to be present
        # from the first block onward to keep shapes consistent.
        block_range = range(len(transformer.transformer_blocks))
    transformer._graph_block_range = set(block_range)
    transformer._graph_base_text_len = None

    # Wrap transformer forward to stash graph embeddings
    if not hasattr(transformer, "_orig_forward"):
        transformer._orig_forward = transformer.forward

        def forward_with_graph(*args, graph_local=None, graph_global=None, clear_graph=False, **kwargs):
            # Persist caches across calls unless explicitly cleared.
            if graph_local is not None:
                transformer._graph_local_cache = graph_local
            if graph_global is not None:
                transformer._graph_global_cache = graph_global
            if transformer._graph_mode == "token" and transformer._graph_local_cache is not None:
                txt_ids = kwargs.get("txt_ids", None)
                if txt_ids is not None:
                    base_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
                    extra = transformer._graph_local_cache.shape[1] + 1  # global + locals
                    pad = torch.zeros(
                        extra,
                        base_ids.shape[-1],
                        device=base_ids.device,
                        dtype=base_ids.dtype,
                    )
                    new_txt_ids = torch.cat([base_ids, pad], dim=0)
                    kwargs["txt_ids"] = new_txt_ids.unsqueeze(0) if txt_ids.ndim == 3 else new_txt_ids
            out = transformer._orig_forward(*args, **kwargs)
            if clear_graph:
                transformer._graph_local_cache = None
                transformer._graph_global_cache = None
            return out

        # Keep this as a plain function (not a bound method). This plays nicely with Accelerate's CPU-offload hooks
        # which replace `forward` with a closure that expects *args/**kwargs *without* an extra `self` positional arg.
        transformer.forward = forward_with_graph

    # Wrap target blocks
    for idx, block in enumerate(transformer.transformer_blocks):
        if idx not in transformer._graph_block_range:
            continue
        if hasattr(block, "_orig_forward_graph"):
            continue
        block._orig_forward_graph = block.forward

        def make_wrapped(b=block, i=idx):
            def _wrapped(hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
                gl = transformer._graph_local_cache
                gg = transformer._graph_global_cache
                # Only inject if provided
                if gl is not None and gg is not None:
                    if transformer._graph_mode == "token":
                        if transformer._graph_base_text_len is None:
                            transformer._graph_base_text_len = encoder_hidden_states.shape[1]
                        if encoder_hidden_states.shape[1] == transformer._graph_base_text_len:
                            graph_tokens = _project_graph_tokens(transformer, gl, gg)
                            encoder_hidden_states = torch.cat([encoder_hidden_states, graph_tokens], dim=1)
                    else:  # temb
                        temb = temb + transformer.graph_global_to_temb(gg)
                return b._orig_forward_graph(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            return _wrapped

        block.forward = make_wrapped()

    return transformer


def set_graph_condition(transformer, graph_local=None, graph_global=None):
    """Cache graph embeddings on a patched transformer."""
    transformer._graph_local_cache = graph_local
    transformer._graph_global_cache = graph_global


def clear_graph_condition(transformer):
    """Remove cached graph embeddings."""
    transformer._graph_local_cache = None
    transformer._graph_global_cache = None
