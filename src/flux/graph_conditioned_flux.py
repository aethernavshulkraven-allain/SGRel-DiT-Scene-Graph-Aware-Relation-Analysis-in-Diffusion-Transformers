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
    # Blocks where *non-zero* graph conditioning should be applied.
    transformer._graph_block_range = set(int(x) for x in block_range)
    transformer._graph_base_text_len = None
    transformer._graph_token_extra = None
    transformer._graph_tokens_filled = False

    # Wrap transformer forward to stash graph embeddings
    if not hasattr(transformer, "_orig_forward"):
        transformer._orig_forward = transformer.forward

        def forward_with_graph(*args, graph_local=None, graph_global=None, clear_graph=False, **kwargs):
            args = list(args)
            transformer._graph_tokens_filled = False
            # Persist caches across calls unless explicitly cleared.
            if graph_local is not None:
                transformer._graph_local_cache = graph_local
            if graph_global is not None:
                transformer._graph_global_cache = graph_global
            if transformer._graph_mode == "token" and transformer._graph_local_cache is not None:
                extra = transformer._graph_local_cache.shape[1] + 1  # global + locals
                transformer._graph_token_extra = extra

                # Ensure txt_ids length matches the padded encoder_hidden_states length.
                txt_ids = kwargs.get("txt_ids", None)
                if txt_ids is None and len(args) >= 6:
                    txt_ids = args[5]
                if txt_ids is not None:
                    base_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
                    pad = torch.zeros(
                        extra,
                        base_ids.shape[-1],
                        device=base_ids.device,
                        dtype=base_ids.dtype,
                    )
                    new_txt_ids = torch.cat([base_ids, pad], dim=0)
                    if txt_ids.ndim == 3:
                        new_txt_ids = new_txt_ids.unsqueeze(0)
                    if "txt_ids" in kwargs:
                        kwargs["txt_ids"] = new_txt_ids
                    else:
                        args[5] = new_txt_ids

                # Pad encoder_hidden_states with placeholder tokens so the sequence length is constant
                # across all blocks. We only fill the placeholder region with *actual* graph tokens
                # inside `block_range`; outside it, the placeholder tokens are forced to zero.
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
                if encoder_hidden_states is None and len(args) >= 2:
                    encoder_hidden_states = args[1]
                if encoder_hidden_states is not None:
                    stored = transformer._graph_base_text_len
                    if stored is not None and encoder_hidden_states.shape[1] == int(stored) + extra:
                        base_len = int(stored)
                    else:
                        base_len = int(encoder_hidden_states.shape[1])
                    transformer._graph_base_text_len = int(base_len)
                    if encoder_hidden_states.shape[1] == base_len:
                        pad_tokens = torch.zeros(
                            encoder_hidden_states.shape[0],
                            extra,
                            encoder_hidden_states.shape[2],
                            device=encoder_hidden_states.device,
                            dtype=encoder_hidden_states.dtype,
                        )
                        padded = torch.cat([encoder_hidden_states, pad_tokens], dim=1)
                        if "encoder_hidden_states" in kwargs:
                            kwargs["encoder_hidden_states"] = padded
                        else:
                            args[1] = padded
            out = transformer._orig_forward(*args, **kwargs)
            if clear_graph:
                transformer._graph_local_cache = None
                transformer._graph_global_cache = None
            return out

        # Keep this as a plain function (not a bound method). This plays nicely with Accelerate's CPU-offload hooks
        # which replace `forward` with a closure that expects *args/**kwargs *without* an extra `self` positional arg.
        transformer.forward = forward_with_graph

    # Wrap blocks. For token-mode we need to keep placeholder tokens zero outside `block_range`,
    # so we wrap all blocks. For temb-mode, we wrap only the requested injection range.
    patch_indices = (
        range(len(transformer.transformer_blocks)) if mode == "token" else transformer._graph_block_range
    )
    for idx, block in enumerate(transformer.transformer_blocks):
        if idx not in patch_indices:
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
                        extra = transformer._graph_token_extra or (gl.shape[1] + 1)
                        base_len = transformer._graph_base_text_len
                        if base_len is None:
                            base_len = int(encoder_hidden_states.shape[1])
                            transformer._graph_base_text_len = int(base_len)
                        base_len = int(base_len)

                        if encoder_hidden_states.shape[1] == base_len:
                            pad_tokens = torch.zeros(
                                encoder_hidden_states.shape[0],
                                extra,
                                encoder_hidden_states.shape[2],
                                device=encoder_hidden_states.device,
                                dtype=encoder_hidden_states.dtype,
                            )
                            encoder_hidden_states = torch.cat([encoder_hidden_states, pad_tokens], dim=1)

                        if i in transformer._graph_block_range:
                            # Only "turn on" graph tokens once per forward pass; after that we let them
                            # evolve through subsequent blocks in the injection range.
                            if not transformer._graph_tokens_filled:
                                graph_tokens = _project_graph_tokens(transformer, gl, gg)
                                encoder_hidden_states = torch.cat(
                                    [encoder_hidden_states[:, :base_len, :], graph_tokens], dim=1
                                )
                                transformer._graph_tokens_filled = True
                        else:
                            zeros = torch.zeros_like(encoder_hidden_states[:, base_len:, :])
                            encoder_hidden_states = torch.cat([encoder_hidden_states[:, :base_len, :], zeros], dim=1)
                    else:  # temb
                        if i in transformer._graph_block_range:
                            temb = temb + transformer.graph_global_to_temb(gg)

                enc_out, img_out = b._orig_forward_graph(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if (
                    transformer._graph_mode == "token"
                    and gl is not None
                    and gg is not None
                    and i not in transformer._graph_block_range
                    and transformer._graph_base_text_len is not None
                ):
                    base_len = int(transformer._graph_base_text_len)
                    if enc_out.shape[1] > base_len:
                        enc_out = torch.cat([enc_out[:, :base_len, :], torch.zeros_like(enc_out[:, base_len:, :])], dim=1)
                return enc_out, img_out

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
