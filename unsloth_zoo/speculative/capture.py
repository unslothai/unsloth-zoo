# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import contextlib
import torch

__all__ = [
    "resolve_text_parts",
    "capture_hidden_states",
    "run_frozen_target",
]

_NORM_NAMES = ("norm", "final_layernorm", "norm_f", "ln_f")


def _unwrap(model):
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            base = get_base_model()
            if base is not None and base is not model:
                model = base
                continue
        inner = getattr(getattr(model, "base_model", None), "model", None)
        if inner is not None and inner is not model and hasattr(model, "peft_config"):
            model = inner
            continue
    return model


def _has_layers(module):
    return isinstance(getattr(module, "layers", None), torch.nn.ModuleList)


def resolve_text_parts(model):
    """Locate the decoder stack, its final norm, embeddings, lm_head and text config."""
    top = _unwrap(model)
    backbone = None
    inner = getattr(top, "model", None)
    for candidate in (
        getattr(inner, "language_model", None),
        inner,
        getattr(top, "language_model", None),
        top,
    ):
        if candidate is not None and _has_layers(candidate):
            backbone = candidate
            break
    if backbone is None:
        raise ValueError(
            f"Unsloth: Could not find the decoder layers of `{type(model).__name__}` for speculative training."
        )
    final_norm = None
    for name in _NORM_NAMES:
        final_norm = getattr(backbone, name, None)
        if isinstance(final_norm, torch.nn.Module):
            break
        final_norm = None
    if final_norm is None:
        raise ValueError(
            f"Unsloth: Could not find the final norm of `{type(backbone).__name__}` inside `{type(model).__name__}`."
        )
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if embed_tokens is None and hasattr(top, "get_input_embeddings"):
        embed_tokens = top.get_input_embeddings()
    lm_head = top.get_output_embeddings() if hasattr(top, "get_output_embeddings") else None
    if lm_head is None:
        lm_head = getattr(top, "lm_head", None)
    if embed_tokens is None or lm_head is None:
        raise ValueError(
            f"Unsloth: Could not find the embeddings or lm_head of `{type(model).__name__}`."
        )
    config = getattr(backbone, "config", None)
    if config is None:
        config = top.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return {
        "backbone"     : backbone,
        "final_norm"   : final_norm,
        "embed_tokens" : embed_tokens,
        "lm_head"      : lm_head,
        "config"       : config,
    }


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _store(tensor):
    if tensor.is_inference():
        # Inference tensors cannot be saved for backward; a clone made outside inference mode can.
        with torch.inference_mode(False):
            return tensor.detach().clone()
    return tensor.detach()


@contextlib.contextmanager
def capture_hidden_states(model, layer_ids = (), final = True, embed = False):
    """Record the post-norm final hidden state (`"final"`), decoder layer outputs (`i`) and, with
    `embed`, the embedding output (`"embed"`)."""
    parts = resolve_text_parts(model)
    backbone, layers = parts["backbone"], parts["backbone"].layers
    n_layers = len(layers)
    layer_ids = [int(i) for i in layer_ids]
    for i in layer_ids:
        if not -n_layers <= i < n_layers:
            raise ValueError(f"Unsloth: layer id {i} is out of range for {n_layers} decoder layers.")
    captured = {}
    norm_fired = [False]
    handles = []

    def layer_hook(key):
        def hook(module, args, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                captured[key] = _store(tensor)
        return hook

    def norm_hook(module, args, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            captured["final"] = _store(tensor)
            norm_fired[0] = True

    def backbone_pre_hook(module, args, kwargs = None):
        norm_fired[0] = False

    def backbone_hook(module, args, output):
        # Fast forwards (e.g. Unsloth's LlamaModel) call the norm functionally, bypassing its
        # hook; the backbone's returned last_hidden_state is post-norm in both paths.
        if not norm_fired[0]:
            tensor = _first_tensor(output)
            if tensor is not None:
                captured["final"] = _store(tensor)

    try:
        if embed:
            handles.append(parts["embed_tokens"].register_forward_hook(layer_hook("embed")))
        for i in layer_ids:
            handles.append(layers[i].register_forward_hook(layer_hook(i)))
        if final:
            handles.append(parts["final_norm"].register_forward_hook(norm_hook))
            handles.append(backbone.register_forward_pre_hook(backbone_pre_hook))
            handles.append(backbone.register_forward_hook(backbone_hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


_MULTIMODAL_KWARGS = (
    "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
    "input_features", "mm_token_type_ids", "token_type_ids", "image_sizes",
)


def run_frozen_target(model, input_ids, attention_mask = None, layer_ids = (), embed = False, **kwargs):
    """No-grad target pass returning the capture dict, skipping the lm_head when possible."""
    parts = resolve_text_parts(model)
    multimodal = any(kwargs.get(name) is not None for name in _MULTIMODAL_KWARGS)
    with torch.no_grad(), capture_hidden_states(model, layer_ids = layer_ids, embed = embed) as captured:
        if not multimodal:
            try:
                parts["backbone"](
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    use_cache = False,
                    **kwargs,
                )
            except TypeError:
                multimodal = True
        if multimodal or "final" not in captured:
            model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                use_cache = False,
                **kwargs,
            )
    return dict(captured)
