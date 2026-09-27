# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Repairs for remote (trust_remote_code) modeling code that is loaded after import.

Remote classes only exist once `from_pretrained` imports them, so these repairs hook
`transformers.dynamic_module_utils.get_class_in_module`, which every remote class load
passes through, and fix the loaded modules in place:

1. Legacy hybrid caches (e.g. `NemotronHHybridDynamicCache` in Nemotron-H / Nemotron-3-Nano-Omni)
   are plain classes, not `transformers.Cache` subclasses. transformers >= 5.x mask building
   (`masking_utils._preprocess_mask_arguments`) calls `past_key_values.get_query_offset(layer_idx)`,
   so every cached generation fails with AttributeError. `Cache.get_query_offset` is defined as
   `get_seq_length(layer_idx)`; the same method is added to such classes.

2. InternVL-style multimodal forwards (Nemotron-3-Nano-Omni, Nemotron Nano VL, InternVL chat)
   merge image features with `inputs_embeds[selected] = inputs_embeds[selected] * 0.0 + vit_embeds`.
   Under training the embedding output is a leaf that requires grad (input-require-grads hooks
   for gradient checkpointing), so that in-place write raises "a view of a leaf Variable that
   requires grad is being used in an in-place operation". The forward also reads
   `pixel_values.shape` and cannot take the list of differently sized (3, H, W) tiles that
   dynamic-resolution processors return for a batch of images of different sizes. The forward
   is replaced by one that merges out of place (masked_scatter, the same values) and runs the
   vision tower per tile group. Language model call and loss are the original's.

3. RADIO vision towers compute the `summary_idxs` buffer in __init__; checkpoints that do not
   store it get zeros or uninitialized memory from the meta-device load (4-bit loads then hit a
   device-side assert). It is recomputed from the config on the first forward.
"""

import functools
import inspect
import re
import sys

import torch

from .common import TEMPORARY_PATCHES

__all__ = [
    "repair_remote_hybrid_cache",
    "repair_internvl_style_forward",
    "repair_radio_summary_idxs",
    "repair_remote_modules",
]


def _get_query_offset(self, layer_idx = 0):
    # transformers.Cache.get_query_offset: equal to the cached length except for MTP caches.
    return self.get_seq_length(layer_idx)


def repair_remote_hybrid_cache(cls):
    """Add `get_query_offset` to a cache-like class that predates it. True if added."""
    if not isinstance(cls, type) or "get_query_offset" in dir(cls):
        return False
    if not (callable(getattr(cls, "get_seq_length", None)) and callable(getattr(cls, "get_mask_sizes", None))):
        return False
    try:
        from transformers.cache_utils import Cache
        if issubclass(cls, Cache):
            return False
    except Exception:
        pass
    cls.get_query_offset = _get_query_offset
    return True


_INPLACE_MERGE = re.compile(
    r"(inputs?_embeds)\[\s*selected\s*\]\s*=\s*\1\[\s*selected\s*\]\s*\*\s*0\.0\s*\+\s*vit_embeds"
)
_REQUIRED_PARAMETERS = ("pixel_values", "input_ids", "image_flags", "labels")


def _is_internvl_style_forward(cls):
    original = cls.__dict__.get("forward")
    if original is None or getattr(original, "_unsloth_internvl_forward", False):
        return False
    if not callable(getattr(cls, "extract_feature", None)):
        return False
    try:
        parameters = inspect.signature(original).parameters
        source = inspect.getsource(original)
    except (TypeError, ValueError, OSError):
        return False
    if any(name not in parameters for name in _REQUIRED_PARAMETERS):
        return False
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values()):
        return False
    return _INPLACE_MERGE.search(source) is not None


def _tile_groups(pixel_values):
    """Tensor or list of (3, H, W) / (n, 3, H, W) tensors -> list of 4D tile batches."""
    items = list(pixel_values) if isinstance(pixel_values, (list, tuple)) else [pixel_values]
    items = [x.unsqueeze(0) if x.dim() == 3 else x for x in items]
    if len(items) > 1 and all(x.shape[1:] == items[0].shape[1:] for x in items):
        # Same tile size: one vision call, as the original forward does for a stacked tensor.
        return [torch.cat(items, dim = 0)]
    return items


def _image_features(self, pixel_values, image_flags, hidden_size):
    groups = _tile_groups(pixel_values)
    n_tiles = sum(int(g.shape[0]) for g in groups)
    flags = None
    if image_flags is not None:
        flags = image_flags.reshape(-1)
        if flags.numel() != n_tiles:
            if bool((flags == 1).all()):
                # All-ones flags of another length carry no selection.
                flags = None
            else:
                raise ValueError(
                    f"Unsloth: image_flags has {flags.numel()} entries but pixel_values has {n_tiles} tiles."
                )
    features, start = [], 0
    for group in groups:
        feature = self.extract_feature(group)
        count = int(group.shape[0])
        if flags is not None:
            keep = flags[start:start + count].to(feature.device) == 1
            feature = feature[keep]
        start += count
        features.append(feature.reshape(-1, hidden_size))
    return torch.cat(features, dim = 0) if len(features) > 1 else features[0]


def repair_internvl_style_forward(cls):
    """Replace an InternVL-style forward with an out-of-place, ragged-aware one. True if replaced."""
    if not _is_internvl_style_forward(cls):
        return False
    original = cls.__dict__["forward"]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        pixel_values = a.get("pixel_values")
        input_ids = a.get("input_ids")
        inputs_embeds = a.get("inputs_embeds", a.get("input_embeds"))
        labels = a.get("labels")
        return_dict = a.get("return_dict")
        if return_dict is None:
            config = self.config
            return_dict = getattr(config, "use_return_dict", getattr(config, "return_dict", True))

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            hidden_size = inputs_embeds.shape[-1]
            vit_embeds = _image_features(self, pixel_values, a.get("image_flags"), hidden_size)
            selected = (input_ids == self.img_context_token_id).to(inputs_embeds.device)
            n_tokens = int(selected.sum())
            if vit_embeds.shape[0] < n_tokens:
                raise ValueError(
                    f"Unsloth: {n_tokens} image tokens in input_ids but only {vit_embeds.shape[0]} image features."
                )
            # The original keeps the first n_tokens features when there are more.
            vit_embeds = vit_embeds[:n_tokens].to(inputs_embeds.device, inputs_embeds.dtype)
            # Out of place: same values as `x[selected] = x[selected] * 0.0 + vit_embeds`.
            inputs_embeds = inputs_embeds.masked_scatter(selected.unsqueeze(-1), vit_embeds)

        outputs = self.language_model(
            inputs_embeds = inputs_embeds,
            attention_mask = a.get("attention_mask"),
            position_ids = a.get("position_ids"),
            past_key_values = a.get("past_key_values"),
            use_cache = a.get("use_cache"),
            output_attentions = a.get("output_attentions"),
            output_hidden_states = a.get("output_hidden_states"),
            return_dict = return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = torch.nn.CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + tuple(outputs[1:])
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values = getattr(outputs, "past_key_values", None),
            hidden_states = getattr(outputs, "hidden_states", None),
            attentions = getattr(outputs, "attentions", None),
        )

    forward._unsloth_internvl_forward = True
    forward.__signature__ = signature
    cls.forward = forward
    cls._unsloth_original_internvl_forward = original
    return True


def _expected_summary_idxs(config):
    args = getattr(config, "args", None)
    teachers = args.get("teachers") if isinstance(args, dict) else getattr(args, "teachers", None)
    if not isinstance(teachers, (list, tuple)):
        return None
    return [i for i, t in enumerate(teachers) if not isinstance(t, dict) or t.get("use_summary", True)]


def _restore_summary_idxs(model):
    base = getattr(model, "radio_model", None)
    buffer = getattr(base, "summary_idxs", None)
    if not isinstance(buffer, torch.Tensor) or buffer.device.type == "meta":
        return
    expected = _expected_summary_idxs(getattr(model, "config", None))
    if expected is None or buffer.numel() != len(expected):
        return
    expected = torch.tensor(expected, dtype = buffer.dtype)
    if not torch.equal(buffer.detach().cpu(), expected):
        with torch.no_grad():
            buffer.copy_(expected.to(buffer.device))


def repair_radio_summary_idxs(cls):
    """RADIO (C-RADIO vision towers) computes the `summary_idxs` buffer in __init__ and checkpoints
    such as Nemotron-3-Nano-Omni do not store it. transformers 5 builds the model on the meta
    device, so the missing buffer comes back as zeros or uninitialized memory; 4-bit loads then
    index out of range (`all_summary[:, self.summary_idxs]`, a device-side assert). Recompute it
    from the config on the first forward. True if the forward was wrapped."""
    if getattr(cls, "__name__", "") != "RADIOModel":
        return False
    original = cls.__dict__.get("forward")
    init = cls.__dict__.get("__init__")
    if original is None or init is None or getattr(original, "_unsloth_summary_idxs", False):
        return False
    try:
        if "summary_idxs" not in inspect.getsource(init):
            return False
    except (TypeError, OSError):
        return False

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if not self.__dict__.get("_unsloth_summary_idxs_checked", False):
            _restore_summary_idxs(self)
            self.__dict__["_unsloth_summary_idxs_checked"] = True
        return original(self, *args, **kwargs)

    forward._unsloth_summary_idxs = True
    cls.forward = forward
    return True


_REPAIRED_MODULES = set()


def repair_remote_modules():
    """Repair every loaded remote-code module once. Returns the repaired class names."""
    repaired = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("transformers_modules"):
            continue
        key = (name, id(module))
        if key in _REPAIRED_MODULES:
            continue
        _REPAIRED_MODULES.add(key)
        for value in list(vars(module).values()):
            if not isinstance(value, type) or getattr(value, "__module__", None) != name:
                continue
            try:
                if repair_remote_hybrid_cache(value):
                    repaired.append(f"{value.__name__}.get_query_offset")
                if repair_internvl_style_forward(value):
                    repaired.append(f"{value.__name__}.forward")
                if repair_radio_summary_idxs(value):
                    repaired.append(f"{value.__name__}.summary_idxs")
            except Exception:
                continue
    return repaired


def patch_remote_code_vlm():
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except Exception:
        return
    original = getattr(dynamic_module_utils, "get_class_in_module", None)
    if original is None or getattr(original, "_unsloth_remote_repair", False):
        return

    @functools.wraps(original)
    def get_class_in_module(*args, **kwargs):
        cls = original(*args, **kwargs)
        try:
            repair_remote_modules()
        except Exception:
            pass
        return cls

    get_class_in_module._unsloth_remote_repair = True
    dynamic_module_utils.get_class_in_module = get_class_in_module
    try:
        repair_remote_modules()
    except Exception:
        pass
pass
TEMPORARY_PATCHES.append(patch_remote_code_vlm)
