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

"""Save expert-LoRA adapters of re-laid-out expert classes in the factorization vanilla PEFT builds.

PEFT's ParamWrapper reads a 3D expert stack [E, d1, d2] as (in, out) = (d2, d1) unless the experts module says
`is_transposed`. Unsloth changes that answer for two classes: it flags Llama4TextExperts `is_transposed = True`
(its stacks really are [E, in, out]) and stores Qwen3VLMoeTextExperts transposed ([E, H, 2I] instead of
[E, 2I, H]). PEFT then puts lora_A on the other dimension of the stack, so an adapter trained under Unsloth has
lora_A [E*R, d1] / lora_B [d2, R*E] where vanilla transformers + PEFT expect lora_A [E*R, d2] / lora_B [d1, R*E]
and refuse to load it (size mismatch).

Both are rank-R factorizations of the same per-expert delta: Unsloth's is delta_e = A_e^T B_e^T and vanilla's is
B'_e A'_e, so B'_e = A_e^T and A'_e = B_e^T, keeping PEFT's packing (A read as (E, R, in), B as (out, R, E)). The
map is an entry permutation, its own inverse, and changes no value. get_peft_model_state_dict (every save,
including trainer checkpoints) writes the vanilla factors; set_peft_model_state_dict (every load) maps a vanilla
pair back when its shapes only fit swapped. A square stack (d1 == d2) cannot be told apart by shape, so it is left
as is on both sides: no real Llama-4 or Qwen3-VL-MoE checkpoint has one.

The legacy lora_B packing (UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert, column e*R + r is expert e) has the same
[out, R*E] shape as PEFT's (column r*E + e), so no PEFT can tell them apart and every one of them reads it with the
experts mixed. Converting it to PEFT's packing on save would not help: set_peft_model_state_dict gets bare tensors,
so a later load in the same legacy process could not tell a converted file from one an older Unsloth wrote. Such a
lora_B is saved as [out, E, R] instead, which says its packing in its shape: vanilla PEFT (whose lora_B is always the
2D weight of nn.Linear(R*E, out)) refuses it with a size mismatch on every version, and Unsloth's loaders flatten it
back to the 2D bytes, whose packing the adapter_config.json lora_B_layout marker still records.
"""

from __future__ import annotations

import os
import sys

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = [
    "SWAPPED_FACTORS_FLAG",
    "swap_expert_lora_factors",
    "patch_expert_lora_factor_swap",
]

# Set on an experts class by the zoo patch that changes how PEFT factors its stacks: True when the stacks are
# stored transposed (any PEFT), "is_transposed" when only the class's is_transposed flag differs (a PEFT that
# ignores is_transposed, such as 0.18, factors it as vanilla does).
SWAPPED_FACTORS_FLAG = "_unsloth_expert_lora_factors_swapped"
_PEFT_READS_IS_TRANSPOSED = []


def _peft_reads_is_transposed():
    if not _PEFT_READS_IS_TRANSPOSED:
        try:
            import inspect
            from peft.tuners.lora.layer import ParamWrapper
            _PEFT_READS_IS_TRANSPOSED.append("is_transposed" in inspect.getsource(ParamWrapper))
        except Exception:
            _PEFT_READS_IS_TRANSPOSED.append(True)
    return _PEFT_READS_IS_TRANSPOSED[0]


def _factors_swapped(experts_class):
    flag = getattr(experts_class, SWAPPED_FACTORS_FLAG, False)
    if flag == "is_transposed":
        return _peft_reads_is_transposed()
    return bool(flag)


def swap_expert_lora_factors(lora_A: torch.Tensor, lora_B: torch.Tensor, num_experts: int):
    """(A [E*R, a_in], B [b_out, R*E]) -> (A' [E*R, b_out], B' [a_in, R*E]) with B'_e = A_e^T, A'_e = B_e^T."""
    E = int(num_experts)
    rows, a_in = lora_A.shape
    b_out, cols = lora_B.shape
    if E <= 0 or rows % E or cols != rows:
        raise ValueError(f"Unsloth: cannot swap expert LoRA factors {tuple(lora_A.shape)} / {tuple(lora_B.shape)} "
                         f"over {E} experts")
    R = rows // E
    new_B = lora_A.reshape(E, R, a_in).permute(2, 1, 0).reshape(a_in, R * E).contiguous()
    new_A = lora_B.reshape(b_out, R, E).permute(2, 1, 0).reshape(E * R, b_out).contiguous()
    return new_A, new_B


def _legacy_grouped(wrapper, adapter_name):
    """True when this wrapper reads lora_B with the legacy (out, E, R) packing (UNSLOTH_MOE_LORA_B_LAYOUT). Its
    factors are not swapped; lora_B is saved 3D instead (see _to_vanilla)."""
    try:
        from .moe_utils import LORA_B_LAYOUT_GROUPED_BY_EXPERT, moe_lora_b_layout_for_wrapper
        return moe_lora_b_layout_for_wrapper(wrapper, adapter_name) == LORA_B_LAYOUT_GROUPED_BY_EXPERT
    except Exception:
        return False


def _innermost(module):
    while type(module).__name__ == "ParamWrapper":
        module = module.base_layer
    return module


def _num_experts(experts):
    for name in ("num_experts", "num_local_experts"):
        value = getattr(experts, name, None)
        if isinstance(value, int) and value > 0:
            return value
    for param in experts.parameters(recurse = False):
        if param.dim() == 3:
            return param.shape[0]
    return None


def _swapped_wrappers(model):
    """{module name: num_experts} for every PEFT ParamWrapper over a flagged experts class."""
    out = {}
    for name, module in model.named_modules():
        if type(module).__name__ != "ParamWrapper":
            continue
        experts = _innermost(module)
        if not _factors_swapped(type(experts)):
            continue
        num_experts = _num_experts(experts)
        if num_experts:
            out[name] = num_experts
    return out


def _expert_wrappers(model):
    """{module name: num_experts} for every PEFT ParamWrapper over a multi-expert stack (any family)."""
    out = {}
    for name, module in model.named_modules():
        if type(module).__name__ != "ParamWrapper":
            continue
        num_experts = getattr(module, "num_experts", None)
        if isinstance(num_experts, int) and num_experts > 1:
            out[name] = num_experts
    return out


def _bare(name):
    while name.startswith("_orig_mod."):  # torch.compile wrappers
        name = name[len("_orig_mod."):]
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _pair_keys(state_dict, module_name):
    """lora_A / lora_B key pairs of `module_name` in a PEFT state dict (adapter name present or stripped). Keys and
    module names are matched without PEFT's `base_model.model.` prefix, which either side may carry (a PeftModel
    checkpoint loaded into a model from inject_adapter_in_model, or the reverse)."""
    target = _bare(module_name)
    pairs = []
    for key in state_dict:
        if ".lora_A." not in key or not key.endswith("weight"):
            continue
        name, middle = key.split(".lora_A.", 1)
        # The direct child only: a nested wrapper is `name.base_layer.lora_A...`.
        if _bare(name) != target or "base_layer" in middle:
            continue
        key_b = f"{name}.lora_B.{middle}"
        if key_b in state_dict:
            pairs.append((key, key_b))
    return pairs


def _to_vanilla(model, state_dict, adapter_name = "default"):
    wrappers = _swapped_wrappers(model)
    experts = _expert_wrappers(model)
    if not wrappers and not experts:
        return state_dict
    modules = dict(model.named_modules())
    state_dict = dict(state_dict)
    legacy = {name for name in experts if _legacy_grouped(modules[name], adapter_name)}
    for module_name in legacy:
        num_experts = experts[module_name]
        for _, key_b in _pair_keys(state_dict, module_name):
            b = state_dict[key_b]
            if b.dim() == 2 and b.shape[1] % num_experts == 0:
                # Legacy packing: same 2D shape as PEFT's, read wrong by every PEFT. [out, E, R] is refused instead.
                state_dict[key_b] = b.reshape(b.shape[0], num_experts, b.shape[1] // num_experts)
    for module_name, num_experts in wrappers.items():
        if module_name in legacy:
            continue
        for key_a, key_b in _pair_keys(state_dict, module_name):
            a, b = state_dict[key_a], state_dict[key_b]
            if a.shape[1] == b.shape[0]:
                continue   # square stack: ambiguous by shape, kept as trained
            state_dict[key_a], state_dict[key_b] = swap_expert_lora_factors(a, b, num_experts)
    return state_dict


def _flatten_grouped_lora_B(model, state_dict, adapter_name, experts, modules):
    """A legacy-packed lora_B saved as [out, E, R] back to the 2D bytes it was trained as. Their packing is then the
    adapter_config.json marker's business, as for an adapter an older Unsloth saved 2D: kept under
    UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert, repacked to PEFT's by PeftModel.load_adapter otherwise."""
    for module_name, num_experts in experts.items():
        lora_B_dict = getattr(modules[module_name], "lora_B", None)
        if lora_B_dict is None or adapter_name not in lora_B_dict:
            continue
        want_out = lora_B_dict[adapter_name].weight.shape[0]
        for _, key_b in _pair_keys(state_dict, module_name):
            b = state_dict[key_b]
            if b.dim() == 3 and b.shape[0] == want_out and b.shape[1] == num_experts:
                state_dict[key_b] = b.reshape(b.shape[0], -1)
    return state_dict


def _from_vanilla(model, state_dict, adapter_name):
    wrappers = _swapped_wrappers(model)
    experts = _expert_wrappers(model)
    if not wrappers and not experts:
        return state_dict
    modules = dict(model.named_modules())
    state_dict = _flatten_grouped_lora_B(model, dict(state_dict), adapter_name, experts, modules)
    for module_name, num_experts in wrappers.items():
        wrapper = modules[module_name]
        lora_A_dict = getattr(wrapper, "lora_A", None)
        if lora_A_dict is None or adapter_name not in lora_A_dict or _legacy_grouped(wrapper, adapter_name):
            continue
        # Orientation from lora_A's input dimension only: a hotswap may bring another rank.
        want_in = lora_A_dict[adapter_name].weight.shape[1]
        for key_a, key_b in _pair_keys(state_dict, module_name):
            a, b = state_dict[key_a], state_dict[key_b]
            if a.shape[1] == want_in:
                continue   # trained under Unsloth before this change, or square
            new_a, new_b = swap_expert_lora_factors(a, b, num_experts)
            if new_a.shape[1] == want_in:
                state_dict[key_a], state_dict[key_b] = new_a, new_b
    return state_dict


def _wrap_get(original):
    def get_peft_model_state_dict(model, state_dict = None, adapter_name = "default", *args, **kwargs):
        result = original(model, state_dict, adapter_name, *args, **kwargs)
        try:
            return _to_vanilla(model, result, adapter_name)
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: kept expert LoRA factors as trained: {e}")
            return result
    get_peft_model_state_dict._unsloth_factor_swap = True
    get_peft_model_state_dict.__wrapped__ = original
    return get_peft_model_state_dict


def _wrap_set(original):
    def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name = "default", *args, **kwargs):
        try:
            peft_model_state_dict = _from_vanilla(model, peft_model_state_dict, adapter_name)
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: loaded expert LoRA factors unchanged: {e}")
        return original(model, peft_model_state_dict, adapter_name, *args, **kwargs)
    set_peft_model_state_dict._unsloth_factor_swap = True
    set_peft_model_state_dict.__wrapped__ = original
    return set_peft_model_state_dict


def _wrap_hotswap(original):
    def hotswap_adapter_from_state_dict(model, state_dict, adapter_name, *args, **kwargs):
        try:
            state_dict = _from_vanilla(model, state_dict, adapter_name)
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: hotswapped expert LoRA factors unchanged: {e}")
        return original(model, state_dict, adapter_name, *args, **kwargs)
    hotswap_adapter_from_state_dict._unsloth_factor_swap = True
    hotswap_adapter_from_state_dict.__wrapped__ = original
    return hotswap_adapter_from_state_dict


def _converted_load_kwargs(model, kwargs, adapter_name, scratch_dir = None):
    """The load kwargs with converted factors, or the original kwargs when nothing needs converting (the
    file-backed path stays). Under disk offload, which indexes checkpoint files, the converted factors are written
    to a safetensors file in `scratch_dir` and passed as the checkpoint instead."""
    state_dict = kwargs.get("state_dict")
    files = list(kwargs.get("checkpoint_files") or [])
    from_files = state_dict is None and files and all(os.path.isfile(str(f)) for f in files)
    if from_files:
        # transformers' own reader: it keeps the torch.load safety check for .bin files.
        from transformers.modeling_utils import load_state_dict as _hf_load_state_dict
        state_dict = {}
        for f in files:
            state_dict.update(_hf_load_state_dict(str(f), map_location = "cpu"))
    if state_dict is None:
        return kwargs
    converted = _from_vanilla(model, state_dict, adapter_name)
    if all(converted.get(key) is value for key, value in state_dict.items()):
        return kwargs
    device_map = getattr(model, "hf_device_map", None) or {}
    if from_files and scratch_dir is not None and "disk" in set(device_map.values()):
        from safetensors.torch import save_file
        path = os.path.join(scratch_dir, "adapter_model.safetensors")
        save_file({key: value.contiguous() for key, value in converted.items()}, path)
        return dict(kwargs, checkpoint_files = [path])
    kwargs = dict(kwargs, state_dict = converted)
    if from_files:
        kwargs["checkpoint_files"] = None
    return kwargs


def _wrap_load_adapter(original):
    """transformers 5.x PeftAdapterMixin.load_adapter injects the adapter, then loads its tensors with
    _load_pretrained_model (not PEFT's set_peft_model_state_dict): convert the factors on that path too."""
    def load_adapter(self, peft_model_id = None, adapter_name = None, *args, **kwargs):
        name = adapter_name or "default"
        cls_load = type(self)._load_pretrained_model
        had = "_load_pretrained_model" in self.__dict__

        def _load_pretrained_model(*a, **k):
            model = k.get("model", self)
            # Only expert adapters need it (re-laid-out experts, or a legacy lora_B saved 3D): everything else
            # keeps transformers' own path (disk offload indexes the checkpoint files it was given).
            if _swapped_wrappers(model) or _expert_wrappers(model):
                try:
                    k = _converted_load_kwargs(model, k, name, scratch.name)
                except Exception as e:
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.warning(f"Unsloth: loaded expert LoRA factors unchanged: {e}")
            # Called once, outside the try: a real load error is raised, not retried.
            return cls_load(*a, **k)

        import tempfile
        scratch = tempfile.TemporaryDirectory(prefix = "unsloth_lora_swap_")
        self.__dict__["_load_pretrained_model"] = _load_pretrained_model
        try:
            return original(self, peft_model_id, adapter_name, *args, **kwargs)
        finally:
            if not had:
                self.__dict__.pop("_load_pretrained_model", None)
            scratch.cleanup()
    load_adapter._unsloth_factor_swap = True
    load_adapter.__wrapped__ = original
    return load_adapter


def _patch_transformers_load_adapter():
    try:
        from transformers.integrations.peft import PeftAdapterMixin
    except Exception:
        return
    original = PeftAdapterMixin.__dict__.get("load_adapter")
    if original is None or getattr(original, "_unsloth_factor_swap", False):
        return
    PeftAdapterMixin.load_adapter = _wrap_load_adapter(original)


def patch_expert_lora_factor_swap():
    try:
        import peft.utils.save_and_load as save_and_load
    except Exception:
        return
    try:
        import peft.utils.hotswap as hotswap
    except Exception:
        hotswap = None
    for source, name, wrap in (
        (save_and_load, "get_peft_model_state_dict", _wrap_get),
        (save_and_load, "set_peft_model_state_dict", _wrap_set),
        (hotswap, "hotswap_adapter_from_state_dict", _wrap_hotswap),
    ):
        original = getattr(source, name, None)
        if original is None or getattr(original, "_unsloth_factor_swap", False):
            continue
        patched = wrap(original)
        # peft re-exports these (peft, peft.utils, peft.peft_model, peft.mixed_model, ...): rebind every alias.
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("peft"):
                continue
            if getattr(module, name, None) is original:
                setattr(module, name, patched)
    _patch_transformers_load_adapter()
pass

TEMPORARY_PATCHES.append(patch_expert_lora_factor_swap)
