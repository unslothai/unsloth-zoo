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
"""

from __future__ import annotations

import sys

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger

__all__ = [
    "SWAPPED_FACTORS_FLAG",
    "swap_expert_lora_factors",
    "patch_expert_lora_factor_swap",
]

# Set on an experts class by the zoo patch that changes how PEFT factors its stacks.
SWAPPED_FACTORS_FLAG = "_unsloth_expert_lora_factors_swapped"


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
        if not getattr(type(experts), SWAPPED_FACTORS_FLAG, False):
            continue
        num_experts = _num_experts(experts)
        if num_experts:
            out[name] = num_experts
    return out


def _pair_keys(state_dict, module_name):
    """lora_A / lora_B key pairs of `module_name` in a PEFT state dict (adapter name present or stripped)."""
    names = [module_name]
    for prefix in ("base_model.model.", "base_model."):
        if module_name.startswith(prefix):
            names.append(module_name[len(prefix):])
    pairs = []
    for key in state_dict:
        for name in names:
            head = f"{name}.lora_A."
            if not key.startswith(head) or not key.endswith("weight"):
                continue
            # The direct child only: a nested wrapper is `name.base_layer.lora_A...`.
            middle = key[len(head):]
            if "base_layer" in middle:
                continue
            key_b = f"{name}.lora_B.{middle}"
            if key_b in state_dict:
                pairs.append((key, key_b))
    return pairs


def _to_vanilla(model, state_dict):
    wrappers = _swapped_wrappers(model)
    if not wrappers:
        return state_dict
    state_dict = dict(state_dict)
    for module_name, num_experts in wrappers.items():
        for key_a, key_b in _pair_keys(state_dict, module_name):
            a, b = state_dict[key_a], state_dict[key_b]
            if a.shape[1] == b.shape[0]:
                continue   # square stack: ambiguous by shape, kept as trained
            state_dict[key_a], state_dict[key_b] = swap_expert_lora_factors(a, b, num_experts)
    return state_dict


def _from_vanilla(model, state_dict, adapter_name):
    wrappers = _swapped_wrappers(model)
    if not wrappers:
        return state_dict
    modules = dict(model.named_modules())
    state_dict = dict(state_dict)
    for module_name, num_experts in wrappers.items():
        wrapper = modules[module_name]
        lora_A_dict = getattr(wrapper, "lora_A", None)
        if lora_A_dict is None or adapter_name not in lora_A_dict:
            continue
        lora_A = lora_A_dict[adapter_name]
        want = tuple(lora_A.weight.shape)
        for key_a, key_b in _pair_keys(state_dict, module_name):
            a, b = state_dict[key_a], state_dict[key_b]
            if tuple(a.shape) == want:
                continue   # trained under Unsloth before this change, or square
            new_a, new_b = swap_expert_lora_factors(a, b, num_experts)
            if tuple(new_a.shape) == want:
                state_dict[key_a], state_dict[key_b] = new_a, new_b
    return state_dict


def _wrap_get(original):
    def get_peft_model_state_dict(model, state_dict = None, adapter_name = "default", *args, **kwargs):
        result = original(model, state_dict, adapter_name, *args, **kwargs)
        try:
            return _to_vanilla(model, result)
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


def patch_expert_lora_factor_swap():
    try:
        import peft.utils.save_and_load as save_and_load
    except Exception:
        return
    for name, wrap in (("get_peft_model_state_dict", _wrap_get), ("set_peft_model_state_dict", _wrap_set)):
        original = getattr(save_and_load, name, None)
        if original is None or getattr(original, "_unsloth_factor_swap", False):
            continue
        patched = wrap(original)
        # peft re-exports both (peft, peft.utils, peft.peft_model, peft.mixed_model, ...): rebind every alias.
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("peft"):
                continue
            if getattr(module, name, None) is original:
                setattr(module, name, patched)
pass

TEMPORARY_PATCHES.append(patch_expert_lora_factor_swap)
