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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Route every transformers `@use_experts_implementation` experts module through
Unsloth's MoE forward, and tell the 4-bit quantizer which experts it may pack.

transformers 5 dispatches the experts forward of every decorated experts class
(Inkling, Qwen3-MoE, Mixtral, Gemma-4, ...) through `ALL_EXPERTS_FUNCTIONS`,
keyed by `config._experts_implementation`, defaulting to its own "grouped_mm".
Unsloth used to patch the forward of a fixed list of classes by name, while the
4-bit quantizer claimed experts structurally (any module with 3-D gate_up_proj
and down_proj Parameters). A class on the second list but not the first, such
as InklingExperts, had its expert stacks packed to uint8 and then handed to
transformers' own grouped GEMM, which raised
"Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Byte".

Two things fix that for every architecture at once:

* an "unsloth" experts implementation registered with transformers and made
  the default, so any decorated experts class takes Unsloth's grouped path,
  with the class-name patches kept for transformers 4.x and for classes that
  are not decorated;
* `expert_forward_is_handled`, which the quantizer consults before packing a
  module: only experts whose forward is Unsloth's are quantized, everything
  else stays in the checkpoint dtype and trains through its own forward.
"""
import weakref

import torch
import torch.nn as nn

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger

UNSLOTH_EXPERTS_IMPLEMENTATION = "unsloth"

__all__ = [
    "UNSLOTH_EXPERTS_IMPLEMENTATION",
    "expert_forward_is_handled",
    "unsloth_experts_forward",
    "patch_experts_interface",
]


def _experts_interface():
    """transformers' registry, or None below transformers 5."""
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
        return ALL_EXPERTS_FUNCTIONS
    except Exception:
        return None


def _forward_is_unsloth(forward) -> bool:
    """True when `forward` is one of Unsloth's expert forwards: a function from
    unsloth_zoo (the per-class patches and the dispatcher), or the copy of the
    dispatcher that lives in unsloth_compiled_cache as `unsloth_cached_moe_utils`.

    A model class rewritten into `unsloth_compiled_module_<model>` keeps the
    model's own forward, so the compiled-cache module name alone does not count."""
    fn = getattr(forward, "__func__", forward)
    if getattr(fn, "_unsloth_moe_forward", False):
        return True
    module = getattr(fn, "__module__", "") or ""
    # The compiled cache copy loads as `unsloth_cached_moe_utils`, or as the bare `moe_utils`
    # a generated module imports; any other name that merely contains it is someone's code.
    if module.startswith("unsloth_zoo.") or module in ("unsloth_cached_moe_utils", "moe_utils"):
        return True
    return getattr(fn, "__name__", "") == "forward_moe_backend"


def expert_forward_is_handled(module: nn.Module) -> bool:
    """Whether this experts module's forward will read its weights the way
    Unsloth's grouped path does, so that packing them to 4-bit is safe.

    True for a class whose forward Unsloth patched by name (transformers 4.x and
    the explicit patches), and for a decorated class whose config dispatches to
    the "unsloth" implementation. False for anything else: a module whose
    forward is the model's own code (Llama-4's per-expert bmm, an ungated
    up-projection only module, a user-forced transformers implementation) must
    keep its weights in a dtype that forward understands."""
    cls = type(module)
    forward = getattr(cls, "forward", None)
    if forward is None:
        return False
    if _forward_is_unsloth(forward):
        return True
    if not hasattr(forward, "__wrapped__"):
        return False
    # Decorated by transformers: the implementation name on the config decides.
    # A class unsloth_experts_forward hands back to transformers is not handled.
    if getattr(module, "has_gate", True) is False or _has_custom_gate(module):
        return False
    config = getattr(module, "config", None)
    implementation = getattr(config, "_experts_implementation", None)
    return implementation == UNSLOTH_EXPERTS_IMPLEMENTATION


def _has_custom_gate(module) -> bool:
    """True when the experts class overrides transformers' default `_apply_gate`.

    DeepSeek-V4, GLM-5-Next, HY-V4, MiniMax-M3 and others clamp or offset gate and
    up; Unsloth's backends implement SiLU (or `act_fn`) and gpt-oss by name only."""
    try:
        from transformers.integrations.moe import _default_apply_gate
    except Exception:
        return False
    gate = getattr(type(module), "_apply_gate", None)
    if gate is None or gate is _default_apply_gate:
        return False
    # A class the bnb 4-bit route marked: every backend applies its own _apply_gate.
    if getattr(type(module), "_unsloth_own_apply_gate", False):
        return False
    return "GptOss" not in type(module).__name__


def _packs_fp4_experts(model) -> bool:
    """A config that stores experts as FP4 (``expert_dtype = "fp4"``, DeepSeek-V4): the FP8
    experts module holds them as packed int8, which only transformers' dispatchers decode."""
    config = getattr(model, "config", None)
    configs = [config]
    try:
        configs.append(config.get_text_config())
    except Exception:
        pass
    return any(getattr(c, "expert_dtype", None) == "fp4" for c in configs if c is not None)


_EXPERT_STACK_NAMES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")


def _holds_packed_4bit_experts(model) -> bool:
    """Expert stacks already packed as bitsandbytes 4-bit, which only Unsloth's dispatcher
    decodes; transformers' implementations would multiply the packed bytes."""
    try:
        for module in model.modules():
            for name in _EXPERT_STACK_NAMES:
                param = module._parameters.get(name) if hasattr(module, "_parameters") else None
                if param is not None and type(param).__name__ == "Params4bit":
                    return True
    except Exception:
        return False
    return False


# ids of sub-configs (text_config, ...) of a model loaded with expert parallelism: transformers
# sets distributed_config on the outer config only, and a composite model builds its language
# model from the same text_config object. Kept off the configs so it is never serialized.
_EXPERT_PARALLEL_SUBCONFIGS = set()


def _mark_subconfigs_expert_parallel(config) -> None:
    for key in getattr(config, "sub_configs", None) or {}:
        sub = getattr(config, key, None)
        if sub is None or id(sub) in _EXPERT_PARALLEL_SUBCONFIGS:
            continue
        try:
            weakref.finalize(sub, _EXPERT_PARALLEL_SUBCONFIGS.discard, id(sub))
        except TypeError:
            continue
        _EXPERT_PARALLEL_SUBCONFIGS.add(id(sub))
        _mark_subconfigs_expert_parallel(sub)


def _expert_parallel_requested(model) -> bool:
    """Expert parallelism routes non-local slots to a `num_experts` sentinel that
    only transformers' own implementations mask. The outer model's check runs
    before its nested models are built, so it marks their sub-configs too."""
    config = getattr(model, "config", None)
    distributed = getattr(config, "distributed_config", None)
    if bool(getattr(distributed, "enable_expert_parallel", False)):
        _mark_subconfigs_expert_parallel(config)
        return True
    return config is not None and id(config) in _EXPERT_PARALLEL_SUBCONFIGS


# Kept out of Dynamo like the per-model MoE block patches: a compiled MoE block
# that inlined the dispatch (dequantization, permutation, grouped GEMM) had
# AOT autograd save every dequantized expert stack for backward, 13 GB a layer
# on Inkling-Small, and ran out of memory on the first forward.
@torch.compiler.disable
def unsloth_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """The "unsloth" experts implementation: Unsloth's backend dispatcher
    (bnb 4-bit, FP8, grouped_mm, Triton, or the eager loop), for any decorated
    experts class. An ungated module (up_proj only) is not a shape the grouped
    path knows, and a class with its own `_apply_gate` (clamped or offset SwiGLU)
    computes an activation the backends do not reproduce, so both take
    transformers' own implementation."""
    if getattr(self, "has_gate", True) is False or _has_custom_gate(self):
        interface = _experts_interface()
        fallback = interface["grouped_mm"] if interface is not None and "grouped_mm" in interface else None
        if fallback is not None:
            # transformers' grouped_mm needs the same torch._grouped_mm support Unsloth's
            # backend selection checks; without it take the class's own eager forward.
            from .moe_utils import _check_torch_grouped_mm_supported
            if not _check_torch_grouped_mm_supported():
                fallback = None
        if fallback is None:
            return type(self).forward.__wrapped__(self, hidden_states, top_k_index, top_k_weights)
        return fallback(self, hidden_states, top_k_index, top_k_weights)
    from .moe_utils import get_forward_moe_backend
    return get_forward_moe_backend()(self, hidden_states, top_k_index, top_k_weights)


def patch_experts_interface():
    """Register the implementation and make it transformers' default choice."""
    interface = _experts_interface()
    if interface is None:
        return  # transformers 4.x: the class-name patches carry the MoE path
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as e:
        return logger.warning(f"Unsloth: could not patch the experts interface: {e}")

    if UNSLOTH_EXPERTS_IMPLEMENTATION not in interface:
        interface[UNSLOTH_EXPERTS_IMPLEMENTATION] = unsloth_experts_forward

    original = getattr(PreTrainedModel, "get_correct_experts_implementation", None)
    if original is None or getattr(original, "_unsloth_patched", False):
        return

    def get_correct_experts_implementation(self, requested_experts):
        # Only the default is ours. A user who asked for a specific implementation
        # keeps it, and the quantizer then leaves those experts unpacked.
        if _expert_parallel_requested(self):
            if requested_experts == UNSLOTH_EXPERTS_IMPLEMENTATION:
                # Expert parallel routing emits `num_experts` sentinels that only
                # transformers' own implementations mask.
                logger.warning(
                    "Unsloth: the 'unsloth' experts implementation does not support expert "
                    "parallelism; using transformers' default instead."
                )
                requested_experts = None
            return original(self, requested_experts)
        if requested_experts is None and not _packs_fp4_experts(self):
            return UNSLOTH_EXPERTS_IMPLEMENTATION
        if requested_experts not in (None, UNSLOTH_EXPERTS_IMPLEMENTATION) and _holds_packed_4bit_experts(self):
            # A runtime switch after a 4-bit load: the experts are already packed, and the
            # quantizer only leaves them unpacked for an implementation chosen at load time.
            raise RuntimeError(
                f"Unsloth: cannot switch the experts implementation to {requested_experts!r} "
                "after the experts were loaded in 4-bit; only the 'unsloth' implementation "
                "reads packed experts. Reload with `experts_implementation=...` instead."
            )
        return original(self, requested_experts)

    get_correct_experts_implementation._unsloth_patched = True
    patch_function(PreTrainedModel, "get_correct_experts_implementation", get_correct_experts_implementation, force = True)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: registered the 'unsloth' experts implementation as transformers' default")
pass

TEMPORARY_PATCHES.append(patch_experts_interface)
