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

"""Route transformers @use_experts_implementation experts through Unsloth's MoE forward; tell the 4-bit quantizer which it may pack."""
import contextlib
import functools
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
    if "experts_interface" not in _LAZY:
        try:
            from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
        except Exception:
            ALL_EXPERTS_FUNCTIONS = None
        _LAZY["experts_interface"] = ALL_EXPERTS_FUNCTIONS
    return _LAZY["experts_interface"]


def _forward_is_unsloth(forward) -> bool:
    fn = getattr(forward, "__func__", forward)
    if getattr(fn, "_unsloth_moe_forward", False):
        return True
    module = getattr(fn, "__module__", "") or ""
    # Compiled cache copy loads as `unsloth_cached_moe_utils` or bare `moe_utils`; other names are user code.
    if module.startswith("unsloth_zoo.") or module in ("unsloth_cached_moe_utils", "moe_utils"):
        return True
    return getattr(fn, "__name__", "") == "forward_moe_backend"


def expert_forward_is_handled(module: nn.Module) -> bool:
    cls = type(module)
    forward = getattr(cls, "forward", None)
    if forward is None:
        return False
    if _forward_is_unsloth(forward):
        return True
    if not hasattr(forward, "__wrapped__"):
        return False
    if getattr(module, "has_gate", True) is False or _has_custom_gate(module):
        return False
    config = getattr(module, "config", None)
    implementation = getattr(config, "_experts_implementation", None)
    return implementation == UNSLOTH_EXPERTS_IMPLEMENTATION


_LAZY = {}


def _default_apply_gate_or_none():
    if "default_apply_gate" not in _LAZY:
        try:
            from transformers.integrations.moe import _default_apply_gate
        except Exception:
            _default_apply_gate = None
        _LAZY["default_apply_gate"] = _default_apply_gate
    return _LAZY["default_apply_gate"]


def _moe_utils_module():
    module = _LAZY.get("moe_utils")
    if module is None:
        from . import moe_utils as module
        _LAZY["moe_utils"] = module
    return module


def _has_custom_gate(module) -> bool:
    _default_apply_gate = _default_apply_gate_or_none()
    if _default_apply_gate is None:
        return False
    gate = getattr(type(module), "_apply_gate", None)
    if gate is None or gate is _default_apply_gate:
        return False
    if getattr(type(module), "_unsloth_own_apply_gate", False):
        return False
    return "GptOss" not in type(module).__name__


def _packs_fp4_experts(model) -> bool:
    config = getattr(model, "config", None)
    configs = [config]
    try:
        configs.append(config.get_text_config())
    except Exception:
        pass
    return any(getattr(c, "expert_dtype", None) == "fp4" for c in configs if c is not None)


_EXPERT_STACK_NAMES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")


def _holds_packed_4bit_experts(model) -> bool:
    try:
        for module in model.modules():
            for name in _EXPERT_STACK_NAMES:
                param = module._parameters.get(name) if hasattr(module, "_parameters") else None
                if param is not None and type(param).__name__ == "Params4bit":
                    return True
    except Exception:
        return False
    return False


# Sub-config ids under expert parallelism (distributed_config is set on the outer config only); kept off configs so never serialized.
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
    config = getattr(model, "config", None)
    distributed = getattr(config, "distributed_config", None)
    if bool(getattr(distributed, "enable_expert_parallel", False)):
        _mark_subconfigs_expert_parallel(config)
        return True
    return config is not None and id(config) in _EXPERT_PARALLEL_SUBCONFIGS


_TRANSFORMERS_GROUPED_MM = None
# transformers 5.3+ run grouped_mm anywhere via their own fallback; 5.2 needs torch._grouped_mm itself.
_TRANSFORMERS_GROUPED_MM_HAS_FALLBACK = False
# generate() swaps grouped_mm for batched_mm while decoding; this depth counter lets the dense route follow.
_TRANSFORMERS_BATCHED_MM = None
_DECODING_DEPTH = 0

_DENSE_STACK_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _dense_experts_without_expert_lora(module) -> bool:
    params = module._parameters
    state = module.__dict__
    found = False
    for name in _EXPERT_STACK_NAMES:
        param = params.get(name)
        if param is None:
            continue
        if type(param) is not nn.Parameter or param.dtype not in _DENSE_STACK_DTYPES:
            return False
        if state.get("_unsloth_lora_" + name) is not None:
            return False
        found = True
    return found


def unsloth_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if (
        _TRANSFORMERS_GROUPED_MM is not None
        and _dense_experts_without_expert_lora(self)
        and (_TRANSFORMERS_GROUPED_MM_HAS_FALLBACK or _grouped_mm_supported())
    ):
        if _DECODING_DEPTH and _TRANSFORMERS_BATCHED_MM is not None and not torch.is_grad_enabled():
            return _TRANSFORMERS_BATCHED_MM(self, hidden_states, top_k_index, top_k_weights)
        return _TRANSFORMERS_GROUPED_MM(self, hidden_states, top_k_index, top_k_weights)
    return _unsloth_experts_dispatch(self, hidden_states, top_k_index, top_k_weights)


def _grouped_mm_supported() -> bool:
    moe_utils = _moe_utils_module()
    supported = moe_utils._TORCH_GROUPED_MM_SUPPORTED
    if supported is None:
        supported = moe_utils._check_torch_grouped_mm_supported()
    return bool(supported)


# Out of Dynamo: inlining the dispatch made AOT autograd save every dequantized expert stack (OOM on Inkling-Small).
@torch.compiler.disable
def _unsloth_experts_dispatch(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if getattr(self, "has_gate", True) is False or _has_custom_gate(self):
        interface = _experts_interface()
        fallback = interface["grouped_mm"] if interface is not None and "grouped_mm" in interface else None
        if fallback is not None:
            # Without torch._grouped_mm support transformers' grouped_mm fails: take the eager forward.
            if not _moe_utils_module()._check_torch_grouped_mm_supported():
                fallback = None
        if fallback is None:
            return type(self).forward.__wrapped__(self, hidden_states, top_k_index, top_k_weights)
        return fallback(self, hidden_states, top_k_index, top_k_weights)
    return _moe_utils_module().get_forward_moe_backend()(self, hidden_states, top_k_index, top_k_weights)


def _implementation_values(model):
    getter = getattr(model, "get_experts_implementation", None)
    try:
        implementation = getter() if callable(getter) else getattr(model.config, "_experts_implementation", None)
    except Exception:
        implementation = getattr(getattr(model, "config", None), "_experts_implementation", None)
    return list(implementation.values()) if isinstance(implementation, dict) else [implementation]


def _patch_decode_switch():
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        return
    original = GenerationMixin.__dict__.get("_optimize_model_for_decode")
    if original is None or getattr(original, "_unsloth_patched", False):
        return

    @functools.wraps(original)
    @contextlib.contextmanager
    def _optimize_model_for_decode(self, *args, **kwargs):
        global _DECODING_DEPTH
        with original(self, *args, **kwargs):
            switch = self.device.type != "cpu" and UNSLOTH_EXPERTS_IMPLEMENTATION in _implementation_values(self)
            if switch:
                _DECODING_DEPTH += 1
            try:
                yield
            finally:
                if switch:
                    _DECODING_DEPTH -= 1

    _optimize_model_for_decode._unsloth_patched = True
    GenerationMixin._optimize_model_for_decode = _optimize_model_for_decode


def patch_experts_interface():
    interface = _experts_interface()
    if interface is None:
        return
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as e:
        return logger.warning(f"Unsloth: could not patch the experts interface: {e}")

    global _TRANSFORMERS_GROUPED_MM, _TRANSFORMERS_BATCHED_MM, _TRANSFORMERS_GROUPED_MM_HAS_FALLBACK
    if _TRANSFORMERS_GROUPED_MM is None:
        try:
            _TRANSFORMERS_GROUPED_MM = interface["grouped_mm"] if "grouped_mm" in interface else None
            _TRANSFORMERS_BATCHED_MM = interface["batched_mm"] if "batched_mm" in interface else None
        except Exception:
            _TRANSFORMERS_GROUPED_MM = _TRANSFORMERS_BATCHED_MM = None
        try:
            import transformers.integrations.moe as transformers_moe
            _TRANSFORMERS_GROUPED_MM_HAS_FALLBACK = hasattr(transformers_moe, "_can_use_grouped_mm")
        except Exception:
            _TRANSFORMERS_GROUPED_MM_HAS_FALLBACK = False
    _patch_decode_switch()
    if UNSLOTH_EXPERTS_IMPLEMENTATION not in interface:
        interface[UNSLOTH_EXPERTS_IMPLEMENTATION] = unsloth_experts_forward

    original = getattr(PreTrainedModel, "get_correct_experts_implementation", None)
    if original is None or getattr(original, "_unsloth_patched", False):
        return

    def get_correct_experts_implementation(self, requested_experts):
        # Only the default is ours: a user-chosen implementation is kept and its experts stay unpacked.
        if _expert_parallel_requested(self):
            if requested_experts == UNSLOTH_EXPERTS_IMPLEMENTATION:
                # Expert parallel routing emits `num_experts` sentinels that only transformers' implementations mask.
                logger.warning(
                    "Unsloth: the 'unsloth' experts implementation does not support expert "
                    "parallelism; using transformers' default instead."
                )
                requested_experts = None
            return original(self, requested_experts)
        if requested_experts is None and not _packs_fp4_experts(self):
            return UNSLOTH_EXPERTS_IMPLEMENTATION
        if requested_experts == UNSLOTH_EXPERTS_IMPLEMENTATION:
            # transformers 5.0 to 5.6 validate nested re-checks against a fixed name list, not the registry.
            return UNSLOTH_EXPERTS_IMPLEMENTATION
        if requested_experts not in (None, UNSLOTH_EXPERTS_IMPLEMENTATION) and _holds_packed_4bit_experts(self):
            # After a 4-bit load the experts are already packed; only a load-time choice leaves them unpacked.
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
