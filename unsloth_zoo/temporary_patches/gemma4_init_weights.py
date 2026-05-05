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

"""
Skip non-float weight init for Gemma-4 E-series MatFormer 4-bit checkpoints.

The E2B and E4B Gemma-4 unsloth-bnb-4bit checkpoints intentionally omit upper
layer K, V and K-norm projections plus the tied lm_head. Those layers run in
KV-shared mode, so the local k_proj / v_proj / k_norm tensors are never
invoked at runtime, and lm_head is rebound to embed_tokens by tie_weights.

transformers v5 fills the missing slots with torch.empty_like(param) against
the underlying bnb-4bit Params4bit, which preserves the uint8 quant_storage
dtype. Gemma4PreTrainedModel._init_weights then calls
super()._init_weights(...) which dispatches to init.normal_(module.weight)
on the uint8 tensor. PyTorch's normal_kernel_cuda is float-only, so the load
crashes with:

    RuntimeError: "normal_kernel_cuda" not implemented for 'Byte'

This patch wraps Gemma4PreTrainedModel._init_weights so that any module whose
primary weight tensor is non-floating-point is skipped. The placeholder is
either filled from the state_dict immediately afterwards, or belongs to a
KV-shared layer that the runtime never reads. The patch is a no-op on
transformers builds that do not ship Gemma-4, and a no-op on float (non
quantised) checkpoints because module.weight stays bfloat16 / float16 /
float32.
"""

import torch
import torch.nn as nn

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import logger


def _is_floating_point_dtype(dtype):
    if dtype is None:
        return False
    try:
        return dtype.is_floating_point
    except AttributeError:
        return False


def _module_has_only_non_float_params(module):
    """True if every direct param / buffer attached to this module is non-float
    and the module has at least one such attachment.

    Children get visited separately by initialize_weights' smart_apply, so we
    only inspect direct attachments here.
    """
    saw_any = False
    for p in module.parameters(recurse=False):
        saw_any = True
        if _is_floating_point_dtype(p.dtype):
            return False
    for b in module.buffers(recurse=False):
        saw_any = True
        if _is_floating_point_dtype(b.dtype):
            return False
    return saw_any


def _weight_is_non_float(module):
    """nn.Linear and bnb Linear4bit both expose `weight`. For E-series 4-bit
    placeholders this is a uint8 Params4bit, which init.normal_ cannot touch.
    Other Gemma-4 submodules (RMSNorm, Embedding, Router, Experts) keep float
    weights and are left to the original initializer.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        return False
    if not isinstance(weight, torch.Tensor):
        return False
    return not _is_floating_point_dtype(weight.dtype)


def patch_gemma4_init_weights_skip_non_float():
    """Wrap Gemma4PreTrainedModel._init_weights to skip non-float params.

    Idempotent. Safe no-op when transformers does not ship Gemma-4.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4PreTrainedModel,
        )
    except Exception:
        return

    if getattr(Gemma4PreTrainedModel, "_unsloth_init_weights_skip_non_float", False):
        return

    _original_init_weights = Gemma4PreTrainedModel._init_weights

    @torch.no_grad()
    def _patched_init_weights(self, module):
        if _weight_is_non_float(module):
            if UNSLOTH_ENABLE_LOGGING:
                cls_name = type(module).__name__
                weight = getattr(module, "weight", None)
                w_dtype = weight.dtype if isinstance(weight, torch.Tensor) else None
                logger.info(
                    f"Unsloth: Skipping _init_weights on {cls_name} "
                    f"(weight dtype={w_dtype}). Expected for bnb-4bit "
                    f"placeholders on Gemma-4 E-series KV-shared layers."
                )
            return

        if _module_has_only_non_float_params(module):
            if UNSLOTH_ENABLE_LOGGING:
                logger.info(
                    f"Unsloth: Skipping _init_weights on {type(module).__name__} "
                    "(all direct params and buffers are non-floating-point)."
                )
            return

        _original_init_weights(self, module)

    _patched_init_weights.__qualname__ = _original_init_weights.__qualname__
    _patched_init_weights.__name__ = getattr(
        _original_init_weights, "__name__", "_init_weights",
    )
    _patched_init_weights.__doc__ = getattr(_original_init_weights, "__doc__", None)
    _patched_init_weights.__wrapped__ = _original_init_weights

    Gemma4PreTrainedModel._original_init_weights_pre_unsloth = _original_init_weights
    Gemma4PreTrainedModel._init_weights = _patched_init_weights
    Gemma4PreTrainedModel._unsloth_init_weights_skip_non_float = True

    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: Patched Gemma4PreTrainedModel._init_weights to skip "
            "non-float params (E-series MatFormer bnb-4bit fix)."
        )


TEMPORARY_PATCHES.append(patch_gemma4_init_weights_skip_non_float)
