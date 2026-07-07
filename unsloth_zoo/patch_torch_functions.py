
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

__all__ = [
    "patch_torch_functions",
]

import os
from .temporary_patches.common import torch_compile, UNSLOTH_ENABLE_LOGGING
from .log import logger
from torch import Tensor
import torch
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction, grad
from torch.nn.functional import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_variadic,
    normalize, 
    np,
)
from typing import Callable, List, Optional, Tuple, Union


@torch_compile
def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Apply Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            layer_norm,
            (input, weight, bias),
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        ).to(input.dtype)
    return torch.layer_norm(
        input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled
    ).to(input.dtype)
pass


@torch_compile
def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""Compute the cross entropy loss between input logits and target.

    See :class:`~torch.nn.CrossEntropyLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        ).to(input.dtype)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return torch._C._nn.cross_entropy_loss(
        input,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    ).to(input.dtype)
pass


def patch_torch_functions():
    # All Unsloth Zoo code licensed under LGPLv3
    if not hasattr(torch.nn.functional, "_uncompiled_layer_norm"):
        torch.nn.functional._uncompiled_layer_norm = torch.nn.functional.layer_norm
        torch.nn.functional.layer_norm = layer_norm
    # Don't compile cross_entropy (too many errors; already compiled elsewhere)
    # if not hasattr(torch.nn.functional, "_uncompiled_cross_entropy"):
    #     torch.nn.functional._uncompiled_cross_entropy = torch.nn.functional.cross_entropy
    #     torch.nn.functional.cross_entropy = cross_entropy
pass


# Patch TorchAO functions
try:
    import torchao.quantization.qat.fake_quantizer
    if not hasattr(torchao.quantization.qat.fake_quantizer, "__UNSLOTH_PATCHED__"):
        qat_classes = dir(torchao.quantization.qat.fake_quantizer)
        for qat_class in qat_classes:
            if qat_class.startswith("_"): continue
            qat_class = getattr(torchao.quantization.qat.fake_quantizer, qat_class)
            if hasattr(qat_class, "forward"):
                # Skip already compiled functions
                if not hasattr(qat_class.forward, "get_compiler_config"):
                    qat_class.forward = torch_compile(qat_class.forward)
        torchao.quantization.qat.fake_quantizer.__UNSLOTH_PATCHED__ = True
except Exception as e:
    if UNSLOTH_ENABLE_LOGGING:
        logger.warning(f"TorchAO patching failed with exception = {str(e)}")
pass
