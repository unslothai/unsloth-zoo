# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: chunk_kda is exported through unsloth_zoo's safe_gate router, which passes
# safe_gate=True for GLM-5.3-Flash's bounded gates only (see temporary_patches/fla_vendor.py).
# Original MIT header preserved above.

from .chunk import chunk_kda
from .fused_recurrent import fused_recurrent_kda

try:
    from unsloth_zoo.temporary_patches.fla_vendor import _wrap_kda_safe_gate
    chunk_kda = _wrap_kda_safe_gate(chunk_kda)
except Exception:
    pass

__all__ = [
    "chunk_kda",
    "fused_recurrent_kda",
]
