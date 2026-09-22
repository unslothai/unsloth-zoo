# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: narrowed exports. The upstream __init__ also imported
# fused_chunk_simple_gla and parallel_simple_gla, whose closures (fla.ops.common
# .fused_chunk and the parallel attention kernels) are not vendored. This copy
# exposes only the two entry points remote-code lightning-attention models import
# (for example inclusionAI Ling 2.5 / 2.6, `BailingMoeV2_5`):
# `from fla.ops.simple_gla.chunk import chunk_simple_gla` and
# `from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla`.
# Original MIT header preserved above.

from .chunk import chunk_simple_gla
from .fused_recurrent import fused_recurrent_simple_gla

__all__ = [
    'chunk_simple_gla',
    'fused_recurrent_simple_gla',
]
