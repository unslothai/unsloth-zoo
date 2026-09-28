# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: narrowed exports. The upstream __init__ also imported the
# `naive` reference implementation (the vendored copy's only einops dependency).
# This vendored copy exposes only the Triton fast-path entry points that the
# transformers gated-deltanet models import via
# `from fla.ops.gated_delta_rule import chunk_gated_delta_rule,
# fused_recurrent_gated_delta_rule`. Original MIT header preserved above.

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
