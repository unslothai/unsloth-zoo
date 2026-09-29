# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: narrowed exports. The upstream fla/modules/__init__.py
# imported the full module zoo (convolutions, bitlinear, cross-entropy, MLP,
# rotary, ...). This vendored copy exposes only FusedRMSNormGated, which the
# transformers gated-deltanet models import via `from fla.modules import
# FusedRMSNormGated`. Original MIT header preserved above.

from fla.modules.fused_norm_gate import FusedRMSNormGated

__all__ = [
    "FusedRMSNormGated",
]
