# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: narrowed exports. The upstream fla/ops/__init__.py
# eagerly imported every operator family. This vendored copy exposes nothing at
# the fla.ops level; consumers import fla.ops.gated_delta_rule directly. Kept as
# an empty (but real) package so submodule imports resolve. Original MIT header
# preserved above.

__all__: list[str] = []
