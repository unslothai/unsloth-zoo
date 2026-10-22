# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: narrowed exports. The upstream fla/__init__.py eagerly
# imported fla.layers and fla.models (pulling ~368 files). This vendored copy
# ships only the gated-delta-rule fast-path closure, so the top-level package
# just declares its version and stays a real, walkable package. Original MIT
# header preserved above.

__version__ = "0.5.1"

__all__: list[str] = []
