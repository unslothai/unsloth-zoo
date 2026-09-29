# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: the TileLang backend is not registered because its kernel
# (backends/tilelang/chunk_bwd_dqkg.py) is not vendored. Original MIT header preserved above.

"""KDA backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.kda.backends.flashkda import FlashKDABackend

kda_registry = BackendRegistry("kda")
kda_registry.register(FlashKDABackend())


__all__ = ['dispatch', 'kda_registry']
