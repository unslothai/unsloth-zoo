# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Modified by Unsloth: the Triton-Ascend (Huawei NPU) backend is not registered
# because its kernels (backends/triton_ascend/) are not vendored, and the vendored
# tree is only injected on CUDA. With no backend registered, every dispatched
# causal_conv1d op runs its default Triton implementation, as upstream does on
# any non-NPU device. Original MIT header preserved above.

"""Module-level backends for FLA components such as rotary and cross-entropy."""

from fla.ops.backends import BackendRegistry, dispatch

modules_registry = BackendRegistry("modules")

__all__ = ['dispatch', 'modules_registry']
