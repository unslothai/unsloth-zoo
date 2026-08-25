# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Shared helpers for the MLX int8 prefill tests.

The tests run on any MLX backend, including a Linux CUDA build, because what they assert
is either dispatch logic or algorithm arithmetic. The Metal kernels themselves are
exercised by the macOS leg of mlx-ci.yml.
"""

import os

# The Metal backend cannot run off Apple silicon; the portable backend implements the
# identical arithmetic in plain MLX ops.
os.environ.setdefault("UNSLOTH_MLX_INT8_BACKEND", "portable")


def make_quantized_linear(in_dims = 1024, out_dims = 2048, bits = 4, group_size = 64, bias = False):
    import mlx.nn as nn
    linear = nn.Linear(in_dims, out_dims, bias = bias)
    return nn.QuantizedLinear.from_linear(linear, group_size = group_size, bits = bits)


def make_quantized_model(hidden = 1024, inter = 2048):
    """Two eligible projections plus one too small to qualify."""
    import mlx.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp_gate = nn.Linear(hidden, inter, bias = False)
            self.mlp_down = nn.Linear(inter, hidden, bias = False)
            self.tiny = nn.Linear(64, 128, bias = False)

        def __call__(self, x):
            return self.mlp_down(self.mlp_gate(x))

    model = TinyModel()
    nn.quantize(model, group_size = 64, bits = 4)
    return model


def reset_int8_state():
    """Leave MLX unpatched and the allow-list empty."""
    from unsloth_zoo.mlx import int8_prefill
    from unsloth_zoo.mlx.int8_prefill import backends, capability

    int8_prefill.disable()
    capability.reset()
    backends.reset()
