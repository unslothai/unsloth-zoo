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

"""
PR-A integration: exercise unsloth_zoo.mlx.loader._dequantize_selected_mlx_modules.

Builds a synthetic MLX-style model with one QuantizedLinear submodule,
runs PR-A's dequantize-and-replace helper, verifies the result is
a numerically correct nn.Linear with the dequantized weight.

This is the canonical PR-A code path: load_in_4bit=False (or
selective requantize) walks named_modules, finds QuantizedLinear,
calls mx.dequantize with mode='affine', and swaps in nn.Linear.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _make_packed_4bit(values, group_size=8, bits=4):
    """Pack a flat int list of 4-bit values into uint32 words.

    Returns (packed: int32 tensor of shape (1, n_words),
             scales: tensor (1, n_groups), biases: tensor (1, n_groups)).
    """
    elements_per_word = 32 // bits
    n = len(values)
    assert n % elements_per_word == 0, "values must align to packing boundary"
    n_words = n // elements_per_word
    n_groups = n // group_size

    packed = []
    for w in range(n_words):
        word = 0
        for i in range(elements_per_word):
            word |= (values[w * elements_per_word + i] & 0xF) << (i * bits)
        packed.append(word)

    return (
        torch.tensor([packed], dtype=torch.int32),
        torch.ones((1, n_groups), dtype=torch.float32),  # scale = 1
        torch.zeros((1, n_groups), dtype=torch.float32),  # bias  = 0
    )


def test_dequantize_selected_mlx_modules_swap():
    """Build a Module with a QuantizedLinear, dequant-replace, verify swap."""
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import _dequantize_selected_mlx_modules

    # 4-bit weight: 8 input dims (one group), 2 output dims, group_size=8.
    bits, group_size = 4, 8
    in_features, out_features = 8, 2

    # Output row 0 = [0,1,2,3,4,5,6,7], output row 1 = [7,6,5,4,3,2,1,0]
    values_row0 = [0, 1, 2, 3, 4, 5, 6, 7]
    values_row1 = [7, 6, 5, 4, 3, 2, 1, 0]

    packed_row0, scales0, biases0 = _make_packed_4bit(values_row0, group_size, bits)
    packed_row1, scales1, biases1 = _make_packed_4bit(values_row1, group_size, bits)
    packed = torch.cat([packed_row0, packed_row1], dim=0)  # (2, 1)
    scales = torch.cat([scales0, scales1], dim=0)          # (2, 1)
    biases = torch.cat([biases0, biases1], dim=0)          # (2, 1)

    # Build a wrapper Module with a QuantizedLinear submodule.
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.QuantizedLinear(
                in_features, out_features, bias=False,
                group_size=group_size, bits=bits, mode="affine",
            )
            self.layer.weight = packed
            self.layer.scales = scales
            self.layer.biases = biases

    model = TinyModel()

    # Sanity: named_modules walks self + the QuantizedLinear child.
    names = [n for n, _ in model.named_modules()]
    assert "" in names and "layer" in names, f"unexpected names: {names!r}"

    # Run the dequant-replace helper.
    n_replaced = _dequantize_selected_mlx_modules(
        model, predicate=lambda path, mod: isinstance(mod, nn.QuantizedLinear)
    )
    assert n_replaced == 1, f"expected 1 replacement, got {n_replaced}"

    # The QuantizedLinear has been swapped for an nn.Linear.
    assert isinstance(model.layer, nn.Linear), f"got {type(model.layer)!r}"
    assert not isinstance(model.layer, nn.QuantizedLinear)

    # Weight values: row 0 = [0..7], row 1 = [7..0]
    expected_row0 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
    expected_row1 = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.float32)
    torch.testing.assert_close(model.layer.weight[0].float(), expected_row0)
    torch.testing.assert_close(model.layer.weight[1].float(), expected_row1)


def test_dequantize_predicate_filters():
    """Predicate should let some QuantizedLinear modules through unchanged."""
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import _dequantize_selected_mlx_modules

    bits, group_size = 4, 8
    packed, scales, biases = _make_packed_4bit([0]*8, group_size, bits)

    class TwoLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.keep = nn.QuantizedLinear(8, 1, bias=False, group_size=group_size, bits=bits)
            self.keep.weight, self.keep.scales, self.keep.biases = packed, scales, biases
            self.swap = nn.QuantizedLinear(8, 1, bias=False, group_size=group_size, bits=bits)
            self.swap.weight, self.swap.scales, self.swap.biases = packed, scales, biases

    m = TwoLayers()
    # Only swap the .swap submodule
    n = _dequantize_selected_mlx_modules(m, predicate=lambda path, mod: path == "swap")
    assert n == 1
    assert isinstance(m.swap, nn.Linear)
    assert isinstance(m.keep, nn.QuantizedLinear)


def test_dequantize_no_match_returns_zero():
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import _dequantize_selected_mlx_modules

    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)  # NOT quantized

    n = _dequantize_selected_mlx_modules(Empty(), predicate=lambda p, m: True)
    assert n == 0
