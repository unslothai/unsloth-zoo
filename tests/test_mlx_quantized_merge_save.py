# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Quantized-state detection for the merged_4bit save path.

`save_method="merged_4bit"` used to be a silent no-op on an unquantized model:
`LoRALinear.fuse(dequantize=False)` only requantizes when the base was already
quantized, so a model loaded with `load_in_16bit=True` or `full_finetuning=True`
was written at full precision with no warning.

Deciding whether to quantize hinges entirely on "is anything here quantized",
so that predicate is enumerated here across every MLX quantized module type and
grid rather than only the affine-4bit one the default loader happens to produce.
"""

import importlib
import sys

import pytest


def _real_mlx_runtime():
    try:
        importlib.import_module("mlx_lm.tuner.lora")
    except Exception:
        return False
    origin = getattr(sys.modules.get("mlx.core"), "__file__", "") or ""
    return "mlx_simulation" not in origin


if not _real_mlx_runtime():
    pytest.skip("needs the real mlx runtime", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn

from unsloth_zoo.mlx.utils import _model_has_quantized_module

DIMS = 256


class _Stack(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)


def _linear():
    return nn.Linear(DIMS, DIMS, bias=False)


def test_unquantized_model_is_detected_as_unquantized():
    assert not _model_has_quantized_module(_Stack(_linear(), _linear()))


@pytest.mark.parametrize(
    "mode,group_size,bits",
    [
        ("affine", 64, 4),
        ("affine", 64, 8),
        ("affine", 32, 4),
        ("mxfp4", 32, 4),
    ],
)
def test_every_quantized_grid_is_detected(mode, group_size, bits):
    """The decision must not depend on the grid, only on 'is it quantized'.

    Enumerated rather than assumed: the default loader produces affine 4-bit,
    so testing only that would leave load_in_8bit / load_in_mxfp4 models silently
    re-quantized on save.
    """
    quantized = nn.QuantizedLinear.from_linear(
        _linear(), group_size=group_size, bits=bits, mode=mode)
    assert _model_has_quantized_module(_Stack(quantized, _linear()))


def test_quantized_embedding_is_detected():
    embedding = nn.QuantizedEmbedding.from_embedding(
        nn.Embedding(512, DIMS), group_size=64, bits=4)
    assert _model_has_quantized_module(_Stack(embedding))


def test_partially_quantized_model_counts_as_quantized():
    """A normally-loaded 4-bit model *is* partial.

    The loader's predicate skips `embed_tokens` / `lm_head`, so a model loaded
    with load_in_4bit=True has unquantized modules in it. Treating "some
    unquantized modules" as "needs quantizing" would re-quantize an
    already-quantized checkpoint.
    """
    quantized = nn.QuantizedLinear.from_linear(
        _linear(), group_size=64, bits=4, mode="affine")
    model = _Stack(quantized, _linear(), nn.Embedding(512, DIMS))
    assert _model_has_quantized_module(model)


def test_switch_linear_experts_are_detected():
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    experts = switch_layers.SwitchLinear(DIMS, DIMS, num_experts=2, bias=False)
    experts = experts.to_quantized(group_size=64, bits=4, mode="affine")
    assert _model_has_quantized_module(_Stack(experts))
