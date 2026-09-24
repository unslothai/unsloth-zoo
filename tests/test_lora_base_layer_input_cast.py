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

"""The outside-autocast cast of x to the LoRA base layer's weight dtype.

It exists for float32 base weights under a float16 activation (SiGLIP). An FP8 base layer
(transformers FP8Linear after Unsloth's patch, or any Linear holding a float8 weight) quantizes
the activation itself, so casting x to float8 made the wrapped layer return float8 and a plain
no-grad forward after get_peft_model failed with "Promotion for Float8 Types is not supported".
"""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.compiler import _patch_lora_base_layer_input_cast, patch_lora_forwards

FORWARD = """\
def unsloth_forward(self, x, *args, **kwargs):
    if self.disable_adapters:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
    return result
"""

_FP8 = getattr(torch, "float8_e4m3fn", None)


class _Base:
    def __init__(self, weight):
        self.weight = weight
        self.seen = []

    def __call__(self, x, *args, **kwargs):
        self.seen.append(x.dtype)
        return x


def _run(weight, x_dtype = torch.bfloat16):
    namespace = {"torch": torch}
    exec(textwrap.dedent(_patch_lora_base_layer_input_cast(FORWARD)), namespace)
    base = _Base(weight)
    layer = SimpleNamespace(disable_adapters = False, base_layer = base)
    x = torch.zeros(2, 4, dtype = x_dtype)
    namespace["unsloth_forward"](layer, x)
    return base.seen


def test_every_base_layer_call_is_guarded_once():
    rewritten = _patch_lora_base_layer_input_cast(FORWARD)
    assert rewritten.count("x = x.to(self.base_layer.weight.dtype)") == 2
    assert rewritten.count("element_size() > 1") == 2
    assert _patch_lora_base_layer_input_cast("def f(x):\n    return x\n") == "def f(x):\n    return x\n"


@pytest.mark.skipif(_FP8 is None, reason = "no float8 dtype in this torch")
def test_fp8_base_weight_keeps_the_activation_dtype():
    assert _run(torch.zeros(4, 4, dtype = _FP8)) == [torch.bfloat16]


def test_float32_base_weight_still_casts_the_activation():
    """Negative arm: the SiGLIP case the cast was written for is unchanged."""
    assert _run(torch.zeros(4, 4, dtype = torch.float32), x_dtype = torch.float16) == [torch.float32]
    assert _run(torch.zeros(4, 4, dtype = torch.bfloat16), x_dtype = torch.float32) == [torch.bfloat16]


def test_packed_or_integer_weights_never_become_the_activation_dtype():
    assert _run(torch.zeros(4, 4, dtype = torch.uint8)) == [torch.bfloat16]
    assert _run(torch.zeros(4, 4, dtype = torch.int8)) == [torch.bfloat16]
    four_bit = torch.zeros(4, 4, dtype = torch.uint8)
    four_bit.quant_state = object()
    assert _run(four_bit) == [torch.bfloat16]


def test_patch_lora_forwards_routes_through_the_helper():
    assert "_patch_lora_base_layer_input_cast(source)" in inspect.getsource(patch_lora_forwards)
