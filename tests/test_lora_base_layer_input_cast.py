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

"""LoRA base-layer input cast: SiGLIP fp32 weights keep it, FP8 weights must skip it."""

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
    assert _run(torch.zeros(4, 4, dtype = torch.float32), x_dtype = torch.float16) == [torch.float32]
    assert _run(torch.zeros(4, 4, dtype = torch.bfloat16), x_dtype = torch.float32) == [torch.bfloat16]


def test_packed_or_integer_weights_never_become_the_activation_dtype():
    assert _run(torch.zeros(4, 4, dtype = torch.uint8)) == [torch.bfloat16]
    assert _run(torch.zeros(4, 4, dtype = torch.int8)) == [torch.bfloat16]
    four_bit = torch.zeros(4, 4, dtype = torch.uint8)
    four_bit.quant_state = object()
    assert _run(four_bit) == [torch.bfloat16]


class _PackedBase(_Base):
    """A packed base (MXFP4 / GGUF): `weight` decodes the whole matrix on every read, so it must never be read."""

    _unsloth_mxfp4_packed_linear = True

    def __init__(self):
        self.seen, self.reads = [], 0

    @property
    def weight(self):
        self.reads += 1
        return torch.zeros(4, 4, dtype = torch.float32)


def test_packed_base_weight_is_never_read():
    namespace = {"torch": torch}
    exec(textwrap.dedent(_patch_lora_base_layer_input_cast(FORWARD)), namespace)
    for disable in (False, True):
        base = _PackedBase()
        layer = SimpleNamespace(disable_adapters = disable, base_layer = base)
        namespace["unsloth_forward"](layer, torch.zeros(2, 4, dtype = torch.bfloat16))
        assert base.reads == 0
        assert base.seen == [torch.bfloat16]


def test_packed_base_weight_is_not_read_under_torch_compile():
    namespace = {"torch": torch}
    exec(textwrap.dedent(_patch_lora_base_layer_input_cast(FORWARD)), namespace)
    base = _PackedBase()
    layer = SimpleNamespace(disable_adapters = False, base_layer = base)
    compiled = torch.compile(namespace["unsloth_forward"], backend = "eager", fullgraph = True)
    for _ in range(3):
        compiled(layer, torch.zeros(2, 4, dtype = torch.bfloat16))
    assert base.reads == 0


def test_patch_lora_forwards_routes_through_the_helper():
    assert "_patch_lora_base_layer_input_cast(source)" in inspect.getsource(patch_lora_forwards)


class _PackedBaseWithComputeDtype(_PackedBase):
    """A real packed base computes in x's dtype outside autocast, so x must arrive in its `compute_dtype` (the dtype
    its decoded weight has) without `weight` being read."""

    def __init__(self, compute_dtype):
        super().__init__()
        self.compute_dtype = compute_dtype


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize(
    "compute_dtype, x_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_packed_base_gets_its_compute_dtype_without_a_weight_read(disable, compute_dtype, x_dtype):
    namespace = {"torch": torch}
    exec(textwrap.dedent(_patch_lora_base_layer_input_cast(FORWARD)), namespace)
    base = _PackedBaseWithComputeDtype(compute_dtype)
    layer = SimpleNamespace(disable_adapters = disable, base_layer = base)
    namespace["unsloth_forward"](layer, torch.zeros(2, 4, dtype = x_dtype))
    assert base.seen == [compute_dtype]
    assert base.reads == 0


def test_packed_base_compute_dtype_cast_under_torch_compile():
    namespace = {"torch": torch}
    exec(textwrap.dedent(_patch_lora_base_layer_input_cast(FORWARD)), namespace)
    base = _PackedBaseWithComputeDtype(torch.bfloat16)
    layer = SimpleNamespace(disable_adapters = False, base_layer = base)
    compiled = torch.compile(namespace["unsloth_forward"], backend = "eager", fullgraph = True)
    for _ in range(3):
        out = compiled(layer, torch.ones(2, 4, dtype = torch.float32))
        assert out.dtype == torch.bfloat16
    assert base.reads == 0
