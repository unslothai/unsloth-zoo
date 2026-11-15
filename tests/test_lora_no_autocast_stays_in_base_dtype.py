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

"""Rewritten PEFT LoRA Linear forward without autocast, bf16 base + fp32 adapters (FastModel 16-bit LoRA):
vanilla LoRA stays in bf16 (fp32 copies of x / result cost 1.4x step time, 5x memory); DoRA still gets fp32."""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from unsloth_zoo import compiler

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402


def _rewritten_forward():
    source = inspect.getsource(LoraLinear.forward)
    spaces = source.find("def")
    source = "\n".join(line[spaces:] for line in source.split("\n"))
    # Same order as patch_lora_forwards.
    source = compiler._patch_lora_update(source).replace("def forward", "def unsloth_forward", 1)
    source = compiler._patch_lora_input_cast(source, force_float32 = False)
    source = source.replace("self._check_forward_args(x, *args, **kwargs)", "")
    namespace = dict(vars(inspect.getmodule(LoraLinear)))
    exec(compiler.COMPILED_LORA_FORWARD + source, namespace)
    return namespace["unsloth_forward"]


def _layer(use_dora):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 48, bias = False)).to(torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r = 8, lora_alpha = 16, target_modules = ["0"], use_dora = use_dora))
    layer = model.base_model.model[0]
    assert layer.lora_A["default"].weight.dtype == torch.float32
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(0, 0.02)
    layer.forward = types.MethodType(_rewritten_forward(), layer)
    return layer


def _run(layer, x):
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, t.numel())) or t, lambda t: t):
        out = layer(x)
    out.float().sum().backward()
    return out, saved


def test_vanilla_lora_without_autocast_saves_no_fp32_activation():
    layer = _layer(use_dora = False)
    x = torch.randn(4, 16, 64, dtype = torch.bfloat16, requires_grad = True)
    out, saved = _run(layer, x)
    assert out.dtype == torch.bfloat16
    fp32_activations = [n for dtype, n in saved if dtype == torch.float32 and n >= x.numel()]
    assert not fp32_activations, f"fp32 activation copies saved for backward: {saved}"
    assert layer.lora_A["default"].weight.grad.dtype == torch.float32
    assert layer.lora_B["default"].weight.grad.abs().sum() > 0

    # Same numbers as the bf16 LoRA reference (what the autocast path computes).
    A = layer.lora_A["default"].weight.to(torch.bfloat16)
    B = layer.lora_B["default"].weight.to(torch.bfloat16)
    base = x.detach() @ layer.base_layer.weight.t()
    ref = base + layer.scaling["default"] * ((x.detach() @ A.t()) @ B.t())
    torch.testing.assert_close(out.detach(), ref, rtol = 1.6e-2, atol = 1e-2)


def test_dora_without_autocast_still_runs_in_adapter_dtype():
    layer = _layer(use_dora = True)
    x = torch.randn(2, 8, 64, dtype = torch.bfloat16, requires_grad = True)
    out, _ = _run(layer, x)
    assert out.dtype == torch.bfloat16
    assert layer.lora_A["default"].weight.grad is not None
