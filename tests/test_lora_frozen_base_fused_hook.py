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


"""The compiled PEFT lora.Linear forward calls a frozen base's `_unsloth_fused_lora_forward` (set by Unsloth on
fp8 / int8 LoRA bases) first, and runs PEFT's forward unchanged when it is absent or returns None."""

import inspect

import pytest
import torch

peft = pytest.importorskip("peft")


@pytest.fixture
def unsloth_lora_forwards(tmp_path, monkeypatch):
    from unsloth_zoo import compiler

    monkeypatch.chdir(tmp_path)
    saved = [(cls, cls.__dict__.get("forward")) for cls, _, _ in compiler.get_lora_layer_modules()]
    compiler.patch_lora_forwards({})
    try:
        yield
    finally:
        for cls, forward in saved:
            if forward is not None:
                cls.forward = forward


def _layer():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16, bias = False))
    model = peft.get_peft_model(model, peft.LoraConfig(r = 4, target_modules = ["0"]))
    layer = model.base_model.model[0]
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_()
    return layer


def test_hook_is_inserted_once_and_compiles():
    from unsloth_zoo.compiler import _patch_lora_frozen_base_fused

    source = inspect.getsource(peft.tuners.lora.layer.Linear.forward)
    source = "\n".join(line[4:] for line in source.split("\n")).replace("def forward", "def unsloth_forward", 1)
    patched = _patch_lora_frozen_base_fused(source)
    assert patched.count("_unsloth_fused_lora_forward") == 1
    assert _patch_lora_frozen_base_fused(patched) == patched
    compile(patched, "<lora>", "exec")
    # A forward without *args / **kwargs to forward is left alone.
    plain = "def unsloth_forward(self, x):\n    return self.base_layer(x)\n"
    assert _patch_lora_frozen_base_fused(plain) == plain


def test_fused_hook_runs_first_and_falls_back(unsloth_lora_forwards):
    layer = _layer()
    x = torch.randn(3, 32)
    expected = layer.base_layer(x) + layer.scaling["default"] * layer.lora_B["default"](layer.lora_A["default"](x))
    assert torch.allclose(layer(x), expected, atol = 1e-5)  # no hook: PEFT's forward

    seen = []
    sentinel = torch.full((3, 16), 7.0)

    def fused(proj, x, args, kwargs):
        seen.append((proj, args, dict(kwargs)))
        return sentinel

    layer.base_layer.__dict__["_unsloth_fused_lora_forward"] = fused
    assert layer(x) is sentinel and seen == [(layer, (), {})]
    layer.base_layer.__dict__["_unsloth_fused_lora_forward"] = lambda proj, x, args, kwargs: None
    assert torch.allclose(layer(x), expected, atol = 1e-5)  # hook declined: PEFT's forward
