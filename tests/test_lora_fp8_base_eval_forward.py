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

"""Generated LoRA forward over an FP8 base: no-autocast eval must not cast x to float8
(RMSNorm "Promotion for Float8 Types is not supported" on FP8 MoE checkpoints)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

_FP8 = getattr(torch, "float8_e4m3fn", None)
pytestmark = pytest.mark.skipif(_FP8 is None, reason = "no float8 dtype in this torch")

peft = pytest.importorskip("peft")


class _BlockFP8Linear(torch.nn.Linear):
    """Like FP8Linear / FbgemmFp8Linear: returns the activation dtype, float8 in -> float8 out."""

    def __init__(self, in_features, out_features, generator):
        super().__init__(in_features, out_features, bias = False)
        w = torch.randn(out_features, in_features, generator = generator) * 0.05
        scale = w.abs().amax() / 448.0
        self.weight = torch.nn.Parameter((w / scale).to(_FP8), requires_grad = False)
        self.register_buffer("weight_scale_inv", scale.reshape(1).float())

    def dequantized(self, dtype):
        return (self.weight.float() * self.weight_scale_inv).to(dtype)

    def forward(self, x):
        out = x.float() @ self.dequantized(torch.float32).t()
        return out.to(x.dtype)


class _SwiGLU(torch.nn.Module):
    def __init__(self, hidden, inter, seed = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_proj = _BlockFP8Linear(hidden, inter, g)
        self.up_proj = _BlockFP8Linear(hidden, inter, g)
        self.down_proj = _BlockFP8Linear(inter, hidden, g)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _reference(weights, x):
    gate = x @ weights["gate_proj"].to(x.dtype).t()
    up = x @ weights["up_proj"].to(x.dtype).t()
    return (F.silu(gate) * up) @ weights["down_proj"].to(x.dtype).t()


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


def _peft_block(hidden = 32, inter = 64):
    from peft import LoraConfig, get_peft_model

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _SwiGLU(hidden, inter)

        def forward(self, x):
            return self.mlp(x)

    base = _Wrap()
    weights = {
        name: getattr(base.mlp, name).dequantized(torch.float32)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    model = get_peft_model(base, LoraConfig(
        r = 4, lora_alpha = 4, lora_dropout = 0.0, target_modules = ["gate_proj", "up_proj", "down_proj"],
    ))
    # Unsloth keeps adapters in float32; older peft copies a float base dtype (float8 here).
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
    return model, weights


def test_lora_layer_is_the_generated_forward(unsloth_lora_forwards):
    model, _ = _peft_block()
    layer = model.base_model.model.mlp.gate_proj
    assert type(layer).forward.__name__ == "unsloth_forward"
    assert layer.base_layer.weight.dtype == _FP8


@pytest.mark.parametrize("training", [False, True])
def test_no_grad_forward_without_autocast_keeps_bf16(unsloth_lora_forwards, training):
    model, weights = _peft_block()
    model.train(training)
    x = torch.randn(3, 5, 32, generator = torch.Generator().manual_seed(1)).to(torch.bfloat16)
    assert not torch.is_autocast_enabled()
    with torch.no_grad():
        gate = model.base_model.model.mlp.gate_proj(x)
        out = model(x)
    assert gate.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, _reference(weights, x), rtol = 2e-2, atol = 5e-3)


_AUTOCAST_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _AUTOCAST_DEVICES)
def test_autocast_forward_keeps_bf16(unsloth_lora_forwards, device):
    """is_autocast_enabled() reports CUDA only, so CPU autocast needs the dtype clauses."""
    model, weights = _peft_block()
    model.to(device)
    x = torch.randn(3, 5, 32, generator = torch.Generator().manual_seed(2)).to(device, torch.bfloat16)
    with torch.autocast(device, dtype = torch.bfloat16), torch.no_grad():
        out = model(x)
    assert out.dtype == torch.bfloat16
    ref = _reference({k: v.to(device) for k, v in weights.items()}, x)
    torch.testing.assert_close(out, ref, rtol = 2e-2, atol = 5e-3)


def test_lora_delta_is_applied_after_the_fp8_base(unsloth_lora_forwards):
    model, weights = _peft_block()
    layer = model.base_model.model.mlp.gate_proj
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_(generator = torch.Generator().manual_seed(3))
    x = torch.randn(2, 32, generator = torch.Generator().manual_seed(4)).to(torch.bfloat16)
    with torch.no_grad():
        got = layer(x)
    want = x @ weights["gate_proj"].to(torch.bfloat16).t()
    want = want + (x.float() @ layer.lora_A["default"].weight.t() @ layer.lora_B["default"].weight.t()).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want, rtol = 2e-2, atol = 2e-2)


def test_float32_base_under_half_activation_still_casts(unsloth_lora_forwards):
    from peft import LoraConfig, get_peft_model

    seen = []

    class _F32(torch.nn.Linear):
        def forward(self, x):
            seen.append(x.dtype)
            return super().forward(x)

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _F32(8, 8)

        def forward(self, x):
            return self.proj(x)

    model = get_peft_model(_Wrap(), LoraConfig(r = 2, target_modules = ["proj"]))
    with torch.no_grad():
        out = model(torch.randn(2, 8).to(torch.float16))
    assert seen == [torch.float32]
    assert out.dtype == torch.float32
