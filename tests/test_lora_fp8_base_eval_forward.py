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

"""The real generated PEFT LoRA forward over an FP8 base layer, outside autocast.

FP8 MoE checkpoints (Qwen3-30B-A3B-FP8, Qwen3.5-122B-A10B-FP8, Mistral-Small-4 FP8) keep
their attention and shared-expert projections as FP8 linears that quantize the activation
themselves. After get_peft_model, a plain `model.eval()` + `torch.no_grad()` forward (an eval
loss, or trainer.evaluate() before training) runs without autocast; the generated LoRA forward
cast x to the float8 weight dtype, the projection returned float8, and the next op failed:
RMSNorm with "Promotion for Float8 Types is not supported", the shared-expert SiLU with
'"silu_cuda" not implemented for Float8_e4m3fn'. Inside trainer.train() accelerate wraps
forward in autocast, which skips the cast, so the same eval worked only after training.

These tests run patch_lora_forwards on the installed peft, wrap a small FP8 layer, and drive
the gate / up / down projections of a SwiGLU block with and without autocast.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

_FP8 = getattr(torch, "float8_e4m3fn", None)
pytestmark = pytest.mark.skipif(_FP8 is None, reason = "no float8 dtype in this torch")

peft = pytest.importorskip("peft")


class _BlockFP8Linear(torch.nn.Linear):
    """A frozen per-tensor FP8 linear that quantizes nothing and dequantizes its own weight.

    Like transformers FP8Linear under Unsloth's patch (and FbgemmFp8Linear), it accepts a
    16/32-bit activation and returns the activation dtype. Fed a float8 activation it returns
    float8, which is what the pre-fix LoRA input cast produced.
    """

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
    """bf16 math on the dequantized weights; zero-init LoRA adds nothing."""
    gate = x @ weights["gate_proj"].to(x.dtype).t()
    up = x @ weights["up_proj"].to(x.dtype).t()
    return (F.silu(gate) * up) @ weights["down_proj"].to(x.dtype).t()


@pytest.fixture
def unsloth_lora_forwards(tmp_path, monkeypatch):
    """Install the generated LoRA forwards, then put peft's own forwards back."""
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
    """The eval-before-train forward: no autocast, grad off, either module mode."""
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
    """The training / generate forward runs under autocast.

    On CUDA the guard's `torch.is_autocast_enabled()` already skipped the cast, so this arm is
    unchanged. That call reports CUDA autocast only, so under CPU autocast the old guard still
    cast x to float8; the dtype clauses cover that too.
    """
    model, weights = _peft_block()
    model.to(device)
    x = torch.randn(3, 5, 32, generator = torch.Generator().manual_seed(2)).to(device, torch.bfloat16)
    with torch.autocast(device, dtype = torch.bfloat16), torch.no_grad():
        out = model(x)
    assert out.dtype == torch.bfloat16
    ref = _reference({k: v.to(device) for k, v in weights.items()}, x)
    torch.testing.assert_close(out, ref, rtol = 2e-2, atol = 5e-3)


def test_lora_delta_is_applied_after_the_fp8_base(unsloth_lora_forwards):
    """A nonzero adapter still adds its delta on top of the bf16 base output."""
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
    """Negative arm: the SiGLIP fp32 base / fp16 activation case keeps its cast."""
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
