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

"""CPU backend, PEFT LoRA Linear under bf16 / fp16 autocast: the input is no longer cast to the fp32 adapter dtype
(autocast casts it straight back for lora_A), with the same loss and gradients bit for bit. Outside autocast, with
LoRA dropout, or with a LoRA variant (DoRA) PEFT's cast stays."""

from __future__ import annotations

import copy

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from unsloth_zoo.cpu import ops  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self, width = 48):
        super().__init__()
        self.up = torch.nn.Linear(width, 2 * width, bias = False)
        self.down = torch.nn.Linear(2 * width, width, bias = False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class _InputUpcasts(TorchDispatchMode):
    """Counts fp32 copies of bf16 / fp16 activations with `rows` rows (the LoRA input casts)."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.count = rows, 0

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if func is torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.float32 \
                and args[0].dtype in (torch.bfloat16, torch.float16) and args[0].dim() == 2 \
                and args[0].shape[0] == self.rows:
            self.count += 1
        return func(*args, **kwargs)


def _model(dtype, dropout = 0.0, use_dora = False, seed = 0):
    torch.manual_seed(seed)
    net = _Net().to(dtype)
    model = get_peft_model(net, LoraConfig(r = 8, lora_alpha = 16, target_modules = ["up", "down"],
                                           lora_dropout = dropout, use_dora = use_dora))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.05)
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)
    return model


def _step(model, x, amp_dtype):
    model.zero_grad(set_to_none = True)
    torch.manual_seed(1)
    with torch.autocast("cpu", dtype = amp_dtype, enabled = amp_dtype is not None, cache_enabled = False):
        loss = model(x).pow(2).mean()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.fixture
def stock_cast():
    """PEFT's own cast for the reference arm, then the patch for the arm under test."""
    original = LoraLinear._cast_input_dtype
    stock = getattr(original, "__wrapped__", original)
    yield stock
    LoraLinear._cast_input_dtype = original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_no_input_upcast_and_same_bits_under_autocast(dtype, stock_cast):
    model = _model(dtype)
    x = torch.randn(32, 48).to(dtype)
    LoraLinear._cast_input_dtype = stock_cast
    with _InputUpcasts(32) as ref_casts:
        ref_loss, ref_grads = _step(model, x, dtype)
    assert ops.patch_peft_lora_autocast_input()
    assert ops.patch_peft_lora_autocast_input()  # idempotent
    with _InputUpcasts(32) as casts:
        loss, grads = _step(model, x, dtype)
    # PEFT: both LoRA inputs upcast in forward, and the down layer's input grad comes back through fp32 again.
    assert ref_casts.count == 3
    assert casts.count == 0
    assert torch.equal(loss, ref_loss)
    assert grads.keys() == ref_grads.keys() and len(grads) == 4
    for name in grads:
        assert torch.equal(grads[name], ref_grads[name]), name


@pytest.mark.parametrize("case", ["no_autocast", "dropout", "dora", "autocast_other_dtype"])
def test_cast_kept_where_autocast_would_not_undo_it(case, stock_cast):
    dtype = torch.bfloat16
    model = _model(dtype, dropout = 0.1 if case == "dropout" else 0.0, use_dora = case == "dora")
    x = torch.randn(32, 48).to(dtype)
    amp = {"no_autocast": None, "autocast_other_dtype": torch.float16}.get(case, dtype)
    LoraLinear._cast_input_dtype = stock_cast
    with _InputUpcasts(32) as ref_casts:
        ref_loss, ref_grads = _step(copy.deepcopy(model), x, amp)
    assert ops.patch_peft_lora_autocast_input()
    with _InputUpcasts(32) as casts:
        loss, grads = _step(model, x, amp)
    if case == "autocast_other_dtype":
        # The bf16 input into the first layer keeps its cast; under fp16 autocast the second layer's input is fp16
        # (the base layer ran in fp16), which autocast does round-trip exactly, so only that one is skipped.
        assert ref_casts.count == 3 and casts.count == 1
    else:
        assert casts.count == ref_casts.count >= 2
    assert torch.equal(loss, ref_loss)
    for name in grads:
        assert torch.equal(grads[name], ref_grads[name]), name


def test_installed_by_cpu_training_defaults(monkeypatch, stock_cast):
    from unsloth_zoo.cpu import trainer
    LoraLinear._cast_input_dtype = stock_cast
    trainer.install_cpu_training_defaults()
    assert getattr(LoraLinear._cast_input_dtype, "_unsloth_cpu", False)
