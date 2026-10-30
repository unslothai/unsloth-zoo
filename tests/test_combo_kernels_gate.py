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

"""Inductor combo kernels were off everywhere because they gave incompatible gradient sizes on torch 2.6. They are
now on from torch 2.10 for the compiled model modules (UNSLOTH_COMBO_KERNELS=0 / 1 overrides). CPU checks: the
gate, the compiler call site, and a compiled forward + backward with combo kernels on matching eager. CPU Inductor
emits C++, so this proves the scheduler path and the gradients' sizes, not the Triton combo kernels themselves."""

import inspect

import pytest
import torch


def _minor():
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


def test_gate_follows_torch_version_and_env(monkeypatch):
    from unsloth_zoo.temporary_patches import common
    monkeypatch.delenv("UNSLOTH_COMBO_KERNELS", raising = False)
    assert common.combo_kernels_supported() == (_minor() >= (2, 10))
    for version, expected in (((2, 6), False), ((2, 9), False), ((2, 10), True), ((2, 15), True)):
        monkeypatch.setattr(common, "_torch_major_minor", lambda v = version: v)
        assert common.combo_kernels_supported() is expected
        monkeypatch.setenv("UNSLOTH_COMBO_KERNELS", "0")
        assert common.combo_kernels_supported() is False
        monkeypatch.setenv("UNSLOTH_COMBO_KERNELS", "1")
        assert common.combo_kernels_supported() is True
        monkeypatch.delenv("UNSLOTH_COMBO_KERNELS")


def test_compiled_modules_use_the_gate():
    from unsloth_zoo import compiler
    source = inspect.getsource(compiler)
    assert "combo_kernels=combo_kernels_supported()" in source
    assert "combo_kernels=False" not in source


@pytest.mark.skipif(_minor() < (2, 10), reason = "combo kernels stay off before torch 2.10")
def test_combo_kernels_forward_backward_matches_eager():
    from torch._inductor.scheduler import Scheduler
    from unsloth_zoo.temporary_patches.common import get_torch_compile_options
    options = get_torch_compile_options(combo_kernels = True)
    assert options.get("combo_kernels") is True

    calls = []
    original = Scheduler.create_combo_kernel_nodes
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    def block(x, y, weight, bias):
        # Independent branches (combo candidates) plus a norm with a trainable weight (odd row count).
        a = torch.nn.functional.silu(x) * 2.0
        b = torch.tanh(y) + bias
        variance = x.float().pow(2).mean(-1, keepdim = True)
        normed = (x.float() * torch.rsqrt(variance + 1e-6)) * weight
        return a.sum() + b.square().mean() + normed.square().sum(), a, b

    torch.manual_seed(0)
    inputs = [torch.randn(37, 64, requires_grad = True), torch.randn(53, 48, requires_grad = True),
              torch.randn(64, requires_grad = True), torch.randn(48, requires_grad = True)]
    ref = block(*inputs)
    ref_grads = torch.autograd.grad(ref[0], inputs)

    torch._dynamo.reset()
    Scheduler.create_combo_kernel_nodes = counted
    import torch._inductor.config as inductor_config
    try:
        # A warm FX graph cache (another test compiled this graph) would skip the scheduler entirely.
        with inductor_config.patch(force_disable_caches = True):
            compiled = torch.compile(block, fullgraph = True, dynamic = False, options = options)
            act = compiled(*inputs)
            act_grads = torch.autograd.grad(act[0], inputs)
    finally:
        Scheduler.create_combo_kernel_nodes = original
        torch._dynamo.reset()
    assert calls, "the combo-kernel pass never ran"
    for r, a in zip(ref[1:], act[1:]):
        torch.testing.assert_close(a, r)
    for r, a in zip(ref_grads, act_grads):
        assert r.shape == a.shape
        torch.testing.assert_close(a, r, rtol = 1e-4, atol = 1e-5)
