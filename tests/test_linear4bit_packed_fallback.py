# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""The Linear4bit fallback never hands a packed 4-bit weight to F.linear.

Qwen3.5's GatedDeltaNet projections can reach the patched forward with a
weight that is still the packed uint8 nf4 buffer and no ``quant_state`` on the
parameter (unsloth#9867). The old fallback ran ``F.linear`` on that buffer and
died with ``mat1 and mat2 shapes cannot be multiplied (BxH and 1xNUMEL)``. The
patched fallback must instead re-derive the quant state from the module, and
when that is impossible, fail with an error naming the real problem.

bitsandbytes is faked (house pattern: test_broken_bitsandbytes_import.py), so
this runs on CPU-only hosts and pins the control flow, not bnb's kernels.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _install_fake_bitsandbytes(monkeypatch, *, fix_attaches_state):
    calls = {"fix": 0, "matmul": 0}

    class _QuantState:
        pass

    class _Params4bit(torch.nn.Parameter):
        pass

    def fix_4bit_weight_quant_state_from_module(module):
        calls["fix"] += 1
        if fix_attaches_state:
            module.weight.quant_state = _QuantState()

    def matmul_4bit(x, weight, bias = None, quant_state = None):
        calls["matmul"] += 1
        # Shape-checked stand-in: the real kernel dequantizes to (out, in).
        return x.new_zeros(x.shape[0], 6144)

    bnb = types.ModuleType("bitsandbytes")
    bnb.matmul_4bit = matmul_4bit
    bnb.nn = types.ModuleType("bitsandbytes.nn")
    bnb.nn.modules = types.ModuleType("bitsandbytes.nn.modules")

    class Linear4bit(torch.nn.Module):
        # patch_function replaces an existing forward; signature must match.
        def forward(self, x: torch.Tensor):
            raise NotImplementedError

    bnb.nn.modules.Linear4bit = Linear4bit
    bnb.nn.Linear4bit = Linear4bit
    bnb.nn.modules.Params4bit = _Params4bit
    bnb.nn.modules.fix_4bit_weight_quant_state_from_module = (
        fix_4bit_weight_quant_state_from_module
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", bnb.nn)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn.modules", bnb.nn.modules)
    return bnb, Linear4bit, calls


def _patched_forward(monkeypatch, *, fix_attaches_state):
    bnb, Linear4bit, calls = _install_fake_bitsandbytes(
        monkeypatch, fix_attaches_state = fix_attaches_state
    )
    from unsloth_zoo.temporary_patches import bitsandbytes as patch_module

    patch_module.patch_bitsandbytes_linear4bit_forward()
    return Linear4bit, calls


def _packed_module(Linear4bit):
    # The reporter's in_proj_z: a (6144, 5120) nf4 weight packs to
    # (6144 * 5120 / 2, 1) uint8 — F.linear on it is the mat1/mat2 error.
    module = Linear4bit()
    module.weight = torch.nn.Parameter(
        torch.zeros(6144 * 5120 // 2, 1, dtype = torch.uint8), requires_grad = False
    )
    module.bias = None
    module.compute_type_is_set = True
    module.compute_dtype = torch.float16
    return module


def test_packed_weight_recovers_its_quant_state(monkeypatch):
    Linear4bit, calls = _patched_forward(monkeypatch, fix_attaches_state = True)
    module = _packed_module(Linear4bit)
    # shape[-1] == 1 also triggers the init-time probe, so the packed branch is
    # exercised with a wide packed layout too, which that probe does not cover.
    module.weight.data = module.weight.data.reshape(1, -1)
    x = torch.zeros(82, 5120, dtype = torch.float16)
    out = module.forward(x)
    assert calls["fix"] >= 1
    assert calls["matmul"] == 1, "recovered state must route through matmul_4bit"
    assert out.shape == (82, 6144)


def test_unrecoverable_packed_weight_fails_with_a_named_error(monkeypatch):
    Linear4bit, calls = _patched_forward(monkeypatch, fix_attaches_state = False)
    module = _packed_module(Linear4bit)
    module.weight.data = module.weight.data.reshape(1, -1)
    x = torch.zeros(82, 5120, dtype = torch.float16)
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(x)
    message = str(excinfo.value)
    assert "packed uint8 weight" in message
    assert "quant_state" in message
    assert "mat1" not in message


def test_a_genuinely_unquantized_layer_still_takes_f_linear(monkeypatch):
    Linear4bit, calls = _patched_forward(monkeypatch, fix_attaches_state = False)
    module = Linear4bit()
    module.weight = torch.nn.Parameter(torch.zeros(8, 4, dtype = torch.float32))
    module.bias = None
    x = torch.zeros(2, 4, dtype = torch.float32)
    out = module.forward(x)
    assert out.shape == (2, 8)
    assert calls["matmul"] == 0
