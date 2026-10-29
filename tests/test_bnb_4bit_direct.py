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

"""CPU: cgemm_4bit_* args match bitsandbytes' _gemm_4bit_kernel_impl. CUDA: outputs match bitwise."""
import contextlib
from types import SimpleNamespace

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
# Not importorskip: on a CPU-only torch this module raises AttributeError at import
# (torch._C has no _cuda_getCurrentRawStream), which importorskip lets through as a collection error.
try:
    from bitsandbytes.backends.cuda import ops as bops
except Exception as exc:
    pytest.skip(f"bitsandbytes CUDA ops do not import here: {exc!r}", allow_module_level = True)
if not hasattr(bops, "_gemm_4bit_kernel_impl"):
    pytest.skip("bitsandbytes without the fused 4-bit GEMM", allow_module_level = True)

from bitsandbytes.functional import QuantState  # noqa: E402
from unsloth_zoo.temporary_patches import bitsandbytes as Z  # noqa: E402

_STREAM = 7


class _Lib:
    compiled_with_cuda = True

    def __init__(self):
        self.calls = []
        for d in ("bf16", "fp16", "fp32"):
            setattr(self, f"cgemm_4bit_{d}", self._rec(d))

    def _rec(self, d):
        def fn(*args):
            self.calls.append((d, args))
        fn.argtypes = [None] * 14  # bitsandbytes' _setup_ctypes
        return fn


@pytest.fixture
def lib(monkeypatch):
    fake = _Lib()
    monkeypatch.setattr(bops, "lib", fake)
    monkeypatch.setattr(bops, "_get_raw_stream", lambda i: _STREAM)
    monkeypatch.setattr(bops, "_cuda_device_of", lambda a: contextlib.nullcontext())
    monkeypatch.setattr(bops, "_gemm_4bit_use_custom_fn", lambda *a: True)
    monkeypatch.setattr(Z, "_cuda_raw_stream", lambda i: _STREAM)
    monkeypatch.setattr(Z, "_on_current_cuda", lambda x: True)
    monkeypatch.setitem(Z._BNB_DIRECT, "ops", None)
    monkeypatch.setitem(Z._BNB_DIRECT, "on", True)
    monkeypatch.setitem(Z._BNB_DIRECT, "fn", {})
    monkeypatch.setitem(Z._BNB_DIRECT, "checked", {
        (d, n, b, q): True for d in (torch.bfloat16, torch.float16, torch.float32) for n in (0, 1) for b in (0, 1)
        for q in (1, 2)})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(Z, "BNB_4BIT_DIRECT_CALLS", {"direct": 0})
    monkeypatch.setattr(torch.ops.bitsandbytes, "gemm_4bit", SimpleNamespace(default = bops._gemm_4bit_kernel_impl))
    return fake


def _layer(N = 64, K = 128, dtype = torch.bfloat16, nested = False, bias = True, blocksize = 64):
    g = torch.Generator().manual_seed(N + K)
    weight = torch.randint(0, 255, (N * K // 2, 1), dtype = torch.uint8, generator = g)
    absmax = torch.rand(N * K // blocksize, generator = g)
    code = torch.linspace(-1, 1, 16)
    if nested:
        state2 = QuantState(absmax = torch.rand(N * K // blocksize // 256 + 1, generator = g), code = torch.linspace(-1, 1, 256),
                            blocksize = 256, quant_type = "nf4", dtype = torch.float32)
        qs = QuantState(absmax = torch.randint(0, 255, (N * K // blocksize,), dtype = torch.uint8, generator = g),
                        shape = torch.Size([N, K]), code = code, blocksize = blocksize, quant_type = "nf4", dtype = dtype,
                        offset = torch.tensor(0.25), state2 = state2)
    else:
        qs = QuantState(absmax = absmax, shape = torch.Size([N, K]), code = code, blocksize = blocksize,
                        quant_type = "nf4", dtype = dtype)
    layer = SimpleNamespace(weight = SimpleNamespace(data = weight, quant_state = qs, dtype = torch.uint8,
                                                     data_ptr = weight.data_ptr, get_device = weight.get_device),
                            bias = torch.randn(N, generator = g).to(dtype) if bias else None,
                            compute_dtype = dtype, compute_type_is_set = True)
    return layer, qs


def _strip_out(call, bias_copy = False):
    d, args = call
    return d, args[:6] + (None, None if bias_copy else args[7]) + args[8:]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("bias", [True, False, "fp32"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [1, 4])
def test_same_kernel_call_as_bitsandbytes(lib, nested, bias, dtype, M):
    layer, qs = _layer(dtype = dtype, nested = nested, bias = bool(bias))
    if bias == "fp32":
        layer.bias = layer.bias.float()  # nn.Linear's default; the forward casts it to the compute dtype
    x = torch.randn(1, M, 128).to(dtype)
    with torch.no_grad():
        ref_bias = None if layer.bias is None else layer.bias.to(dtype)
        ref = bnb.matmul_4bit(x, layer.weight.data.t(), bias = ref_bias, quant_state = qs)
        out = Z._bnb_4bit_direct(layer, x, qs)
    assert out is not None and out.shape == ref.shape and out.dtype == ref.dtype
    assert len(lib.calls) == 2
    assert _strip_out(lib.calls[0], bias == "fp32") == _strip_out(lib.calls[1], bias == "fp32")
    assert (lib.calls[1][1][7] is None) == (not bias)
    assert Z.BNB_4BIT_DIRECT_CALLS["direct"] == 1


def test_rebuilt_when_quant_tensors_change(lib):
    layer, qs = _layer(nested = True)
    x = torch.randn(1, 1, 128).to(torch.bfloat16)
    with torch.no_grad():
        Z._bnb_4bit_direct(layer, x, qs)
        plan = layer.__dict__["_unsloth_bnb_direct"]
        Z._bnb_4bit_direct(layer, x, qs)
        assert layer.__dict__["_unsloth_bnb_direct"] is plan
        # QuantState.to() swaps tensors on the same object; a recast offset too.
        for name in ("absmax", "offset"):
            setattr(qs, name, getattr(qs, name).clone())
            ref = bnb.matmul_4bit(x, layer.weight.data.t(), bias = layer.bias, quant_state = qs)
            Z._bnb_4bit_direct(layer, x, qs)
            assert layer.__dict__["_unsloth_bnb_direct"] is not plan
            plan = layer.__dict__["_unsloth_bnb_direct"]
            assert _strip_out(lib.calls[-2]) == _strip_out(lib.calls[-1]), name
        qs.state2.absmax = qs.state2.absmax.clone()
        Z._bnb_4bit_direct(layer, x, qs)
        assert lib.calls[-1][1][2] == qs.state2.absmax.data_ptr()
        layer.weight.data = layer.weight.data.clone()
        layer.weight.data_ptr = layer.weight.data.data_ptr
        Z._bnb_4bit_direct(layer, x, qs)
        assert lib.calls[-1][1][1] == layer.weight.data.data_ptr()


@pytest.mark.parametrize("case", ["grad", "bias_grad", "dtype", "compiling", "off", "no_custom", "K", "bias_2d", "fp32_view",
                                  "not_current_device", "nested_bs", "abi", "activation_elsewhere",
                                  "jit_trace", "functorch", "dispatch_mode", "subclass", "fp16_offset", "rows_over_cap"])
def test_declines(lib, monkeypatch, case):
    layer, qs = _layer(nested = case == "nested_bs")
    x = torch.randn(1, 1, 128).to(torch.bfloat16)
    ctx = torch.no_grad()
    if case == "grad":
        x.requires_grad_(True)
        ctx = contextlib.nullcontext()
    elif case == "bias_grad":
        layer.bias.requires_grad_(True)
        ctx = contextlib.nullcontext()
    elif case == "dtype":
        x = x.half()
    elif case == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif case == "off":
        monkeypatch.setitem(Z._BNB_DIRECT, "on", False)
    elif case == "no_custom":
        monkeypatch.setattr(bops, "_gemm_4bit_use_custom_fn", lambda *a: False)
    elif case == "K":
        x = torch.randn(1, 1, 64).to(torch.bfloat16)
    elif case == "bias_2d":
        layer.bias = layer.bias[None]
    elif case == "fp32_view":
        layer.weight.dtype = torch.float32
    elif case == "not_current_device":
        monkeypatch.setattr(Z, "_on_current_cuda", lambda x: False)
    elif case == "nested_bs":
        qs.state2.blocksize = 128
    elif case == "abi":
        lib.cgemm_4bit_bf16.argtypes = [None] * 15  # a kernel built with another argument list
    elif case == "activation_elsewhere":
        x = x.as_subclass(_OnDevice1)
        monkeypatch.setattr(Z, "_plain_eager", lambda x: True)
    elif case == "jit_trace":
        monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    elif case == "functorch":
        monkeypatch.setattr(Z, "_is_wrapped", lambda x: True)  # vmap / grad transforms
    elif case == "dispatch_mode":
        monkeypatch.setattr(Z, "_dispatch_depth", lambda: 1)  # FakeTensorMode and friends
    elif case == "subclass":
        x = x.as_subclass(_OnDevice1)
    elif case == "fp16_offset":
        layer, qs = _layer(nested = True)
        qs.offset = qs.offset.half()
    elif case == "rows_over_cap":
        monkeypatch.setattr(bops, "_gemm_4bit_custom_max_m", 0)
    with ctx:
        assert Z._bnb_4bit_direct(layer, x, qs) is None
    assert not lib.calls and Z.BNB_4BIT_DIRECT_CALLS["direct"] == 0


class _OnDevice1(torch.Tensor):
    def get_device(self):
        return 1


def test_warm_plan_follows_metadata_edits(lib):
    layer, qs = _layer(nested = True)
    x = torch.randn(1, 1, 128).to(torch.bfloat16)
    with torch.no_grad():
        Z._bnb_4bit_direct(layer, x, qs)
        n = len(lib.calls)
        qs.state2.blocksize = 128  # bitsandbytes raises for this nested blocksize: left to it
        assert Z._bnb_4bit_direct(layer, x, qs) is None and len(lib.calls) == n
        qs.state2.blocksize = 256
        qs.quant_type = "fp4"  # requantized in place, same pointers
        Z._bnb_4bit_direct(layer, x, qs)
        assert lib.calls[-1][1][12] == 1
        qs.shape = torch.Size([32, 128])  # pruned rows, same pointers
        out = Z._bnb_4bit_direct(layer, x, qs)
        assert out.shape[-1] == 32 and lib.calls[-1][1][9] == 32
    assert lib.calls[-1][1][5] == qs.offset.data_ptr()


def test_plan_keeps_no_tensors_alive(lib):
    import gc
    import weakref
    layer, qs = _layer(nested = True)
    x = torch.randn(1, 1, 128).to(torch.bfloat16)
    with torch.no_grad():
        Z._bnb_4bit_direct(layer, x, qs)
    old = [weakref.ref(qs.absmax), weakref.ref(qs.state2.absmax), weakref.ref(qs.offset)]
    qs.absmax, qs.state2.absmax, qs.offset = qs.absmax.clone(), qs.state2.absmax.clone(), qs.offset.clone()
    gc.collect()
    assert all(r() is None for r in old)


def test_row_picks_stay_bounded(lib, monkeypatch):
    layer, qs = _layer()
    monkeypatch.setattr(Z, "_PICKS_MAX", 4)
    with torch.no_grad():
        for M in range(1, 12):
            Z._bnb_4bit_direct(layer, torch.randn(1, M, 128).to(torch.bfloat16), qs)
    assert len(layer.__dict__["_unsloth_bnb_direct"][9]) <= 4


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("M", [1, 4])
def test_helper_same_kernel_call_as_bitsandbytes(lib, nested, M):
    layer, qs = _layer(nested = nested, bias = False)
    weight = layer.weight
    x = torch.randn(1, M, 128).to(torch.bfloat16)
    with torch.no_grad():
        bnb.matmul_4bit(x, weight.data.t(), quant_state = qs)
        out = Z.bnb_matmul_4bit_direct(x, weight, qs)
    assert out is not None and out.shape == (1, M, 64) and len(lib.calls) == 2
    assert _strip_out(lib.calls[0]) == _strip_out(lib.calls[1])
    assert "_unsloth_bnb_direct" in qs.__dict__ and Z.BNB_4BIT_DIRECT_CALLS["direct"] == 1


def test_helper_writes_into_a_matching_out_buffer(lib):
    layer, qs = _layer(bias = False)
    x = torch.randn(1, 4, 128).to(torch.bfloat16)
    with torch.no_grad():
        buf = torch.empty(1, 4, 64, dtype = torch.bfloat16)
        assert Z.bnb_matmul_4bit_direct(x, layer.weight, qs, out = buf) is buf
        assert lib.calls[-1][1][6] == buf.data_ptr()
        wide = torch.empty(1, 4, 128, dtype = torch.bfloat16)[..., :64]
        assert Z.bnb_matmul_4bit_direct(x, layer.weight, qs, out = wide) is wide
        assert lib.calls[-1][1][6] != wide.data_ptr()


def test_helper_plan_does_not_keep_the_quant_state_alive(lib):
    import gc
    import weakref
    layer, qs = _layer(bias = False)
    with torch.no_grad():
        assert Z.bnb_matmul_4bit_direct(torch.randn(1, 2, 128).to(torch.bfloat16), layer.weight, qs) is not None
    ref = weakref.ref(qs)
    del qs, layer
    gc.disable()
    try:
        assert ref() is None  # freed by refcount alone, no cycle through its own plan
    finally:
        gc.enable()


def test_helper_declines_like_the_forward(lib, monkeypatch):
    layer, qs = _layer(bias = False)
    x = torch.randn(1, 2, 128).to(torch.bfloat16)
    monkeypatch.setitem(Z._BNB_DIRECT, "on", False)
    with torch.no_grad():
        assert Z.bnb_matmul_4bit_direct(x, layer.weight, qs) is None
    monkeypatch.setitem(Z._BNB_DIRECT, "on", True)
    x.requires_grad_(True)
    assert Z.bnb_matmul_4bit_direct(x, layer.weight, qs) is None
    assert not lib.calls


def test_bitsandbytes_api_drift_turns_it_off(lib, monkeypatch):
    def changed(A, B, shapeB, absmax, blocksize, quant_type, bias = None, absmax_8bit = None, absmax_code = None,
                absmax_offset = None, extra = None):
        raise AssertionError
    monkeypatch.setattr(bops, "_gemm_4bit_kernel_impl", changed)
    layer, qs = _layer()
    with torch.no_grad():
        assert Z._bnb_4bit_direct(layer, torch.randn(1, 1, 128).to(torch.bfloat16), qs) is None
    assert Z._BNB_DIRECT["ops"] is False and not lib.calls


@pytest.mark.parametrize("agree", [True, False])
def test_first_call_is_checked_against_bitsandbytes(lib, monkeypatch, agree):
    monkeypatch.setitem(Z._BNB_DIRECT, "checked", {})
    layer, qs = _layer()
    x = torch.randn(1, 1, 128).to(torch.bfloat16)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: torch.zeros(*a, **k))
    ref = torch.zeros(1, 1, 64, dtype = torch.bfloat16) if agree else torch.ones(1, 1, 64, dtype = torch.bfloat16)
    monkeypatch.setattr(bnb, "matmul_4bit", lambda *a, **k: ref)
    with torch.no_grad():
        out = Z._bnb_4bit_direct(layer, x, qs)
        assert torch.equal(out, ref)
        assert Z._BNB_DIRECT["checked"][(torch.bfloat16, False, True, 2)] is agree
        again = Z._bnb_4bit_direct(layer, x, qs)
    assert (again is not None) is agree
    assert Z.BNB_4BIT_DIRECT_CALLS["direct"] == (2 if agree else 0)
    monkeypatch.setitem(Z._BNB_DIRECT, "checked", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    n = len(lib.calls)
    with torch.no_grad():
        assert Z._bnb_4bit_direct(layer, x, qs) is None
    assert len(lib.calls) == n


def test_patched_forward_takes_the_direct_path(monkeypatch):
    Z.patch_bitsandbytes_linear4bit_forward()
    forward = bnb.nn.modules.Linear4bit.forward
    layer, qs = _layer()
    layer.weight.shape = (64, 64)
    sentinel = torch.zeros(1)
    seen = []
    monkeypatch.setattr(Z, "_bnb_4bit_direct", lambda self, x, q: seen.append(q) or sentinel)
    assert forward(layer, torch.randn(1, 1, 128).to(torch.bfloat16)) is sentinel and seen == [qs]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "NVIDIA CUDA")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_bitwise_against_bitsandbytes(nested, dtype, monkeypatch):
    Z.patch_bitsandbytes_linear4bit_forward()
    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(2880, 4096, bias = True, compute_dtype = dtype, quant_type = "nf4",
                            compress_statistics = nested).cuda()
    if Z._bnb_direct_ops() is False:
        pytest.skip("bitsandbytes without the fused 4-bit GEMM on this build")
    before = Z.BNB_4BIT_DIRECT_CALLS["direct"]
    for M in (1, 4, 64):
        x = torch.randn(1, M, 2880, device = "cuda", dtype = dtype)
        with torch.no_grad():
            monkeypatch.setitem(Z._BNB_DIRECT, "on", False)
            ref = lin(x)
            monkeypatch.setitem(Z._BNB_DIRECT, "on", True)
            for _ in range(2):
                out = lin(x)
                assert torch.equal(out, ref), M
    assert Z.BNB_4BIT_DIRECT_CALLS["direct"] > before
    lin.cpu().cuda()
    x = torch.randn(1, 1, 2880, device = "cuda", dtype = dtype)
    with torch.no_grad():
        monkeypatch.setitem(Z._BNB_DIRECT, "on", False)
        ref = lin(x)
        monkeypatch.setitem(Z._BNB_DIRECT, "on", True)
        assert torch.equal(lin(x), ref)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "NVIDIA CUDA")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_helper_bitwise_against_bitsandbytes(nested, dtype):
    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(2048, 3072, bias = False, compute_dtype = dtype, quant_type = "nf4",
                            compress_statistics = nested).cuda()
    if Z._bnb_direct_ops() is False:
        pytest.skip("bitsandbytes without the fused 4-bit GEMM on this build")
    W, qs = lin.weight, lin.weight.quant_state
    before = Z.BNB_4BIT_DIRECT_CALLS["direct"]
    expected = 0
    for M in (1, 4, 8, 16):
        x = torch.randn(M, 1, 2048, device = "cuda", dtype = dtype)
        fused = M <= bops._gemm_4bit_custom_max_m and bool(bops._gemm_4bit_use_custom_fn(0, dtype, M, 3072, 2048))
        with torch.no_grad():
            ref = bnb.matmul_4bit(x, W.t(), quant_state = qs)
            for _ in range(2):
                buf = torch.empty_like(ref)
                out = Z.bnb_matmul_4bit_direct(x, W, qs, out = buf)
                assert (out is not None) == fused, M
                if fused:
                    assert out is buf and torch.equal(out, ref), M
                    expected += 1
    assert Z.BNB_4BIT_DIRECT_CALLS["direct"] - before == expected
