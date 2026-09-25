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

"""The Triton MoE kernels must match the eager path they replace, or return None."""
import os

import pytest
import torch

import unsloth_zoo.temporary_patches.moe_triton_kernels as mk
from unsloth_zoo.temporary_patches.moe_triton_kernels import (
    moe_triton_kernels_available,
    nf4_dequant_triton,
    weighted_unpermute,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit  # noqa: E402


@pytest.fixture(autouse = True)
def _release_cuda_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _need_triton():
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")


@cuda
@pytest.mark.parametrize("shape", [(8, 64, 128), (3, 5, 64), (128, 1408, 2816), (2, 100 * 64)])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nf4_dequant_matches_bitsandbytes_bitwise(shape, nested, dtype):
    _need_triton()
    torch.manual_seed(0)
    w = torch.randn(*shape, device = "cuda", dtype = dtype)
    q, qs = quantize_4bit(w, blocksize = 64, compress_statistics = nested, quant_type = "nf4")
    ref = dequantize_4bit(q, qs)
    out = nf4_dequant_triton(q, qs)
    assert out is not None
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if dtype == torch.float16:
        # bitsandbytes' fp16 kernel rounds a few elements one ulp differently.
        assert (out.float() - ref.float()).abs().max().item() <= 2 ** -10 * ref.float().abs().max().item()
    else:
        assert torch.equal(out, ref)


@cuda
@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float16, torch.float32])
def test_nf4_dequant_other_quant_storage_dtypes(storage):
    _need_triton()
    w = torch.randn(4, 64, 256, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 64, compress_statistics = True, quant_type = "nf4", quant_storage = storage)
    assert q.dtype == storage
    out = nf4_dequant_triton(q, qs)
    assert out is not None and torch.equal(out, dequantize_4bit(q, qs))


@cuda
def test_nf4_dequant_reloaded_quant_state():
    """A prequantized checkpoint arrives through QuantState.from_dict."""
    _need_triton()
    w = torch.randn(8, 64, 256, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    qs2 = QuantState.from_dict(qs_dict = qs.as_dict(packed = True), device = torch.device("cuda"))
    assert torch.equal(nf4_dequant_triton(q, qs2), dequantize_4bit(q, qs2))


@cuda
def test_nf4_dequant_offset_as_python_float():
    """Older bitsandbytes kept the nested-absmax offset as a float."""
    _need_triton()
    import copy
    w = torch.randn(4, 64, 128, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    qso = copy.copy(qs)
    qso.offset = float(qs.offset)
    assert torch.equal(nf4_dequant_triton(q, qso), dequantize_4bit(q, qs))


@cuda
def test_nf4_dequant_declines_other_states():
    _need_triton()
    import copy
    w = torch.randn(4, 256, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 128, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs) is None            # blocksize 128: not this kernel
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "fp4")
    assert nf4_dequant_triton(q, qs) is None            # fp4 codebook
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs, out_shape = (4, 128)) is None   # wrong element count
    qsc = copy.copy(qs)
    qsc.code = qs.code.cpu()
    assert nf4_dequant_triton(q, qsc) is None           # state tensor on another device
    w = torch.randn(7, 9, device = "cuda", dtype = torch.bfloat16)   # odd numel: bnb pads a byte
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs) is None


def test_gate_never_takes_a_cpu_tensor():
    assert moe_triton_kernels_available(torch.device("cpu")) is False


def test_env_disable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_KERNELS", "0")
    assert moe_triton_kernels_available() is False


@cuda
def test_failed_launch_falls_back_and_disables(monkeypatch):
    """A Triton that imports but cannot compile must not take the run down."""
    _need_triton()
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
    def boom():
        raise RuntimeError("no compiler")
    monkeypatch.setattr(mk, "_get_kernels", boom)
    w = torch.randn(4, 64, 128, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs) is None
    assert mk._DISABLED_REASON is not None and "no compiler" in mk._DISABLED_REASON
    assert moe_triton_kernels_available() is False     # and stays off for the process
    y = torch.randn(8, 16, device = "cuda", dtype = torch.bfloat16)
    assert weighted_unpermute(y, torch.arange(8, device = "cuda"), torch.rand(8, device = "cuda"), 4, 2) is None
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)


def _reference_combine(y, sorted_indices, w_perm, num_tokens, top_k, out_dtype):
    from unsloth_zoo.temporary_patches.moe_utils import combine_permuted_moe_outputs
    return combine_permuted_moe_outputs(
        y * w_perm.unsqueeze(-1), sorted_indices, num_tokens, top_k, out_dtype = out_dtype,
    )


@cuda
@pytest.mark.parametrize("num_tokens,top_k,hidden,E", [(2048, 8, 2816, 128), (7, 2, 100, 4), (1, 1, 1024, 2), (33, 4, 1500, 16),
                                                      (300, 6, 512, 64), (128, 10, 768, 32), (64, 16, 256, 32), (50, 3, 100, 8)])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_weighted_unpermute_matches_eager(num_tokens, top_k, hidden, E, w_dtype):
    _need_triton()
    torch.manual_seed(0)
    flat = torch.randint(0, E, (num_tokens * top_k,), device = "cuda")
    sorted_indices = torch.argsort(flat, stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(num_tokens * top_k, device = "cuda", dtype = w_dtype, requires_grad = True)
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16)
    assert out is not None and out.shape == (num_tokens, hidden) and out.dtype == torch.bfloat16
    y2 = y.detach().clone().requires_grad_(True); w2 = w.detach().clone().requires_grad_(True)
    ref = _reference_combine(y2, sorted_indices, w2, num_tokens, top_k, torch.bfloat16)
    assert torch.equal(out, ref)
    g = torch.randn_like(ref)
    out.backward(g); ref.backward(g)
    assert torch.equal(y.grad, y2.grad)
    # dw sums over hidden in a different order, so it gets a tolerance.
    dw_tol = (2 ** -7 if w_dtype != torch.float32 else 1e-4) * w2.grad.float().abs().max().item() + 1e-6
    assert (w.grad.float() - w2.grad.float()).abs().max().item() <= dw_tol


@cuda
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("y_dtype", [torch.bfloat16, torch.float16])
def test_weighted_unpermute_dtype_pairings(w_dtype, y_dtype):
    """bf16 router with fp16 activations promotes to fp32 and must not round the product."""
    _need_triton()
    torch.manual_seed(1)
    num_tokens, top_k, hidden = 33, 4, 1500
    sorted_indices = torch.argsort(torch.randint(0, 16, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = y_dtype)
    w = torch.rand(num_tokens * top_k, device = "cuda", dtype = w_dtype)
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = y_dtype)
    assert torch.equal(out, _reference_combine(y, sorted_indices, w, num_tokens, top_k, y_dtype))


@cuda
def test_weighted_unpermute_empty_experts():
    _need_triton()
    num_tokens, top_k, hidden = 256, 8, 512
    sorted_indices = torch.argsort(torch.randint(0, 3, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16)
    w = torch.rand(num_tokens * top_k, device = "cuda")
    assert torch.equal(weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16),
                       _reference_combine(y, sorted_indices, w, num_tokens, top_k, torch.bfloat16))


@cuda
def test_weighted_unpermute_detached_weights_skip_dw():
    _need_triton()
    num_tokens, top_k, hidden = 16, 2, 64
    sorted_indices = torch.argsort(torch.randint(0, 4, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(num_tokens * top_k, device = "cuda")           # no grad: the gate-grad identity path
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16)
    out.sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()


@cuda
def test_weighted_unpermute_is_deterministic():
    _need_triton()
    num_tokens, top_k, hidden = 512, 8, 1024
    sorted_indices = torch.argsort(torch.randint(0, 32, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16)
    w = torch.rand(num_tokens * top_k, device = "cuda")
    outs = [weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16) for _ in range(5)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@cuda
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_a_backward_kernel_failure_finishes_in_eager(monkeypatch, w_dtype):
    """A backward compile failure happens after the forward's fallback; it must still produce gradients."""
    _need_triton()
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
    torch.manual_seed(0)
    num_tokens, top_k, hidden, E = 33, 4, 1500, 16
    flat = torch.randint(0, E, (num_tokens * top_k,), device = "cuda")
    sorted_indices = torch.argsort(flat, stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(num_tokens * top_k, device = "cuda", dtype = w_dtype, requires_grad = True)
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16)
    assert out is not None

    triton, nf4, fwd, _, _ = mk._get_kernels()

    class Broken:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("backward compile failed")
            return launch

    monkeypatch.setattr(mk, "_get_kernels", lambda: (triton, nf4, fwd, Broken(), Broken()))
    y2 = y.detach().clone().requires_grad_(True); w2 = w.detach().clone().requires_grad_(True)
    ref = _reference_combine(y2, sorted_indices, w2, num_tokens, top_k, torch.bfloat16)
    g = torch.randn_like(ref)
    out.backward(g); ref.backward(g)
    assert mk._DISABLED_REASON is not None and "backward compile failed" in mk._DISABLED_REASON
    assert torch.equal(y.grad, y2.grad)
    dw_tol = (2 ** -7 if w_dtype != torch.float32 else 1e-4) * w2.grad.float().abs().max().item() + 1e-6
    assert (w.grad.float() - w2.grad.float()).abs().max().item() <= dw_tol
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
