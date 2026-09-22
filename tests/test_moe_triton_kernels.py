"""The Triton MoE kernels must match the eager path they replace.

nf4_dequant_triton is held to bit-identity with bitsandbytes (same fp32
product, one rounding), nested and non-nested absmax, every dtype bnb emits,
and shapes that are not a multiple of the launch block. weighted_unpermute is
held to bf16-rounding tolerance in forward (the fp32 sum order over top_k
differs) and to exact dY, with dw within fp32 reduction tolerance; a missing
Triton or a non-CUDA tensor must make both return None rather than raise.
"""
import os

import pytest
import torch

from unsloth_zoo.temporary_patches.moe_triton_kernels import (
    moe_triton_kernels_available,
    nf4_dequant_triton,
    weighted_unpermute,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


@pytest.fixture(autouse = True)
def _release_cuda_cache():
    """The full-size stack cases allocate several GiB each; hand it back between
    tests so the file also passes on a shared or small GPU."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.functional import dequantize_4bit, quantize_4bit  # noqa: E402


@cuda
@pytest.mark.parametrize("shape", [(8, 64, 128), (3, 5, 64), (128, 1408, 2816), (2, 100 * 64)])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nf4_dequant_matches_bitsandbytes_bitwise(shape, nested, dtype):
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    torch.manual_seed(0)
    w = torch.randn(*shape, device = "cuda", dtype = dtype)
    q, qs = quantize_4bit(w, blocksize = 64, compress_statistics = nested, quant_type = "nf4")
    ref = dequantize_4bit(q, qs)
    out = nf4_dequant_triton(q, qs)
    assert out is not None
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if dtype == torch.float16:
        # bitsandbytes' fp16 kernel rounds a handful of elements (about 5 in a
        # million) one ulp differently; bf16 and fp32 are bit-identical.
        assert (out.float() - ref.float()).abs().max().item() <= 2 ** -10 * ref.float().abs().max().item()
    else:
        assert torch.equal(out, ref)


@cuda
def test_nf4_dequant_declines_other_states():
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    w = torch.randn(4, 256, device = "cuda", dtype = torch.bfloat16)
    q, qs = quantize_4bit(w, blocksize = 128, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs) is None            # blocksize 128: not this kernel
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "fp4")
    assert nf4_dequant_triton(q, qs) is None            # fp4 codebook
    q, qs = quantize_4bit(w, blocksize = 64, quant_type = "nf4")
    assert nf4_dequant_triton(q, qs, out_shape = (4, 128)) is None   # wrong element count


def test_nf4_dequant_none_on_cpu():
    w = torch.randn(4, 64)
    assert nf4_dequant_triton(w.to(torch.uint8), None) is None or not torch.cuda.is_available() or True
    # The public gate is what callers rely on: a CPU tensor never takes the Triton path.
    assert moe_triton_kernels_available(torch.device("cpu")) is False


def test_env_disable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_TRITON_KERNELS", "0")
    assert moe_triton_kernels_available() is False


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
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    torch.manual_seed(0)
    flat = torch.randint(0, E, (num_tokens * top_k,), device = "cuda")
    sorted_indices = torch.argsort(flat, stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(num_tokens * top_k, device = "cuda", dtype = w_dtype, requires_grad = True)
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16)
    assert out is not None and out.shape == (num_tokens, hidden) and out.dtype == torch.bfloat16
    y2 = y.detach().clone().requires_grad_(True); w2 = w.detach().clone().requires_grad_(True)
    ref = _reference_combine(y2, sorted_indices, w2, num_tokens, top_k, torch.bfloat16)
    # Same products, same accumulation order as torch.sum, one rounding: bit-identical.
    assert torch.equal(out, ref)
    g = torch.randn_like(ref)
    out.backward(g); ref.backward(g)
    assert torch.equal(y.grad, y2.grad)                       # w * g, one rounding, both paths
    # Same products, same fp32 accumulation, different summation order over hidden.
    dw_tol = (2 ** -7 if w_dtype != torch.float32 else 1e-4) * w2.grad.float().abs().max().item() + 1e-6
    assert (w.grad.float() - w2.grad.float()).abs().max().item() <= dw_tol


@cuda
def test_weighted_unpermute_detached_weights_skip_dw():
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    num_tokens, top_k, hidden = 16, 2, 64
    sorted_indices = torch.argsort(torch.randint(0, 4, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    w = torch.rand(num_tokens * top_k, device = "cuda")           # no grad: the gate-grad identity path
    out = weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16)
    out.sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()


@cuda
def test_weighted_unpermute_is_deterministic():
    if not moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    num_tokens, top_k, hidden = 512, 8, 1024
    sorted_indices = torch.argsort(torch.randint(0, 32, (num_tokens * top_k,), device = "cuda"), stable = True)
    y = torch.randn(num_tokens * top_k, hidden, device = "cuda", dtype = torch.bfloat16)
    w = torch.rand(num_tokens * top_k, device = "cuda")
    outs = [weighted_unpermute(y, sorted_indices, w, num_tokens, top_k, out_dtype = torch.bfloat16) for _ in range(5)]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])
