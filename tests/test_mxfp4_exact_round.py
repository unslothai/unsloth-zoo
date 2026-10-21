"""MXFP4 dequant: the fp32-multiply + integer-rounding kernel path (default on ROCm).

On gfx11 ROCm Triton lowers bf16 x bf16 to v_dot2_bf16_bf16(a, b, +0) and an fp32 multiply followed by
an fp16 cast to v_fma_mixlo_f16(a, b, +0); both turn -0.0 into +0.0, and the dot flushes subnormals, so
the fused kernel failed its bit-exact self-check there. The exact path must match the torch reference
bit for bit on any GPU (NVIDIA runs it with UNSLOTH_MXFP4_EXACT_ROUND=1), and is the default on ROCm.
"""

import pytest
import torch

import unsloth_zoo.mxfp4_dequant as mxd
from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch

needs_gpu = pytest.mark.skipif(not (torch.cuda.is_available() and mxd._HAS_TRITON), reason = "needs a GPU and Triton")
on_hip = torch.version.hip is not None


def _every_nibble_under_every_scale(device):
    # Each byte 0..255 (both nibbles, so every e2m1 code in both positions) under every e8m0 scale 0..255.
    byte = torch.arange(256, device = device, dtype = torch.int32)
    blocks = byte.view(1, 1, 256).expand(256, 1, 256).reshape(256, 16, 16).to(torch.uint8)
    blocks = blocks.reshape(2, 128, 16, 16)
    scales = torch.arange(256, device = device, dtype = torch.int32).view(256, 1).expand(256, 16)
    return blocks, scales.reshape(2, 128, 16).to(torch.uint8).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _f32_patterns(device):
    # Every exponent with edge and random mantissas, the fp16 / bf16 halfway points, subnormal and
    # overflow boundaries, +-0, +-inf, NaN.
    g = torch.Generator(device = "cpu").manual_seed(0)
    exp = torch.arange(256, dtype = torch.int64).repeat_interleave(64)
    man = torch.randint(0, 1 << 23, (exp.numel(),), generator = g, dtype = torch.int64)
    man[0::64] = 0
    man[1::64] = (1 << 23) - 1
    for i, low in enumerate((0x1000, 0x2000, 0x3000, 0x8000, 0x10000, 0x18000), start = 2):
        man[i::64] = (man[i::64] & ~((1 << 17) - 1)) | low
    base = (exp << 23) | man
    extra = torch.tensor([
        0x00000000, 0x7F800000, 0x7FC00000, 0x7F800001, 0x477FE000, 0x477FEFFF, 0x477FF000, 0x477FF001,
        0x38800000, 0x387FFFFF, 0x387FF000, 0x33000000, 0x33000001, 0x32FFFFFF, 0x33800000, 0x33C00000,
        0x7F7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0x00000001, 0x007FFFFF, 0x3F808000, 0x3F818000,
    ], dtype = torch.int64)
    u = torch.cat([base, extra])
    u = torch.cat([u, u | (1 << 31)])
    u = torch.where(u >= 1 << 31, u - (1 << 32), u).to(torch.int32)
    return u.view(torch.float32).to(device)


if mxd._HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _round_kernel(x_ptr, y_ptr, n, TO_F16: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask = offs < n, other = 0.0)
        y = mxd._f32_to_f16_rne(x) if TO_F16 else mxd._f32_to_bf16_rne(x)
        tl.store(y_ptr + offs, y, mask = offs < n)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rounding_helpers_match_torch(dtype):
    x = _f32_patterns("cuda")
    y = torch.empty_like(x, dtype = dtype)
    _round_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), TO_F16 = dtype == torch.float16, BLOCK = 1024)
    want = x.to(dtype)
    nan = torch.isnan(x)
    assert torch.equal(torch.isnan(y), nan)
    assert torch.equal(_bits(y)[~nan], _bits(want)[~nan])
    # NaN keeps its sign bit (torch's own NaN bits are backend-specific: CUDA emits 0x7FFF).
    assert torch.equal(_bits(y)[nan] < 0, x.view(torch.int32)[nan] < 0)


@needs_gpu
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("exact_round", ["1", "0"])
def test_kernel_is_bit_identical_for_every_nibble_and_scale(monkeypatch, dtype, transpose, exact_round):
    if exact_round == "0" and on_hip:
        pytest.skip("the plain path is the one ROCm Triton miscompiles (signed zero, subnormals)")
    monkeypatch.setenv("UNSLOTH_MXFP4_EXACT_ROUND", exact_round)
    blocks, scales = _every_nibble_under_every_scale("cuda")
    got = mxd._kernel_dequantize(blocks, scales, dtype, transpose, None, None, None)
    want = mxfp4_dequantize_torch(blocks, scales, dtype = dtype, transpose = transpose)
    assert torch.equal(_bits(got), _bits(want)), int((_bits(got) != _bits(want)).sum())


def test_exact_round_is_the_rocm_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MXFP4_EXACT_ROUND", raising = False)
    monkeypatch.setattr(mxd, "_IS_HIP", True)
    assert mxd._exact_round() is True
    monkeypatch.setattr(mxd, "_IS_HIP", False)
    assert mxd._exact_round() is False
    monkeypatch.setenv("UNSLOTH_MXFP4_EXACT_ROUND", "1")
    assert mxd._exact_round() is True
    monkeypatch.setattr(mxd, "_IS_HIP", True)
    monkeypatch.setenv("UNSLOTH_MXFP4_EXACT_ROUND", "0")
    assert mxd._exact_round() is False


@pytest.mark.skipif(not mxd._HAS_TRITON, reason = "needs Triton")
def test_faked_rocm_launches_the_exact_path(monkeypatch):
    # Routing only: on ROCm the launch asks for EXACT_ROUND and not the bf16 bit-built multiply.
    monkeypatch.delenv("UNSLOTH_MXFP4_EXACT_ROUND", raising = False)
    monkeypatch.setattr(mxd, "_IS_HIP", True)
    seen = {}
    monkeypatch.setattr(mxd, "_launch", lambda *a: seen.update(bf16_bits = a[-3], exact = a[-1]))
    blocks = torch.zeros(1, 4, 2, 16, dtype = torch.uint8)
    scales = torch.full((1, 4, 2), 127, dtype = torch.uint8)
    mxd._kernel_dequantize(blocks, scales, torch.bfloat16, True, None, None, None)
    assert seen == {"bf16_bits": False, "exact": True}


@needs_gpu
@pytest.mark.skipif(not on_hip, reason = "real ROCm only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rocm_self_check_passes(dtype):
    mxd._KERNEL_VERIFIED.clear()
    assert mxd.mxfp4_kernel_available(dtype = dtype)
    assert mxd._kernel_verified(torch.device("cuda", torch.cuda.current_device()), dtype, False)


def test_hip_codegen_has_no_fused_mixed_precision_ops():
    # Offline: compile the exact path for gfx1151 and check the two combines that broke it are gone.
    pytest.importorskip("triton")
    if not mxd._HAS_TRITON:
        pytest.skip("needs Triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    try:
        import triton.backends.amd  # noqa: F401
    except Exception:
        pytest.skip("Triton built without the AMD backend")
    target = GPUTarget("hip", "gfx1151", 32)
    for out_ty in ("*bf16", "*fp16"):
        sig = {"blocks_ptr": "*u8", "scales_ptr": "*u8", "out_ptr": out_ty, "experts_ptr": "*i32", "counts_ptr": "*i32",
               "N": "i32", "G": "i32", "stride_out_e": "i32", "stride_out_n": "i32", "stride_out_k": "i32"}
        consts = dict(BLOCK_N = 64, BLOCK_G = 2, USE_EXPERTS = False, USE_COUNTS = False, BF16_BITS = False,
                      LONG_INDEXING = False, EXACT_ROUND = True)
        try:
            k = triton.compile(ASTSource(fn = mxd._mxfp4_dequant_kernel, signature = {**sig, **{c: "constexpr" for c in consts}},
                                         constexprs = consts), target = target)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"offline HIP compile unavailable: {type(e).__name__}: {e}"[:200])
        asm = k.asm["amdgcn"]
        assert "v_fma_mixlo_f16" not in asm and "v_fma_mixhi_f16" not in asm and "v_dot2" not in asm, out_ty
