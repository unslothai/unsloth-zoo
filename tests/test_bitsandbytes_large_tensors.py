# Chunked 4-bit quantize / dequantize for tensors past bitsandbytes' int32 count.
import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

from unsloth_zoo.temporary_patches import bitsandbytes_large_tensors as L


@pytest.fixture(autouse = True)
def _patched(monkeypatch):
    from bitsandbytes import functional as F
    orig_q, orig_d = F.quantize_4bit, F.dequantize_4bit
    L.patch_bitsandbytes_large_tensors()
    yield
    F.quantize_4bit, F.dequantize_4bit = orig_q, orig_d


def _original():
    from bitsandbytes import functional as F
    return F.quantize_4bit.__wrapped__, F.dequantize_4bit.__wrapped__


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("storage", [torch.uint8, torch.bfloat16])
def test_chunked_matches_single_call(monkeypatch, dtype, nested, storage):
    from bitsandbytes import functional as F
    orig_q, orig_d = _original()
    torch.manual_seed(0)
    A = torch.randn(8, 1024, 1536, device = "cuda", dtype = dtype)   # 12.6M values, 5 pieces at the forced limit
    ref_data, ref_state = orig_q(A, blocksize = 64, compress_statistics = nested, quant_type = "nf4", quant_storage = storage)
    # Force the chunked path: pieces of 2^21 values, limit just under the tensor.
    monkeypatch.setattr(L, "BNB_INT32_ELEMENT_LIMIT", A.numel() - 1)
    monkeypatch.setattr(L, "_PIECE_ELEMENTS", 2**21)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = nested, quant_type = "nf4", quant_storage = storage)
    assert data.shape == ref_data.shape and data.dtype == ref_data.dtype
    # Byte comparison: a bf16 / fp16 quant_storage view holds NaN bit patterns,
    # which torch.equal treats as unequal even when the bytes are the same.
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert state.shape == ref_state.shape and state.nested == nested
    assert torch.equal(state.absmax, ref_state.absmax)
    if nested:
        assert torch.equal(state.offset, ref_state.offset)
        assert torch.equal(state.state2.absmax, ref_state.state2.absmax)
    ref_out = orig_d(ref_data, ref_state)
    out = F.dequantize_4bit(data, state)
    assert out.shape == A.shape and out.dtype == dtype
    assert torch.equal(out, ref_out)
    # And the chunked dequant of bitsandbytes' own single-call state.
    assert torch.equal(F.dequantize_4bit(ref_data, ref_state), ref_out)


def test_small_tensors_take_the_original_path():
    from bitsandbytes import functional as F
    orig_q, _ = _original()
    A = torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    ref = orig_q(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    assert torch.equal(data, ref[0]) and torch.equal(state.absmax, ref[1].absmax)


def test_real_tensor_past_int32():
    """Needs about 30 GB free: the bf16 stack, its packed bytes and the dequantized copy."""
    if torch.cuda.mem_get_info()[0] < 40 * 2**30:
        pytest.skip("needs 40 GB of free GPU memory")
    """Inkling-Small's gate_up_proj shape: 256 x 4096 x 4096 = 4.29e9 values."""
    from bitsandbytes import functional as F
    from bitsandbytes.nn import Params4bit
    orig_q, _ = _original()
    # Not compared against the unpatched call: bitsandbytes' C code exits the
    # process on the CUDA "invalid argument" it raises for this size.
    A = torch.randn(256, 4096, 4096, device = "cuda", dtype = torch.bfloat16)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    assert data.shape == ((A.numel() + 1) // 2, 1)
    out = F.dequantize_4bit(data, state)
    assert out.shape == A.shape
    # NF4 round trip error on unit normal data is bounded; the chunk seams must not stand out.
    err = (out.float() - A.float()).abs()
    assert err.mean() < 0.1
    for boundary in (2**30, 2**31):
        seam = err.reshape(-1)[boundary - 4096 : boundary + 4096].mean()
        assert seam < 0.15, seam
    del out, data, state
    # The Params4bit path (what the quantizer uses) goes through the same function.
    p = Params4bit(A.cpu(), requires_grad = False, quant_type = "nf4", compress_statistics = True).cuda()
    assert p.quant_state.shape == A.shape and p.data.shape[0] == (A.numel() + 1) // 2
