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
    # Fork the RNG: seeding here pushed a later unseeded NF4 round trip past its tolerance.
    with torch.random.fork_rng(devices = [torch.cuda.current_device()]):
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
    A = torch.randn(8, 1024, 1536, device = "cuda", dtype = dtype)
    ref_data, ref_state = orig_q(A, blocksize = 64, compress_statistics = nested, quant_type = "nf4", quant_storage = storage)
    monkeypatch.setattr(L, "BNB_INT32_ELEMENT_LIMIT", A.numel() - 1)
    monkeypatch.setattr(L, "_PIECE_ELEMENTS", 2**21)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = nested, quant_type = "nf4", quant_storage = storage)
    assert data.shape == ref_data.shape and data.dtype == ref_data.dtype
    # Bytes, not torch.equal: bf16 / fp16 quant_storage views hold NaN bit patterns.
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
    assert torch.equal(F.dequantize_4bit(ref_data, ref_state), ref_out)


@pytest.mark.parametrize("row_packed", [True, False])
def test_chunked_out_buffer_matches_bitsandbytes(monkeypatch, row_packed):
    from bitsandbytes import functional as F
    orig_q, orig_d = _original()
    torch.manual_seed(0)
    A = torch.randn(1024, 1536, device = "cuda", dtype = torch.bfloat16)
    data, state = orig_q(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    if row_packed:
        data = data.reshape(1, -1)
    ref_buffer = torch.empty(A.shape, device = "cuda", dtype = A.dtype)
    try:
        ref = orig_d(data, state, out = ref_buffer)
    except RuntimeError as e:
        pytest.skip(f"bitsandbytes' own dequantize_4bit(out = ...) fails here: {e}")
    monkeypatch.setattr(L, "BNB_INT32_ELEMENT_LIMIT", A.numel() - 1)
    monkeypatch.setattr(L, "_PIECE_ELEMENTS", 2**18)
    buffer = torch.empty(A.shape, device = "cuda", dtype = A.dtype)
    got = F.dequantize_4bit(data, state, out = buffer)
    assert got.shape == ref.shape and torch.equal(got, ref)
    assert got.data_ptr() == buffer.data_ptr() and torch.equal(buffer, ref_buffer)
    transposed = torch.empty(A.shape[1], A.shape[0], device = "cuda", dtype = A.dtype).t()
    got = F.dequantize_4bit(data, state, out = transposed)
    assert torch.equal(transposed, ref_buffer) and torch.equal(got, ref)


def test_small_tensors_take_the_original_path():
    from bitsandbytes import functional as F
    orig_q, _ = _original()
    A = torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    ref = orig_q(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    assert torch.equal(data, ref[0]) and torch.equal(state.absmax, ref[1].absmax)


def test_real_tensor_past_int32():
    if torch.cuda.mem_get_info()[0] < 40 * 2**30:
        pytest.skip("needs 40 GB of free GPU memory")
    """Inkling-Small's gate_up_proj shape: 256 x 4096 x 4096 = 4.29e9 values."""
    from bitsandbytes import functional as F
    from bitsandbytes.nn import Params4bit
    orig_q, _ = _original()
    # No unpatched baseline: bitsandbytes' C code exits the process at this size.
    A = torch.randn(256, 4096, 4096, device = "cuda", dtype = torch.bfloat16)
    data, state = F.quantize_4bit(A, blocksize = 64, compress_statistics = True, quant_type = "nf4")
    assert data.shape == ((A.numel() + 1) // 2, 1)
    out = F.dequantize_4bit(data, state)
    assert out.shape == A.shape
    err = (out.float() - A.float()).abs()
    assert err.mean() < 0.1
    for boundary in (2**30, 2**31):
        seam = err.reshape(-1)[boundary - 4096 : boundary + 4096].mean()
        assert seam < 0.15, seam
    del out, data, state
    p = Params4bit(A.cpu(), requires_grad = False, quant_type = "nf4", compress_statistics = True).cuda()
    assert p.quant_state.shape == A.shape and p.data.shape[0] == (A.numel() + 1) // 2


def test_small_tensors_keep_the_callers_defaults():
    from bitsandbytes import functional as F
    orig_q, orig_d = _original()
    torch.manual_seed(0)
    A = torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16)
    data, state = F.quantize_4bit(A)
    ref_data, ref_state = orig_q(A)
    assert torch.equal(data, ref_data) and state.blocksize == ref_state.blocksize
    assert torch.equal(F.dequantize_4bit(data, state), orig_d(ref_data, ref_state))
    assert torch.equal(F.dequantize_4bit(data, quant_state = state), orig_d(ref_data, ref_state))
