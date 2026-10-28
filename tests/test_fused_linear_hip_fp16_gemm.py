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


# HIP has no fast fp32-out GEMM, so fp16 GEMMs there take _half_mm_bounded: a half-result GEMM whose rows are
# under the worst-case bound take a plain half GEMM; otherwise eager mode takes the exact fp32 GEMM and CUDA graph
# capture pre-scales those rows by a power of two. The helper is device agnostic, so both modes are checked here on
# CPU (capture is simulated); the routing (HIP only) runs on gfx1151.
import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl


@pytest.fixture(params = [False, True], ids = ["eager", "capture"])
def capturing(request, monkeypatch):
    monkeypatch.setattr(fl, "_capturing", lambda x: request.param)
    return request.param


def _dz_like(R, V, peak, seed):
    # Scaled forward gradient rows: peak * (p - onehot), row L1 <= 2 * peak.
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(R, V, generator = g) * 3, -1)
    t = torch.randint(0, V, (R,), generator = g)
    return (peak * (p - torch.nn.functional.one_hot(t, V))).half()


@pytest.mark.parametrize("w_max", [0.05, 1.0, 4.0, 40.0])
def test_bounded_half_mm_never_overflows_and_matches_fp64(w_max, capturing):
    a = _dz_like(64, 4001, 16384.0, 0)
    g = torch.Generator().manual_seed(1)
    b = (torch.randn(4001, 96, generator = g) * (w_max / 4)).clamp(-w_max, w_max).half()
    b[7, 3] = w_max
    ref = a.double() @ b.double()
    got = fl._half_mm_bounded(a, b)
    assert got.dtype == torch.float32 and torch.isfinite(got).all()
    # One fp16 rounding of each output (2^-11 relative) plus fp32 accumulation, on the row scale.
    err = ((got.double() - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-30)).max().item()
    assert err < 2e-3, err


def test_bounded_half_mm_rescues_an_overflowing_plain_half_gemm(capturing):
    a = _dz_like(32, 2048, 16384.0, 2)
    b = torch.full((2048, 16), 8.0).half()
    b[::2] = -8.0
    b[a[0].float().argmin().item()] = 8.0
    plain = (a @ b).float()
    got = fl._half_mm_bounded(a, b)
    ref = a.double() @ b.double()
    assert torch.isfinite(got).all()
    assert torch.allclose(got.double(), ref, rtol = 2e-3, atol = 1e-2 * ref.abs().max().item() * 2 ** -10)
    if ref.abs().max() > 65504:
        assert not torch.isfinite(plain).all()  # the case the bound exists for


def test_rows_under_the_bound_are_the_plain_half_gemm():
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(16, 512, generator = g) * 0.01).half()
    b = (torch.randn(512, 32, generator = g) * 0.1).half()
    assert torch.equal(fl._half_mm_bounded(a, b), (a @ b).float())


def test_scale_below_fp16_subnormal_keeps_the_row(capturing):
    # A row bound past ~2^39 needs a scale under fp16's smallest subnormal (2^-24); one fp16 factor would round it to 0
    # and drop the row. 65504 * 4096 * 4096 = 1.0990e12, finite in the fp32 result.
    a = torch.full((1, 4096), 65504.0, dtype = torch.float16)
    b = torch.full((4096, 1), 4096.0, dtype = torch.float16)
    out = fl._half_mm_bounded(a, b)
    ref = a.double() @ b.double()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), ref, rtol = 2e-3, atol = 0)


def test_logits_past_half_range_stay_finite(capturing):
    # Raw logits outside fp16 range (+-160000) must stay finite: a plain half GEMM gives +-inf and the log-sum-exp NaNs.
    h = torch.zeros(2, 64, dtype = torch.float16)
    h[:, :16] = 100.0
    w = torch.full((8, 64), 100.0, dtype = torch.float16)
    w[1::2] = -100.0
    out = fl._half_mm_bounded(h, w.t())
    assert not torch.isfinite((h @ w.t()).float()).all()
    ref = h.double() @ w.double().t()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double(), ref, rtol = 2e-3, atol = 0)


def test_eager_keeps_small_entries_of_a_scaled_row(capturing):
    # The bound is worst case: a +-16384 row with a 1e-4 entry trips it although the true result is small. Scaling
    # the row down would flush that entry below fp16's subnormal range (result 0 instead of 6.55), so eager mode takes
    # the exact fp32 GEMM. Capture cannot sync, keeps the scaled path, and stays finite.
    a = torch.tensor([[-16384.0, 16384.0, 1e-4, 0, 0, 0, 0, 0]], dtype = torch.float16)
    b = torch.zeros(8, 1, dtype = torch.float16)
    b[2] = 65504.0
    out = fl._half_mm_bounded(a, b)
    ref = a.double() @ b.double()
    assert torch.isfinite(out).all()
    if not capturing:
        torch.testing.assert_close(out.double(), ref, rtol = 1e-6, atol = 0)
