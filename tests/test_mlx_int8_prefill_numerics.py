# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The arithmetic, which is where output quality is decided.

None of this needs Apple hardware: the W8A8 algorithm is defined by the algorithm, not
by Metal, so the portable backend computes exactly what the M5 kernels are supposed to.
A quality regression that shows up here would show up there.
"""

import pytest

from _mlx_int8_helpers import (
    make_quantized_linear,
    make_quantized_model,
    reset_int8_state,
)


@pytest.fixture(autouse = True)
def clean_state():
    reset_int8_state()
    yield
    reset_int8_state()


@pytest.fixture
def make_ql():
    return make_quantized_linear


@pytest.fixture
def quantized_model():
    return make_quantized_model()


import mlx.core as mx
import mlx.nn as nn
import numpy as np

from unsloth_zoo.mlx.int8_prefill import registry, scales
from unsloth_zoo.mlx.int8_prefill.backends import portable

BITS_GROUPS = [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128)]
DTYPES = [mx.float32, mx.float16, mx.bfloat16]


def quantize(w, bits, group_size):
    packed, s, b = mx.quantize(w, group_size, bits, mode="affine")
    return packed, s, b


class TestScaleBound:
    """The analytic bound versus the true absmax.

    JetBrains derive the per-channel int8 scale from the affine metadata alone, which
    reads like an approximation. It is not, for stock `mx.quantize` output: the
    quantizer sets bias = group min, so code 0 is attained in every group and the bound
    is tight. This test exists so that if MLX ever changes that, we find out here rather
    than through a quality complaint from the one person with an M5.
    """

    @pytest.mark.parametrize("bits,group_size", BITS_GROUPS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_bound_is_tight_for_stock_quantize(self, bits, group_size, dtype):
        """Asserted on the distribution, not the extreme, and that is not a dodge.

        For float32 and float16 the bound equals the true absmax everywhere. For
        bfloat16 a thin tail of channels -- measured at 0.1%, worst case 8/7 -- comes out
        loose, because bf16 rounding of the group scale can leave the top code unused so
        the group's real maximum is 13*s + b rather than 15*s + b. Measured end to end,
        switching those channels to exact scales moves the matmul's mean and max relative
        error by nothing at all (0.00875 either way), which is why the analytic bound is
        the default and `UNSLOTH_MLX_INT8_EXACT_SCALES` exists only for learned
        quantizers. Asserting on the max here would encode bf16 rounding luck as a
        contract; asserting on the median and p99 encodes the property we rely on.
        """
        w = mx.random.normal((1024, 512)).astype(dtype)
        packed, s, b = quantize(w, bits, group_size)

        bound = scales.channel_scale_bound(s, b, bits)
        exact = scales.channel_scale_exact(packed, s, b, bits, group_size)
        mx.eval(bound, exact)

        ratio = np.array((bound / exact).astype(mx.float32), copy=False)

        # Never below 1: a scale that under-estimates clips weights at +-127 and loses
        # them outright. This is a correctness requirement, so it is exact.
        assert ratio.min() >= 1.0 - 1e-6, f"bound under-estimates by {1.0 - ratio.min():.2e}"
        assert np.median(ratio) == pytest.approx(1.0, abs=1e-5)
        assert np.percentile(ratio, 99) == pytest.approx(1.0, abs=1e-3)
        # The tail exists but must stay a tail.
        assert (ratio > 1.01).mean() < 0.02, "too many channels have a loose bound"

    def test_bound_never_clips(self):
        """Even where it is loose, the bound must remain an upper bound -- a scale that
        under-estimates would clip weights at +-127 and lose them outright."""
        w = mx.random.normal((128, 512)).astype(mx.float32)
        packed, s, b = quantize(w, 4, 64)
        # Perturb scales the way a learned quantizer (DWQ/AWQ/GPTQ) would, so the
        # extremes are no longer attained.
        s2 = s * 1.7
        bound = scales.channel_scale_bound(s2, b, 4)
        exact = scales.channel_scale_exact(packed, s2, b, 4, 64)
        mx.eval(bound, exact)
        assert bool((bound >= exact * 0.999).all().item()), "bound must not under-estimate"


class TestPortableBackend:
    @pytest.mark.parametrize("bits,group_size", [(4, 32), (4, 64), (4, 128)])
    def test_close_to_mlx_reference(self, bits, group_size):
        """W8A8 is lossy by design; the bar is that it stays close to the 4-bit op."""
        K, N, M = 1024, 2048, 640
        lin = nn.Linear(K, N, bias=False)
        ql = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
        ok, why = registry.register_module(ql, "w")
        assert ok, why
        entry = registry.get(ql["weight"])

        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        got = portable.matmul(x, entry, out_dtype=mx.float32)
        want = mx.quantized_matmul(
            x, ql["weight"], ql["scales"], ql["biases"], True, group_size, bits
        ).astype(mx.float32)
        try:
            mx.eval(got, want)
        except RuntimeError as exc:
            # MLX 0.32.1's CUDA backend raises cuGraphAddKernelNode "invalid argument"
            # for mx.quantized_matmul at group_size 128. That is stock MLX, reproducible
            # with none of this module loaded, and absent on the CPU and Metal backends.
            # Skipping on the reference we cannot build, rather than on a guess about the
            # backend, keeps the coverage everywhere the reference does build.
            if "cuGraph" not in str(exc):
                raise
            pytest.skip(f"stock mx.quantized_matmul cannot evaluate here: {exc}")

        rel = (mx.abs(got - want).max() / mx.abs(want).max()).item()
        assert rel < 0.15, f"max relative error {rel:.4f}"

        # The mean error is the number that actually predicts output quality; a max
        # driven by one outlier channel is not the same thing as a shifted distribution.
        mean_rel = (mx.abs(got - want).mean() / mx.abs(want).mean()).item()
        assert mean_rel < 0.03, f"mean relative error {mean_rel:.4f}"

    def test_gemm_matches_exact_int32(self):
        """The float32 accumulation must agree with a true int32 GEMM.

        float32 holds integers exactly only to 2**24, and a K-long sum of int8 products
        can exceed that, so this pins K small enough that any disagreement is a bug in
        the algorithm rather than accumulation rounding.
        """
        M, K, N = 64, 512, 128
        xq = mx.random.randint(-127, 128, (M, K)).astype(mx.int8)
        wq = mx.random.randint(-127, 128, (N, K)).astype(mx.int8)
        xs = mx.ones((M,), dtype=mx.float32)
        ws = mx.ones((N,), dtype=mx.float32)

        got = portable.int8_gemm(xq, xs, wq, ws)
        mx.eval(got)
        want = np.matmul(
            np.array(xq, copy=False).astype(np.int32),
            np.array(wq, copy=False).astype(np.int32).T,
        )
        assert np.array_equal(np.array(got, copy=False).astype(np.int64), want.astype(np.int64))

    def test_row_quantization_range(self):
        x = mx.random.normal((32, 256)).astype(mx.bfloat16) * 7.0
        xq, xs = portable.quantize_rows(x)
        mx.eval(xq, xs)
        assert int(mx.abs(xq.astype(mx.int32)).max().item()) <= 127
        # Every row should use most of the int8 range, or the scale is wrong.
        row_max = mx.abs(xq.astype(mx.int32)).max(axis=-1)
        mx.eval(row_max)
        assert int(row_max.min().item()) >= 120

    def test_zero_row_does_not_divide_by_zero(self):
        x = mx.zeros((4, 256), dtype=mx.bfloat16)
        xq, xs = portable.quantize_rows(x)
        mx.eval(xq, xs)
        arr = np.array(xq, copy=False)
        assert np.isfinite(np.array(xs, copy=False)).all()
        assert (arr == 0).all()

    def test_leading_batch_dims_preserved(self):
        K, N = 1024, 2048
        ql = nn.QuantizedLinear.from_linear(nn.Linear(K, N, bias=False), group_size=64, bits=4)
        ok, why = registry.register_module(ql, "w")
        assert ok, why
        entry = registry.get(ql["weight"])
        x = mx.random.normal((2, 3, 512, K)).astype(mx.bfloat16)
        out = portable.matmul(x, entry)
        mx.eval(out)
        assert out.shape == (2, 3, 512, N)


class TestGroupRangeRatio:
    """R is the quantity that decides whether per-channel int8 requant is safe."""

    def test_reports_finite_ratios(self):
        w = mx.random.normal((128, 512)).astype(mx.float32)
        _, s, _ = quantize(w, 4, 64)
        r = scales.group_range_ratio(s, 4)
        mx.eval(r)
        arr = np.array(r, copy=False)
        assert np.isfinite(arr).all()
        assert (arr >= 1.0).all()
