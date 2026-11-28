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

"""GGUF kernel entry points: one output dtype under autocast on every route and under torch.compile, a bias of
another dtype, strided packed views, tiles that do not fit a device's shared memory, and old Triton on Blackwell."""

from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.gguf_kernels import dequantize_gguf, gguf_linear, gguf_linear_backward_input
from unsloth_zoo.gguf_kernels import ops as gguf_ops
from unsloth_zoo.gguf_kernels.formats import GGMLType as T
from unsloth_zoo.gguf_kernels.testing import random_packed

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _weight(rows = 32, cols = 256, seed = 0, device = "cpu"):
    return random_packed(T.Q4_K, rows, cols, seed = seed, special = False, device = device), (rows, cols)


def test_cpu_autocast_output_dtype_matches_under_compile():
    # Eager ran F.linear under autocast (bf16 out) while the op's fake declared x's dtype (float32 compiled).
    packed, shape = _weight()
    x = torch.randn(3, 256, requires_grad = True)
    fn = lambda x: gguf_linear(x, packed, T.Q4_K, shape)
    compiled = torch.compile(fn, backend = "aot_eager", fullgraph = True, dynamic = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        eager = fn(x)
        traced = compiled(x)
    assert eager.dtype == traced.dtype == torch.bfloat16
    torch.testing.assert_close(eager.float(), traced.float())
    eager.float().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    traced.float().sum().backward()
    torch.testing.assert_close(grad, x.grad)


def test_cpu_autocast_keeps_a_float64_input():
    # Autocast leaves float64 alone: so does the entry point (dequantize + F.linear returned float64 before).
    packed, shape = _weight()
    x = torch.randn(3, 256, dtype = torch.float64)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        y = gguf_linear(x, packed, T.Q4_K, shape)
    assert y.dtype == torch.float64
    w = dequantize_gguf(packed, T.Q4_K, shape, torch.float64)
    torch.testing.assert_close(y, x @ w.T)


def test_bias_of_another_dtype_outside_autocast():
    packed, shape = _weight()
    x = torch.randn(5, 256, dtype = torch.bfloat16)
    bias = torch.randn(32, dtype = torch.float32)
    y = gguf_linear(x, packed, T.Q4_K, shape, bias)
    w = dequantize_gguf(packed, T.Q4_K, shape, torch.bfloat16)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, torch.nn.functional.linear(x, w, bias.to(torch.bfloat16)))


class _OutOfResources(RuntimeError):
    pass


_OutOfResources.__name__ = "OutOfResources"


class _FakeMMKernel:
    """Raises OutOfResources for tiles taller than `fits` rows, as Triton does on a device without the shared memory."""

    def __init__(self, fits):
        self.fits, self.tried = fits, []

    def __getitem__(self, grid):
        def launch(*args, BLOCK_M, **kwargs):
            self.tried.append(BLOCK_M)
            if BLOCK_M > self.fits:
                raise _OutOfResources(f"out of resource: shared memory (BLOCK_M = {BLOCK_M})")
        return launch


@pytest.mark.parametrize("grouped", (False, True))
def test_tile_that_does_not_fit_retries_smaller_and_remembers(monkeypatch, grouped):
    fake = _FakeMMKernel(fits = 32)
    monkeypatch.setattr(gguf_ops, "_tk", SimpleNamespace(mm_kernel = fake))
    monkeypatch.setattr(gguf_ops, "_MM_BLOCK_M_CAP", {}, raising = False)
    packed, (N, K) = _weight()
    M, E = 128, 1
    x2 = torch.zeros(M, K, dtype = torch.float16)
    out = torch.empty(M, N, dtype = torch.float16)
    counts = torch.tensor([M], dtype = torch.int32) if grouped else None
    for _ in range(2):
        assert gguf_ops._mm_launch(x2, packed, T.Q4_K, out, None, counts, M, N, K, E, N, True, grouped) is out
    # 128 and 64 do not fit, 32 does; the second call starts at 32.
    assert fake.tried == [128, 64, 32, 32]
    other = _FakeMMKernel(fits = 0)
    monkeypatch.setattr(gguf_ops, "_tk", SimpleNamespace(mm_kernel = other))
    with pytest.raises(RuntimeError, match = "out of resource"):
        gguf_ops._mm_launch(x2, packed, T.Q6_K, out, None, counts, M, N, K, E, N, True, grouped)
    assert other.tried == [128, 64, 32, 16]


def test_old_triton_on_blackwell_keeps_tl_dot_off(monkeypatch):
    # Triton < 3.3 aborts the process lowering tl.dot on sm_100 / sm_120.
    x = SimpleNamespace(is_cuda = True, dtype = torch.bfloat16, device = torch.device("cuda", 0))
    monkeypatch.setattr(gguf_ops, "_HAS_TRITON", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (12, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    for old, expected in ((True, False), (False, True)):
        monkeypatch.setattr(gguf_ops, "_OLD_TRITON", old)
        monkeypatch.setattr(gguf_ops, "_DOT_LOWERS", {}, raising = False)
        monkeypatch.setattr(gguf_ops, "_NATIVE_BF16", {})
        assert gguf_ops._fused_ok(x, T.Q4_K) is expected


@cuda
@pytest.mark.parametrize("rows", (1, 5, 40))
def test_strided_packed_view_reads_its_own_bytes(rows):
    # packed[::2] has the logical (rows, row_bytes) shape but a doubled row stride: the fused kernels index rows by
    # row_bytes, so they must see a contiguous copy.
    full, (_, K) = _weight(rows = 64, device = "cuda")
    strided = full[::2]
    assert not strided.is_contiguous()
    shape = (32, K)
    x = torch.randn(rows, K, device = "cuda", dtype = torch.bfloat16)
    g = torch.randn(rows, 32, device = "cuda", dtype = torch.bfloat16)
    want = gguf_linear(x, strided.contiguous(), T.Q4_K, shape)
    got = gguf_linear(x, strided, T.Q4_K, shape)
    assert torch.equal(got, want)
    assert torch.equal(
        gguf_linear_backward_input(g, strided, T.Q4_K, shape),
        gguf_linear_backward_input(g, strided.contiguous(), T.Q4_K, shape),
    )


@cuda
def test_cuda_autocast_output_dtype_does_not_depend_on_rows():
    # The fused kernels wrote x's dtype, dequantize + F.linear the autocast dtype: fp16 in, bf16 autocast.
    packed, shape = _weight(device = "cuda")
    dtypes = set()
    for rows in (1, 5, 40, 300):
        x = torch.randn(rows, 256, device = "cuda", dtype = torch.float16)
        with torch.autocast("cuda", dtype = torch.bfloat16):
            y = gguf_linear(x, packed, T.Q4_K, shape)
        dtypes.add(y.dtype)
        w = dequantize_gguf(packed, T.Q4_K, shape, torch.bfloat16)
        torch.testing.assert_close(y.float(), (x.to(torch.bfloat16) @ w.t()).float(), atol = 5e-2, rtol = 5e-2)
    assert dtypes == {torch.bfloat16}
