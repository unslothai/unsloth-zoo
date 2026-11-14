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
"""UNSLOTH_MOE_NF4_FUSED=auto on CUDA: below the cap the fused kernel, above it the dequant path bit for bit,
forward and dX, and one compiled frame across the cap (two with the Python-side choice)."""
import pytest
import torch

if (
    not torch.cuda.is_available()
    or getattr(torch.version, "hip", None) is not None
    or torch.cuda.get_device_capability()[0] < 8
):
    pytest.skip("NF4 fused auto needs NVIDIA sm80+ (not ROCm)", allow_module_level = True)
bnb = pytest.importorskip("bitsandbytes")
from bitsandbytes.nn import Params4bit  # noqa: E402

from unsloth_zoo.temporary_patches import moe_nf4_grouped as nf  # noqa: E402

E_, K_, N_ = 8, 256, 128


def _stack():
    torch.manual_seed(0)
    w = torch.randn(E_, K_, N_, dtype = torch.bfloat16) * 0.05
    p = Params4bit(w, requires_grad = False, quant_type = "nf4", compress_statistics = True, blocksize = 64).to("cuda")
    p._original_shape = torch.Size((E_, K_, N_))
    return p


def _ends(rows):
    return (torch.arange(1, E_ + 1, device = "cuda", dtype = torch.int32) * rows // E_).to(torch.int32)


@pytest.fixture
def auto(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_NF4_FUSED", "auto")
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "4")   # 32 rows
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTO_IN_OP", raising = False)


def test_auto_matches_each_path_forward_and_dx(auto):
    p = _stack()
    spec = nf.nf4_grouped_spec(p, lambda w: w, torch.bfloat16, mode = "1")
    deq = bnb.functional.dequantize_4bit(p.data, p.quant_state).view(E_, K_, N_)
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    for rows in (16, 32, 40, 256):
        x = torch.randn(rows, K_, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        dy = torch.randn(rows, N_, device = "cuda", dtype = torch.bfloat16)
        ends = _ends(rows)
        y = nf.maybe_nf4_grouped_mm(x, ends, p, lambda w: w)
        y.backward(dy)
        if rows <= 32:
            ref_y = nf.nf4_grouped_mm(x.detach(), ends, spec)
            ref_dx = torch.ops.unsloth_zoo.nf4_grouped_mm_dx(dy, spec["packed"], spec["absmax"], spec["code"],
                                                            spec["absmax2"], spec["code2"], spec["offset"], ends, K_, N_,
                                                            True, 64, spec["blocksize2"], True, spec["w_dtype"])
        else:
            with torch.no_grad():
                ref_y = _grouped_mm_with_backward_fix(x.detach(), deq, ends)
                ref_dx = _grouped_mm_with_backward_fix(dy, deq.transpose(-2, -1), ends)
        assert torch.equal(y, ref_y), rows
        assert torch.equal(x.grad, ref_dx), rows


@pytest.mark.parametrize("in_op", ["1", "0"])
def test_crossing_the_cap_compiles(auto, monkeypatch, in_op):
    """Rows below the cap first (automatic dynamic settles), then above: in-op adds no frame, Python-side does."""
    from torch._dynamo.testing import CompileCounter
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_IN_OP", in_op)
    p = _stack()
    torch._dynamo.reset()
    cnt = CompileCounter()

    def f(x, ends):
        y = nf.maybe_nf4_grouped_mm(x, ends, p, lambda w: w)
        return x[:, :N_] * 0 if y is None else y

    g = torch.compile(f, backend = cnt, dynamic = True)
    for rows in (16, 24, 8):
        g(torch.randn(rows, K_, device = "cuda", dtype = torch.bfloat16), _ends(rows))
    below = cnt.frame_count
    for rows in (64, 128, 16, 256):
        g(torch.randn(rows, K_, device = "cuda", dtype = torch.bfloat16), _ends(rows))
    if in_op == "1":
        assert cnt.frame_count == below
    else:
        assert cnt.frame_count > below
