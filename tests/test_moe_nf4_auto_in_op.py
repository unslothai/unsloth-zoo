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
"""UNSLOTH_MOE_NF4_FUSED=auto chooses fused vs dequant inside one opaque op: crossing the row cap never recompiles."""
import pytest
import torch

pytest.importorskip("triton")
from unsloth_zoo.temporary_patches import moe_nf4_grouped as ng  # noqa: E402

if ng._AUTO_OP is None:
    pytest.skip("torch.library.custom_op unavailable", allow_module_level = True)

E_, K_, N_ = 4, 64, 32


def _stack(nested):
    torch.manual_seed(0)
    numel = E_ * K_ * N_
    packed = torch.randint(0, 256, (numel // 2,), dtype = torch.uint8)
    if nested:
        absmax = torch.randint(0, 256, (numel // 64,), dtype = torch.uint8)
        code2 = torch.linspace(-1, 1, 256)
        absmax2 = torch.rand(1) + 0.5
        offset = torch.tensor([0.01])
    else:
        absmax = torch.rand(numel // 64) + 0.5
        code2 = absmax2 = offset = absmax
    code = torch.linspace(-1, 1, 16)
    return packed, absmax, code, absmax2, code2, offset


@pytest.fixture
def cpu_paths(monkeypatch):
    """Both branches on CPU: the fused kernel's torch twin and a dequant + per-expert matmul."""
    seen = []

    def fused(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j, bs, bs2, nested, w_dtype):
        seen.append("fused")
        return ng._launch_eager(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j,
                                bs, bs2, nested, w_dtype)

    def dequant(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j, bs, bs2, nested, w_dtype):
        seen.append("dequant")
        return ng._launch_eager(a, packed, absmax, code2, absmax2, offset, code, ends, inner, outer, s_i, s_j,
                                bs, bs2, nested, w_dtype)

    monkeypatch.setattr(ng, "_launch", fused)
    monkeypatch.setattr(ng, "_dequant_grouped_mm", dequant)
    return seen


def _call(x, st, ends, max_rows):
    packed, absmax, code, absmax2, code2, offset = st
    nested = absmax.dtype == torch.uint8
    return ng._AUTO_OP(x, packed, absmax, code, absmax2, code2, offset, ends, K_, N_, True, 64,
                       256 if nested else 1, nested, 2, max_rows)


@pytest.mark.parametrize("nested", [True, False])
def test_branch_follows_runtime_rows(cpu_paths, nested):
    st = _stack(nested)
    for rows, want in ((8, "fused"), (16, "fused"), (17, "dequant"), (64, "dequant")):
        x = torch.randn(rows, K_)
        ends = torch.linspace(rows // E_, rows, E_).round().to(torch.int32)
        ends[-1] = rows
        cpu_paths.clear()
        y = _call(x, st, ends, 4)   # cap 4 rows per expert = 16 rows
        assert cpu_paths == [want] and y.shape == (rows, N_)


def test_backward_uses_the_same_rule(cpu_paths):
    st = _stack(True)
    for rows, want in ((12, "fused"), (40, "dequant")):
        x = torch.randn(rows, K_, requires_grad = True)
        ends = torch.tensor([rows // 4, rows // 2, 3 * rows // 4, rows], dtype = torch.int32)
        cpu_paths.clear()
        _call(x, st, ends, 4).sum().backward()
        assert cpu_paths == [want, want] and x.grad.shape == x.shape


def test_crossing_the_cap_does_not_recompile(cpu_paths):
    from torch._dynamo.testing import CompileCounter
    torch._dynamo.reset()
    st = _stack(True)
    cnt = CompileCounter()

    @torch.compile(backend = cnt, dynamic = True)
    def f(x, ends):
        return _call(x, st, ends, 4) * 2

    for rows in (8, 12, 40, 64, 9):
        x = torch.randn(rows, K_)
        ends = torch.tensor([rows // 4, rows // 2, 3 * rows // 4, rows], dtype = torch.int32)
        f(x, ends)
    assert cnt.frame_count == 1
    assert set(cpu_paths) == {"fused", "dequant"}


def test_python_side_choice_would_recompile():
    """The control: the shipped Python-side `rows <= cap * E` guard compiles one frame per side of the cap."""
    from torch._dynamo.testing import CompileCounter
    torch._dynamo.reset()
    cnt = CompileCounter()

    @torch.compile(backend = cnt, dynamic = True)
    def f(x):
        if ng.nf4_auto_fused(x.shape[0], E_):
            return x * 2
        return x * 3

    for rows in (8, 12, 2048, 4096):
        f(torch.randn(rows, K_))
    assert cnt.frame_count == 2


def test_in_op_switch(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTO_IN_OP", raising = False)
    assert ng.nf4_auto_in_op() is True
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_IN_OP", "0")
    assert ng.nf4_auto_in_op() is False


def test_cap_by_capability(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", raising = False)
    cuda = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d = None: (8, 0))
    assert ng.nf4_auto_max_rows(cuda) == 32
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d = None: (8, 9))
    assert ng.nf4_auto_max_rows(cuda) == 128
    assert ng.nf4_auto_max_rows() == ng.AUTO_MAX_ROWS
    monkeypatch.setenv("UNSLOTH_MOE_NF4_AUTO_MAX_ROWS", "32")
    assert ng.nf4_auto_max_rows(cuda) == 32
