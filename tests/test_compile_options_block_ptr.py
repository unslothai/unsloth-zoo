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

"""`triton.use_block_ptr` is only requested where Triton still has block pointers (Triton 3.9 / torch 2.15
removed them; Inductor then ignores the flag and raises a FutureWarning at codegen)."""

import sys
import types
import warnings

import pytest
import torch

from unsloth_zoo.temporary_patches import common

KEY = "triton.use_block_ptr"


def _has_key():
    return all(part in common.inductor_config_source for part in KEY.split("."))


def _installed_block_ptr():
    try:
        import triton.language as tl
    except Exception:
        return False
    return hasattr(tl, "make_block_ptr") and hasattr(tl, "advance")


def test_requested_block_ptr_follows_the_installed_triton():
    if not _has_key():
        pytest.skip("this Inductor has no triton.use_block_ptr")
    assert common.get_torch_compile_options(use_block_ptr = True)[KEY] is _installed_block_ptr()
    assert common.get_torch_compile_options(use_block_ptr = False)[KEY] is False


@pytest.mark.parametrize("has_advance", [True, False])
def test_probe_without_torch_helper(monkeypatch, has_advance):
    # torch < 2.15 has no torch.utils._triton.has_triton_block_ptr: probe triton.language directly.
    import torch.utils._triton as torch_triton
    monkeypatch.delattr(torch_triton, "has_triton_block_ptr", raising = False)
    tl = types.SimpleNamespace(make_block_ptr = object())
    if has_advance:
        tl.advance = object()
    triton = types.ModuleType("triton")
    triton.language = tl
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    assert common._triton_block_ptr_ok() is has_advance


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU for Triton codegen")
def test_block_ptr_options_compile_without_future_warning():
    from unsloth_zoo.temporary_patches.gpt_oss import fused_torch_compile_options

    options = {k: v for k, v in fused_torch_compile_options.items() if k != "triton.cudagraphs"}
    f = torch.compile(lambda x: (x * 2 + 1).sum(-1), options = options, fullgraph = True)
    x = torch.randn(64, 128, device = "cuda")
    torch._dynamo.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = f(x)
    torch.testing.assert_close(out, (x * 2 + 1).sum(-1))
