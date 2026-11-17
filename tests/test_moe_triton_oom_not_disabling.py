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
"""An OOM inside a Triton MoE kernel call reaches the caller and leaves the kernels enabled."""
import pytest
import torch

import unsloth_zoo.temporary_patches.moe_triton_kernels as mk


def test_disable_reraises_oom(monkeypatch):
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        mk._disable("nf4_dequant_triton", torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert mk._DISABLED_REASON is None


def test_disable_still_disables_on_a_kernel_fault(monkeypatch):
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
    mk._disable("nf4_dequant_triton", RuntimeError("no compiler"))
    assert mk._DISABLED_REASON is not None and "no compiler" in mk._DISABLED_REASON
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_nf4_dequant_oom_propagates(monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    if not mk.moe_triton_kernels_available():
        pytest.skip("Triton unavailable")
    monkeypatch.setattr(mk, "_DISABLED_REASON", None)
    w = torch.randn(4, 64, 128, device = "cuda", dtype = torch.bfloat16)
    q, qs = bnb.functional.quantize_4bit(w, blocksize = 64, quant_type = "nf4")
    real_empty = torch.empty

    def oom_empty(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 64 KiB")

    monkeypatch.setattr(torch, "empty", oom_empty)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        mk.nf4_dequant_triton(q, qs)
    monkeypatch.setattr(torch, "empty", real_empty)
    assert mk._DISABLED_REASON is None and mk.moe_triton_kernels_available()
