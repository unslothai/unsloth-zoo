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

"""An OOM in the fp16-in / fp32-out GEMM must not be cached as "out_dtype unsupported" for the process. CPU."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg


@pytest.mark.parametrize("error, cached", [(torch.OutOfMemoryError("CUDA out of memory"), None), (RuntimeError("no out_dtype"), False)])
def test_mm_out_dtype_probe(monkeypatch, error, cached):
    monkeypatch.setattr(mg, "_MM_OUT_DTYPE", {})
    real_mm = torch.mm

    def mm(*args, **kwargs):
        if "out_dtype" in kwargs:
            raise error
        return real_mm(*args, **kwargs)

    monkeypatch.setattr(mg.torch, "mm", mm)
    x, w = torch.randn(4, 8, dtype = torch.float16), torch.randn(8, 3, dtype = torch.float16)
    if cached is None:
        with pytest.raises(torch.OutOfMemoryError):
            mg._mm(x, w, torch.float32)
    else:
        y = mg._mm(x, w, torch.float32)
        torch.testing.assert_close(y, x.float() @ w.float())
    assert mg._MM_OUT_DTYPE.get(torch.float16) is cached
