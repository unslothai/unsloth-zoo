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

"""Sparse attention dispatch across Triton / torch versions and under torch.compile: gather is only selected where
its Triton kernels import, and traced dispatch does not recompile on lazily-filled state."""

import os
import sys

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo import sparse_attention as sa  # noqa: E402

_VENDORED_TRITON = ("unsloth_zoo._vendored.attn_gym._backends.triton",
                    "unsloth_zoo._vendored.attn_gym.sparse.gather_attn.impl.triton")


def test_gather_needs_its_triton_kernels_not_only_the_api(monkeypatch):
    # torch 2.4 - 2.8 bundle Triton 3.0 - 3.4, which lack triton.constexpr_function: the api and op registrations
    # import there, the kernels (loaded on the first call) do not, so gather must not be selected.
    triton = pytest.importorskip("triton")
    monkeypatch.setattr(sa, "_GATHER", None)
    if hasattr(triton, "constexpr_function"):
        assert sa._gather_importable()   # api + ops cached; their op definitions never run twice below
        monkeypatch.delattr(triton, "constexpr_function")
        for name in [m for m in sys.modules if m.startswith(_VENDORED_TRITON)]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sa, "_GATHER", None)
    assert sa._gather_importable() is False
