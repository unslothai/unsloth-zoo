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


"""CPU checks of the GEMV layout ROCm takes: the one-row Q4_1 GEMV uses the flat layout there (gfx1151 Linux and
Windows decode its split layout wrongly). NVIDIA keeps the layout it had for every format."""

import pytest

from unsloth_zoo import rocm_build
from unsloth_zoo.gguf_kernels import GGML_BLOCK
from unsloth_zoo.gguf_kernels import GGMLType as T
from unsloth_zoo.gguf_kernels import ops as O


@pytest.fixture(params = [True, False], ids = ["rocm", "cuda"])
def rocm(request, monkeypatch):
    monkeypatch.setattr(rocm_build, "is_rocm_torch", lambda torch_mod = None: request.param)
    return request.param


def test_q4_1_gemv_layout(rocm):
    assert O._gemv_split(T.Q4_1) is (not rocm)
    for t in (T.Q4_0, T.Q5_0, T.Q5_1, T.Q8_0, T.Q4_K, T.Q6_K, T.IQ4_XS):
        assert O._gemv_split(t) is True, t


def test_cuda_layout_unchanged_for_every_format(monkeypatch):
    monkeypatch.setattr(rocm_build, "is_rocm_torch", lambda torch_mod = None: False)
    for no_split in (frozenset(), frozenset(GGML_BLOCK)):
        monkeypatch.setattr(O, "_NO_SPLIT_GEMV", no_split)
        for t in GGML_BLOCK:
            assert O._gemv_split(t) is (t not in no_split), t


def test_no_split_list_still_wins(rocm, monkeypatch):
    monkeypatch.setattr(O, "_NO_SPLIT_GEMV", frozenset((T.Q4_0, T.Q4_1)))
    assert O._gemv_split(T.Q4_0) is False and O._gemv_split(T.Q4_1) is False
