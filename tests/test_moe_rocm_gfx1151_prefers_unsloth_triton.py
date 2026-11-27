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

"""On ROCm gfx1151 the auto MoE backend is unsloth_triton even when torch._grouped_mm passes the training probe
(there it is a synced per-expert loop); other archs and an explicit UNSLOTH_MOE_BACKEND keep their choice. CPU only."""

import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils as mu


@pytest.fixture
def host(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_BACKEND", raising = False)
    monkeypatch.setattr(mu, "_check_torch_grouped_mm_supported", lambda *a, **k: True)
    monkeypatch.setattr(mu, "_check_grouped_gemm_available", lambda *a, **k: True)
    monkeypatch.setattr(mu, "_select_grouped_mm_on_triton", lambda *a, **k: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def make(rocm, arch):
        monkeypatch.setattr(mu, "is_rocm_torch", lambda *a, **k: rocm)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(gcnArchName = arch))
    return make


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1151:sramecc-:xnack-"])
def test_gfx1151_auto_takes_unsloth_triton(host, arch):
    host(True, arch)
    assert mu._select_moe_backend() == "unsloth_triton"


@pytest.mark.parametrize("rocm, arch", [(True, "gfx942:sramecc+:xnack-"), (True, "gfx1100"), (False, "")])
def test_other_archs_keep_grouped_mm(host, rocm, arch):
    host(rocm, arch)
    assert mu._select_moe_backend() == "grouped_mm"


def test_explicit_grouped_mm_wins_on_gfx1151(host, monkeypatch):
    host(True, "gfx1151")
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "grouped_mm")
    assert mu._select_moe_backend() == "grouped_mm"


def test_gfx1151_without_the_triton_kernels_keeps_grouped_mm(host, monkeypatch):
    host(True, "gfx1151")
    monkeypatch.setattr(mu, "_check_grouped_gemm_available", lambda *a, **k: False)
    assert mu._select_moe_backend() == "grouped_mm"
