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

from unittest import mock

import torch

from unsloth_zoo import tiled_mlp


def _fake_cuda(free_bytes):
    fake = mock.MagicMock()
    fake.is_available.return_value = True
    fake.mem_get_info.return_value = (free_bytes, free_bytes * 2)
    return fake


def test_target_gb_cuda_uses_free_memory():
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), \
         mock.patch.object(torch, "cuda", _fake_cuda(8 * 1024 ** 3)):
        assert tiled_mlp._default_target_gb() == 4.0  # half of 8 GB free


def test_target_gb_xpu_uses_free_memory_not_constant():
    # Regression: on a small XPU the budget must track free memory, not a fixed
    # 4 GB that could exceed what is left and OOM the tiler.
    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    fake_xpu = mock.MagicMock()
    fake_xpu.is_available.return_value = True
    fake_xpu.mem_get_info.return_value = (2 * 1024 ** 3, 4 * 1024 ** 3)
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", fake_xpu, create=True):
        assert tiled_mlp._default_target_gb() == 1.0  # half of 2 GB free, not 4.0


def test_target_gb_xpu_unsupported_falls_back_to_host():
    # Some Intel GPUs (Arc B580, Lunar Lake) raise on mem_get_info; must not crash.
    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    fake_xpu = mock.MagicMock()
    fake_xpu.is_available.return_value = True
    fake_xpu.mem_get_info.side_effect = RuntimeError(
        "The device doesn't support querying the available free memory."
    )
    fake_vm = mock.MagicMock(available=6 * 1024 ** 3)
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "xpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch.object(torch, "xpu", fake_xpu, create=True), \
         mock.patch("psutil.virtual_memory", return_value=fake_vm):
        assert tiled_mlp._default_target_gb() == 3.0  # half of 6 GB host RAM


def test_target_gb_cpu_uses_host_memory():
    fake_cuda = mock.MagicMock()
    fake_cuda.is_available.return_value = False
    fake_vm = mock.MagicMock(available=10 * 1024 ** 3)
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cpu"), \
         mock.patch.object(torch, "cuda", fake_cuda), \
         mock.patch("psutil.virtual_memory", return_value=fake_vm):
        assert tiled_mlp._default_target_gb() == 5.0  # half of 10 GB host RAM
