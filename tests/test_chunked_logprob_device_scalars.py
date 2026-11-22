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


"""`_device_scalars` stages its values on the CPU even when torch's default device is not the CPU."""

import torch

from unsloth_zoo.fused_losses import chunked_logprob as cl


def test_staging_tensor_is_on_the_cpu_under_another_default_device():
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        out = cl._device_scalars([1.0, 2.0, 0.5], torch.device("cpu"))
    finally:
        torch.set_default_device(previous)
    assert out.device.type == "cpu"
    assert out.tolist() == [1.0, 2.0, 0.5]
