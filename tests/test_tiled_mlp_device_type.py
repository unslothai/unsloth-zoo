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
"""TiledMLP's autocast decorators need a device string torch knows.

`torch.amp.custom_fwd` parses its `device_type` as a torch device on the first decorated call.
`DEVICE_TYPE` is the zoo's own label and is "mlx" on Apple silicon with mlx installed, which
torch rejects with "Expected one of cpu, cuda, ... at start of device string: mlx", so every
tiled forward died inside the decorator. `DEVICE_TYPE_TORCH` is the translated spelling
("mlx" -> "mps", "hip" -> "cuda") and is what torch APIs take everywhere else in the package.
"""

import torch

from unsloth_zoo import tiled_mlp


def test_amp_decorators_take_a_torch_device_type():
    for name in ("torch_amp_custom_fwd", "torch_amp_custom_bwd"):
        decorator = getattr(tiled_mlp, name)
        device_type = decorator.keywords["device_type"]
        # The parse torch itself performs, so a label torch does not know fails here first.
        torch.device(device_type)
pass


def test_module_uses_the_translated_device_type():
    """The source must read DEVICE_TYPE_TORCH, so an mlx or hip host is covered off-machine too."""
    source = (tiled_mlp.__file__).replace(".pyc", ".py")
    with open(source, encoding = "utf-8") as handle:
        text = handle.read()
    for call in ("torch.amp.custom_fwd(device_type = DEVICE_TYPE_TORCH)",
                 "torch.amp.custom_bwd(device_type = DEVICE_TYPE_TORCH)"):
        assert call in text, f"{call} missing; an mlx or hip DEVICE_TYPE would reach torch unchanged"
pass
