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

"""Tests that simulate a CUDA build by setting torch.version.hip = None must also drop a +rocm version tag:
is_rocm_torch falls back to that tag, so on a real ROCm host the simulated build would still read as ROCm."""


def drop_rocm_tag(monkeypatch, torch_mod):
    version = str(torch_mod.__version__)
    if "rocm" in version.lower():
        monkeypatch.setattr(
            torch_mod, "__version__", torch_mod.torch_version.TorchVersion(version.split("+")[0])
        )
