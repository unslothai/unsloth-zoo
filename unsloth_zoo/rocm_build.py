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

__all__ = ["is_rocm_torch"]


def is_rocm_torch(torch_mod = None):
    """ROCm torch build: torch.version.hip, or a +rocm tag in torch.__version__ as a fallback for a build that
    leaves hip unset. The AMD SDK / Radeon gfx1151 wheels checked so far (torch 2.11.0+rocm7.13.0) do set hip."""
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    if getattr(getattr(torch_mod, "version", None), "hip", None):
        return True
    # Never str() a TorchVersion (torch.__version__): Dynamo cannot trace its C str() under fullgraph. A plain str
    # there (set by a caller or test) wins, else the plain-str torch.version.__version__.
    version = getattr(torch_mod, "__version__", None)
    if type(version) is not str:
        version = getattr(getattr(torch_mod, "version", None), "__version__", None)
    if type(version) is not str:
        version = str(version or "")
    return "rocm" in version.lower()
