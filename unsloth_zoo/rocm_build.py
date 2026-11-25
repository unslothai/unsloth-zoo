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
    """ROCm torch build. AMD SDK / Radeon wheels (torch 2.11.0+rocm7.14.0 from repo.amd.com) leave
    torch.version.hip unset and carry the ROCm tag only in torch.__version__, so check both."""
    if torch_mod is None:
        try:
            import torch as torch_mod
        except Exception:
            return False
    if getattr(getattr(torch_mod, "version", None), "hip", None):
        return True
    return "rocm" in str(getattr(torch_mod, "__version__", "") or "").lower()
