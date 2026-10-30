# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from unsloth_zoo.memory_budget import (  # noqa: F401  (re-exported for remote-code saves and external imports)
    _default_target_gb,
    _host_target_gb,
    get_max_flat_qlen,
)
from unsloth_zoo.tiling_registry import install as _install, install_module as _install_module, parse_mode

# Released Unsloth imports these at module level (its DDP patch wraps TiledMLP).
from unsloth_zoo._legacy_tiled_mlp import (  # noqa: F401
    TiledMLP,
    _mxfp4_instance_forward,
    is_custom_module,
    torch_amp_custom_bwd,
    torch_amp_custom_fwd,
)

__all__ = [
    "patch_tiled_mlp",
    "patch_mlp",
    "get_max_flat_qlen",
]


def patch_tiled_mlp(model, patch_options_str = "arctic", padded_length = 128):
    _install(model, parse_mode(patch_options_str, padded_length = padded_length))
    return model


def patch_mlp(mlp_module, target_arctic = True, target_gb = None, padded_length = 128):
    """Old spelling: tile this one module (the caller vouches that it is token-local)."""
    if target_arctic:
        mode = "arctic"
    elif target_gb is not None:
        mode = f"target:{float(target_gb)}"
    else:
        mode = "auto"
    return _install_module(mlp_module, parse_mode(mode, padded_length = padded_length))
