# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""W8A8 backends.

`portable` implements the algorithm in plain MLX ops and runs anywhere, which is what
makes the arithmetic testable without an M5. `metal_mpp` is the real one.
"""

import logging
import os

logger = logging.getLogger(__name__)

_backend = None


def select(name=None):
    """Return the backend module. `name` defaults to UNSLOTH_MLX_INT8_BACKEND, then to
    metal_mpp with a fallback to portable if Metal kernels cannot be imported."""
    global _backend
    if name is None and _backend is not None:
        return _backend

    name = name or os.environ.get("UNSLOTH_MLX_INT8_BACKEND", "metal_mpp")
    if name == "portable":
        from . import portable as mod
    elif name == "metal_mpp":
        from . import metal_mpp as mod
    else:
        raise ValueError(f"unknown backend {name!r}")

    _backend = mod
    return mod


def reset():
    global _backend
    _backend = None
