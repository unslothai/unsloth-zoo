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

__all__ = [
    "is_mlx_available",
]

import functools
import importlib.util
import os
import platform


@functools.cache
def is_mlx_available() -> bool:
    return (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") != "1"
        and platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )
