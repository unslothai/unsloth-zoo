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

"""Stdlib only (imported by the MLX path on Mac, where torch may be absent)."""

import os

__all__ = ["precreate_private_temp"]


def precreate_private_temp(temporary, final):
    """On a re-export over an existing file, create the staging temp at 0600 before the writer fills it, so weights
    the user restricted are never readable at the umask default while written (writers that open with truncate keep
    the mode; the caller copies the old file's mode on before os.replace). First exports and Windows are unchanged."""
    if os.name == "nt" or not os.path.isfile(str(final)):
        return
    temporary = str(temporary)
    try:
        os.unlink(temporary)  # a leftover temp may be wider, or already open elsewhere
    except FileNotFoundError:
        pass
    except OSError:
        return
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError:
        return
    os.close(fd)
