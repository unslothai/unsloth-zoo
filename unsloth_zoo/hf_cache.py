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

"""``unsloth_zoo.hf_cache`` is ``unsloth_zoo_downloads.hf_cache``: one module object under both names,
and the same names for a loader that executes this file by path."""

import sys

from unsloth_zoo_downloads import hf_cache as _module

globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})
if __name__ == "unsloth_zoo.hf_cache":
    sys.modules[__name__] = _module
