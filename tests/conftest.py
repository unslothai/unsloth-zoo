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

"""pytest conftest for the MLX simulation test suite.

Adds ``tests/`` to ``sys.path`` so ``from mlx_simulation import ...``
resolves the bundled torch-on-MLX shim that powers the rest of this
suite. The shim is opt-in test infrastructure: it activates only when a
test calls ``simulate_mlx_on_torch()`` and never touches production
imports of ``unsloth_zoo``.
"""

from __future__ import annotations

import pathlib
import sys


_TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
