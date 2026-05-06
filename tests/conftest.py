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
