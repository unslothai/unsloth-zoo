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

"""Stub ``numpy._core.tests._natype`` when the test sub-package is missing.

Some slim numpy 2.x wheels ship without the ``tests/`` sub-package, but
``numpy/testing/_private/utils.py`` has a top-level
``from numpy._core.tests._natype import pd_NA`` that mandates it. Importing
``numpy.testing`` then raises ``ModuleNotFoundError``, which cascades through
``scipy.linalg`` -> ``scipy.optimize`` -> ``transformers.generation`` ->
any ``trust_remote_code`` modeling file that pulls in ``GenerationMixin``
(e.g. ``deepseek-ai/DeepSeek-OCR``). Unsloth's loader catches the exception
in a bare ``except Exception`` and surfaces the misleading
``Unsloth: No config file found - are you sure the model_name is correct?``.

Install a lightweight stub when (and only when) the real package is absent.
The stub sets ``pd_NA = None``; ``numpy.testing`` only uses it for
``assert_array_equal`` deep-comparison, which always compares with ``is``,
so a ``None`` sentinel is functionally equivalent on the training path.

Idempotent: safe to import twice. Cross-platform: pure stdlib.
"""

import sys
import types

__all__ = ["install_numpy_natype_stub"]


def install_numpy_natype_stub() -> bool:
    if "numpy._core.tests._natype" in sys.modules:
        return True
    try:
        import numpy._core  # noqa: F401
    except Exception:
        # numpy not importable at all. Nothing to stub; let downstream
        # imports fail with their native error.
        return False
    try:
        import numpy._core.tests  # noqa: F401 type: ignore[import-not-found]
        try:
            import numpy._core.tests._natype  # noqa: F401 type: ignore[import-not-found]
            return True  # real module exists, no stub needed.
        except Exception:
            pass
    except Exception:
        # numpy._core.tests missing entirely; create a placeholder package
        # so the _natype submodule has a parent.
        tests_pkg = types.ModuleType("numpy._core.tests")
        tests_pkg.__path__ = []  # mark as namespace pkg
        sys.modules["numpy._core.tests"] = tests_pkg
    if "numpy._core.tests._natype" not in sys.modules:
        natype = types.ModuleType("numpy._core.tests._natype")
        # pd_NA is the pandas-NA sentinel that numpy.testing checks via
        # `is pd_NA` only. A None placeholder is functionally equivalent
        # for the normal training path.
        natype.pd_NA = None
        sys.modules["numpy._core.tests._natype"] = natype
    return True


install_numpy_natype_stub()
