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

"""The mlx shim fixtures must hand the next test file an intact import graph.

test_mlx_preference.py and test_mlx_trainer_internals.py drop every
unsloth_zoo.mlx.* module so their shim is the one those modules bind against,
then put the originals back. Restoring sys.modules alone is not enough, and the
gap is not visible from inside the file that causes it: whoever runs next on the
same xdist worker pays for it.

#1131 restored sys.modules and stopped there. It fixed the two failures it was
aimed at and introduced six others -- in test_pr684_review_fixes_a.py,
test_mlx_text_path_contract.py and test_mlx_distributed_loader.py -- because
`import unsloth_zoo.mlx.trainer as t` reads parent package attributes while a
`from unsloth_zoo.mlx.trainer import MLXTrainer` inside the code under test
reads sys.modules. Nothing asserted the two agreed, so the trade went unnoticed.
This file asserts it.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _owned(name):
    prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    return (
        name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
        or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    )


def _walk_parent_attributes(name):
    """Resolve a dotted module name the way `import a.b.c as x` does."""
    parts = name.split(".")
    module = sys.modules.get(parts[0])
    for part in parts[1:]:
        if module is None:
            return None
        module = getattr(module, part, None)
    return module


def test_restore_puts_parent_attributes_back():
    """A drop-and-reimport cycle must leave both lookup paths on one module."""
    from mlx_simulation import restore_modules, snapshot_modules

    import unsloth_zoo.mlx.trainer  # noqa: F401
    saved = snapshot_modules(_owned)
    before = sys.modules["unsloth_zoo.mlx.trainer"]

    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    import unsloth_zoo.mlx.trainer  # noqa: F401,F811
    assert sys.modules["unsloth_zoo.mlx.trainer"] is not before, (
        "the re-import did not build a new module, so this proves nothing"
    )

    restore_modules(saved, _owned)

    assert sys.modules["unsloth_zoo.mlx.trainer"] is before
    assert _walk_parent_attributes("unsloth_zoo.mlx.trainer") is before, (
        "parent attribute still points at the module built under the shim"
    )
    assert _walk_parent_attributes("unsloth_zoo.mlx") is sys.modules["unsloth_zoo.mlx"]


def test_restore_drops_attributes_for_modules_it_cannot_restore():
    """A submodule imported only under the shim leaves no dangling attribute.

    Putting nothing back in sys.modules for it while leaving parent.child bound
    is the same split in reverse: the attribute walk finds a module that no
    sys.modules lookup can ever reach.
    """
    from mlx_simulation import restore_modules, snapshot_modules

    saved = snapshot_modules(_owned)
    fresh = [
        name for name in ("unsloth_zoo.mlx.trainer", "unsloth_zoo.mlx.utils")
        if name not in saved
    ]
    if not fresh:
        for name in list(sys.modules):
            if name.startswith("unsloth_zoo.mlx."):
                sys.modules.pop(name, None)
                saved.pop(name, None)
        fresh = ["unsloth_zoo.mlx.trainer"]

    __import__(fresh[0])
    restore_modules(saved, _owned)

    for name in fresh:
        assert _walk_parent_attributes(name) is sys.modules.get(name), name


def test_the_shim_files_use_the_shared_restore():
    """Both fixtures must go through restore_modules, not sys.modules.update.

    The bare update is what shipped in #1131. Spelled out here so a future edit
    that reintroduces it is a failing test rather than six failures in unrelated
    files one worker over.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for filename in ("test_mlx_preference.py", "test_mlx_trainer_internals.py"):
        source = (here / filename).read_text(encoding = "utf-8")
        assert "restore_modules(" in source, filename
        assert "sys.modules.update(" not in source, (
            f"{filename} restores sys.modules without repairing parent attributes"
        )
