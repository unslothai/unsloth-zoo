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

"""The gather_qmm guard's chain walk must terminate against a permissive `mx`.

`is_gather_qmm_nax_guard_applied` walks `_unsloth_index_original` until it reads
None. That terminates for a real mlx function, but a stand-in that answers every
attribute (the mlx test shim, a Mock) never yields None, and the unbounded walk
hangs the interpreter rather than failing. This pins the bound.
"""

import threading

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _utils():
    """Imported lazily: unsloth_zoo.mlx.utils imports mlx.core at module scope."""
    from unsloth_zoo.mlx import utils as mlx_utils

    return mlx_utils


class _Permissive:
    """Answers every attribute with a fresh falsy instance, like the mlx shim."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Permissive()

    def __bool__(self):
        return False


def test_guard_chain_walk_terminates_against_a_permissive_mx(monkeypatch):
    mlx_utils = _utils()
    monkeypatch.setattr(mlx_utils.mx, "gather_qmm", _Permissive(), raising = False)

    result = {}

    def _run():
        result["value"] = mlx_utils.is_gather_qmm_nax_guard_applied()

    # Run off-thread: an unbounded walk does not raise, it never returns, so the
    # only way to report it as a failure instead of a hung job is to stop waiting.
    worker = threading.Thread(target = _run, daemon = True)
    worker.start()
    worker.join(timeout = 10)

    assert not worker.is_alive(), (
        "is_gather_qmm_nax_guard_applied did not terminate against an mx whose "
        "attributes never run out: the chain walk needs a bound, not just a None check"
    )
    assert result["value"] is False


def test_guard_chain_walk_still_finds_a_guard_under_a_wrapper(monkeypatch):
    """The bound must not cost the real lookup it exists for."""

    def _guarded():
        pass

    _guarded._unsloth_gather_qmm_guard = True

    def _wrapper():
        pass

    _wrapper._unsloth_index_original = _guarded

    mlx_utils = _utils()
    monkeypatch.setattr(mlx_utils.mx, "gather_qmm", _wrapper, raising = False)
    assert mlx_utils.is_gather_qmm_nax_guard_applied() is True

    mlx_utils = _utils()
    monkeypatch.setattr(mlx_utils.mx, "gather_qmm", _guarded, raising = False)
    assert mlx_utils.is_gather_qmm_nax_guard_applied() is True
