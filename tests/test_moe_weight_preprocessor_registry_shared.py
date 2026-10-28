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

"""The weight-preprocessor registry is one dict shared by every loaded moe_utils copy, so a registration reaches the unsloth_compiled_cache forward (ref 849)."""

import os
import sys

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils

_CACHED_NAME = "unsloth_cached_moe_utils"


@pytest.fixture(scope="module")
def cache_copy(tmp_path_factory):
    """Loads moe_utils from a scratch compile location, as the cache copy is; any earlier moe_utils module is set aside, leaving one copy as in a real process."""
    patch = pytest.MonkeyPatch()
    patch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path_factory.mktemp("compile_location")))
    moe_utils.install_to_cache(moe_utils.__file__, "moe_utils.py")
    patch.setattr(moe_utils, "_CACHED_MOE_UTILS_MODULE", None)
    patch.setattr(moe_utils, "_CACHED_FORWARD_MOE_BACKEND", None)
    previous = sys.modules.pop(_CACHED_NAME, None)
    try:
        copy = moe_utils._load_cached_moe_utils_module()
        assert copy is not None, "the scratch cache copy did not load"
        assert copy is not moe_utils
        yield copy
    finally:
        sys.modules.pop(_CACHED_NAME, None)
        if previous is not None:
            sys.modules[_CACHED_NAME] = previous
        patch.undo()


def test_backend_resolves_to_the_cache_copy(cache_copy):
    # Premise: the forward runs with the cache copy's globals, so a package-local registry is never consulted by it.
    forward = moe_utils.get_forward_moe_backend()
    assert forward is cache_copy.forward_moe_backend
    assert forward.__globals__ is cache_copy.__dict__
    assert cache_copy.preprocess_weight is not moe_utils.preprocess_weight


def test_both_copies_resolve_one_registry(cache_copy):
    assert cache_copy._weight_preprocessor_registry() is moe_utils._WEIGHT_PREPROCESSORS
    assert moe_utils._weight_preprocessor_registry() is moe_utils._WEIGHT_PREPROCESSORS


def test_package_registration_is_visible_from_the_cache_copy(cache_copy, capsys):
    key = "unit_test_registry_shared_arch"
    sentinel = torch.zeros(1)
    fn = lambda weight, proj_type, hidden_dim: sentinel
    moe_utils.register_weight_preprocessor(key, fn)
    try:
        assert cache_copy.get_weight_preprocessor(key) is fn
        # Square weight with no experts_module: unregistered it guesses the layout (ref 849); registered, it dispatches before any shape logic.
        out = cache_copy.preprocess_weight(torch.randn(4, 64, 64), "gate_up", 64, model_type=key)
        assert out is sentinel
        assert "ambiguous" not in capsys.readouterr().err.lower()
    finally:
        moe_utils._WEIGHT_PREPROCESSORS.pop(key, None)


def test_cache_copy_registration_lands_in_the_package(cache_copy):
    key = "unit_test_registry_shared_arch_reverse"
    fn = lambda weight, proj_type, hidden_dim: weight
    cache_copy.register_weight_preprocessor(key, fn)
    try:
        assert moe_utils.get_weight_preprocessor(key) is fn
    finally:
        moe_utils._WEIGHT_PREPROCESSORS.pop(key, None)


def test_bare_moe_utils_copy_shares_the_registry(cache_copy, monkeypatch):
    # A third copy: compiler.py puts the compile location on sys.path, so the bare `from moe_utils import ...` in generated modules is its own module too.
    monkeypatch.syspath_prepend(os.environ["UNSLOTH_COMPILE_LOCATION"])
    # Restore by hand, not monkeypatch: delitem on an absent name records nothing and setitem records the bare-import
    # module, so undo reinstates a tmp-dir copy that every later test resolves moe_utils to; this broke
    # test_compiled_cache_collective.py.
    previous = sys.modules.pop("moe_utils", None)
    key = "unit_test_registry_shared_arch_bare"
    fn = lambda weight, proj_type, hidden_dim: weight
    try:
        import moe_utils as bare
        assert bare is not moe_utils and bare is not cache_copy
        moe_utils.register_weight_preprocessor(key, fn)
        assert bare.get_weight_preprocessor(key) is fn
    finally:
        moe_utils._WEIGHT_PREPROCESSORS.pop(key, None)
        sys.modules.pop("moe_utils", None)
        if previous is not None:
            sys.modules["moe_utils"] = previous
