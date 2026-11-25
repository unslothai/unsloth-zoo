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

"""Guards against a cache-folder writer swapping the file between create_new_function(overwrite =
False)'s equality read and rank 0's second digest, which approved and executed its bytes even
without UNSLOTH_COMPILE_OVERWRITE=0; the swap is simulated by wrapping the decision."""
import os
import sys
import tempfile

import pytest

compiler = pytest.importorskip("unsloth_zoo.compiler")

_MARKER = "UNSLOTH_TEST_RETAINED_SWAP_EXECUTED"
_PLANTED = f"import os\nos.environ[{_MARKER!r}] = '1'\ndef probe():\n    return 'planted'\n"
_GENUINE = "def probe():\n    return 'genuine'\n"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    location = tmp_path / "unsloth_compiled_cache"
    location.mkdir()
    (tmp_path / "tempdir").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tempdir"))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(location))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    monkeypatch.setenv(_MARKER, "0")
    yield location
    for name in list(sys.modules):
        if name.startswith("UnslothRetainedSwapProbe"):
            del sys.modules[name]


def _swap_before_decision(monkeypatch, name, cache_dir):
    original = compiler._compiled_cache_decision

    def swapped(function_location, *args, **kwargs):
        # The equality read already passed; another writer replaces the file now.
        planted = cache_dir / f".{name}.planted"
        planted.write_text(_PLANTED)
        os.replace(planted, function_location)
        return original(function_location, *args, **kwargs)

    monkeypatch.setattr(compiler, "_compiled_cache_decision", swapped)


def test_file_swapped_after_equality_check_is_not_executed(cache_dir, monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_OVERWRITE", raising = False)
    name = "UnslothRetainedSwapProbe"
    compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)  # warm, genuine cache
    sys.modules.pop(name, None)
    _swap_before_decision(monkeypatch, name, cache_dir)

    module = compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)

    assert os.environ.get(_MARKER) != "1", "bytes swapped in after the equality check were executed"
    assert module.probe() == "genuine"


def test_overwrite_zero_still_keeps_an_edited_cache(cache_dir, monkeypatch):
    """UNSLOTH_COMPILE_OVERWRITE=0 keeps the user's own edited file, as before."""
    name = "UnslothRetainedSwapProbeEdited"
    compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)
    sys.modules.pop(name, None)
    path = cache_dir / f"{name}.py"
    path.write_text(path.read_text().replace("'genuine'", "'edited'"))
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")

    module = compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)

    assert module.probe() == "edited"


def test_unchanged_warm_cache_is_kept(cache_dir, monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_OVERWRITE", raising = False)
    name = "UnslothRetainedSwapProbeWarm"
    compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)
    sys.modules.pop(name, None)
    path = cache_dir / f"{name}.py"
    before = os.stat(path).st_ino

    module = compiler.create_new_function(name, _GENUINE, "math", [], overwrite = False)

    assert module.probe() == "genuine"
    assert os.stat(path).st_ino == before, "an unchanged warm cache was rewritten"
