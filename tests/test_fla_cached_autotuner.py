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

"""CachedAutotuner.run skips FLA's disk-cache key unless FLA_CACHE_MODE reads the cache (unsloth#10806)."""

import importlib.util
import pathlib
from unittest.mock import patch

import pytest
from packaging import version

pytest.importorskip("torch")
triton = pytest.importorskip("triton")
# fla_vendor._MIN_TRITON: older triton never loads the vendored fla.
if version.parse(triton.__version__.split("+")[0]) < version.parse("3.3"):
    pytest.skip("vendored fla needs triton>=3.3", allow_module_level=True)
tl = triton.language
Autotuner = triton.runtime.autotuner.Autotuner

_CACHE_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo/_vendored/fla/ops/utils/cache.py"


@pytest.fixture(scope="module")
def cache_mod():
    spec = importlib.util.spec_from_file_location("fla_cache_under_test", _CACHE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tuner(cache_mod):
    @triton.jit
    def kernel(x, N: tl.constexpr):
        pass

    configs = [triton.Config({}, num_warps=w) for w in (1, 2)]
    try:
        return cache_mod.fla_cache_autotune(configs=configs, key=["N"])(kernel)
    except RuntimeError as e:  # older triton (3.3) needs an active GPU driver to build an Autotuner
        pytest.skip(f"no triton driver: {e}")


@pytest.mark.parametrize(
    "mode, expect_lookup",
    [("DISABLED", False), ("DEFAULT", True), ("ALWAYS", True), ("STRICT", True)],
)
def test_run_builds_fla_key_only_when_cache_is_read(cache_mod, tuner, monkeypatch, mode, expect_lookup):
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode[mode])
    with patch.object(cache_mod.AutotuneKey, "build", wraps=cache_mod.AutotuneKey.build) as build, \
         patch.object(tuner, "maybe_load_cached_config") as load, \
         patch.object(Autotuner, "run", return_value="launched") as parent_run:
        assert tuner.run(1, N=4) == "launched"
    parent_run.assert_called_once_with(1, N=4)
    assert build.called is expect_lookup
    assert load.called is expect_lookup
