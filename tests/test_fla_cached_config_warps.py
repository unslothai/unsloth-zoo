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

"""Disk-cached FLA_CACHE_MODE configs whose num_warps the kernel no longer offers are ignored: the vendored fla drops
8-warp chunk_bwd_kernel_dv_local on Blackwell with Triton < 3.8 (fla PR 1399);
CachedAutotuner.maybe_load_cached_config must not revive it."""

import os
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")


@pytest.mark.parametrize("cached_warps, used", [(8, False), (4, True)])
def test_disk_cached_config_outside_the_offered_warps_is_ignored(monkeypatch, cached_warps, used):
    # FLA_CACHE_MODE disk configs go straight into the tuner cache, so an 8-warp entry from another host or from before
    # the Blackwell drop must not revive the racy config.
    pytest.importorskip("triton")
    import triton
    cache_mod = pytest.importorskip("unsloth_zoo._vendored.fla.ops.utils.cache")

    entry = {"kwargs": {}, "num_warps": cached_warps, "num_stages": 2, "num_ctas": 1}
    monkeypatch.setattr(cache_mod, "load_cached_config", lambda name, key: entry)
    tuner = types.SimpleNamespace(
        kernel_name = "chunk_bwd_kernel_dv_local",
        configs = [triton.Config({}, num_warps = w, num_stages = s) for w in (2, 4) for s in (2, 3, 4)],
        cache = {},
    )
    key = types.SimpleNamespace(autotune_key = ("k",))
    cache_mod.CachedAutotuner.maybe_load_cached_config(tuner, key)
    assert (("k",) in tuner.cache) is used
    if used:
        assert tuner.cache[("k",)].num_warps == cached_warps
