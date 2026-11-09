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

"""patch_param_wrapper_for_moe runs from whichever moe_utils copy is current: the package, or the compiled-cache copy
under UNSLOTH_COMPILE_LOCATION. When the location moves in one process, a fresh copy patches over the previous copy's
patched forward. It must keep PEFT's own forward as its original, not the other copy's patched one: a nested chain
runs the separated-LoRA forward twice, and a routed forward that calls the original recursed without end."""

import sys

import pytest

pytest.importorskip("peft")

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402


def _copy_in(location, monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(location))
    assert MU.install_to_cache(MU.__file__)
    module = MU._load_cached_moe_utils_module()
    assert module is not None and module is not MU
    return module


def test_a_second_cached_copy_keeps_pefts_forward_as_its_original(tmp_path, monkeypatch):
    from peft.tuners.lora.layer import ParamWrapper

    monkeypatch.setattr(ParamWrapper, "forward", ParamWrapper.forward)
    monkeypatch.delitem(sys.modules, "unsloth_cached_moe_utils", raising = False)
    pefts = getattr(ParamWrapper.forward, "_unsloth_moe_peft_forward", ParamWrapper.forward)
    monkeypatch.setattr(ParamWrapper, "forward", pefts)

    first = _copy_in(tmp_path / "a", monkeypatch)
    monkeypatch.setattr(first, "_original_param_wrapper_forward", None)
    assert MU.patch_param_wrapper_for_moe()
    assert ParamWrapper.forward is first._patched_param_wrapper_forward
    assert first._original_param_wrapper_forward is pefts

    second = _copy_in(tmp_path / "b", monkeypatch)
    assert second is not first
    monkeypatch.setattr(second, "_original_param_wrapper_forward", None)
    assert MU.patch_param_wrapper_for_moe()
    assert ParamWrapper.forward is second._patched_param_wrapper_forward
    assert second._original_param_wrapper_forward is pefts, (
        "the second copy wrapped the first copy's patched forward instead of PEFT's"
    )


def test_the_package_patching_after_a_cached_copy_keeps_pefts_forward(tmp_path, monkeypatch):
    from peft.tuners.lora.layer import ParamWrapper

    monkeypatch.setattr(ParamWrapper, "forward", ParamWrapper.forward)
    monkeypatch.delitem(sys.modules, "unsloth_cached_moe_utils", raising = False)
    pefts = getattr(ParamWrapper.forward, "_unsloth_moe_peft_forward", ParamWrapper.forward)
    monkeypatch.setattr(ParamWrapper, "forward", pefts)

    first = _copy_in(tmp_path / "a", monkeypatch)
    monkeypatch.setattr(first, "_original_param_wrapper_forward", None)
    assert MU.patch_param_wrapper_for_moe()
    assert ParamWrapper.forward is first._patched_param_wrapper_forward

    # No cached copy at the new location: the package module patches itself.
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "empty"))
    monkeypatch.setattr(MU, "_original_param_wrapper_forward", None)
    assert MU.patch_param_wrapper_for_moe()
    assert ParamWrapper.forward is MU._patched_param_wrapper_forward
    assert MU._original_param_wrapper_forward is pefts
