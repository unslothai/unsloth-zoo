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

import importlib
import sys
import types

import pytest
import torch

import unsloth_zoo.triton_kernels_compat as tkc


def _fake_package(name, complete = True, lazy_absolute = False):
    """A triton_kernels-shaped package; `lazy_absolute` mimics vLLM's copy importing itself as `triton_kernels`."""
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    subs = {}
    mo = types.ModuleType(name + ".matmul_ogs")
    if complete:
        for attr in ("matmul_ogs", "PrecisionConfig", "FnSpecs", "FusedActivation"):
            setattr(mo, attr, object())
    tensor = types.ModuleType(name + ".tensor")
    for attr in ("convert_layout", "wrap_torch_tensor", "FP4"):
        setattr(tensor, attr, object())
    subs["matmul_ogs"] = mo
    subs["tensor"] = tensor
    subs["swiglu"] = types.ModuleType(name + ".swiglu")
    subs["routing"] = types.ModuleType(name + ".routing")
    details = types.ModuleType(name + ".tensor_details")
    details.__path__ = []
    subs["tensor_details"] = details
    subs["tensor_details.layout"] = types.ModuleType(name + ".tensor_details.layout")
    if lazy_absolute:
        def _needs_alias(*_):
            assert "triton_kernels" in sys.modules, "vendored copy imported without the alias"
        subs["routing"].seen = _needs_alias
    return pkg, subs


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(tkc, "_MODULE", tkc._UNRESOLVED)
    monkeypatch.setattr(tkc, "_DEVICE_OK", {})
    for key in [k for k in sys.modules if k == "triton_kernels" or k.startswith("triton_kernels.")]:
        monkeypatch.delitem(sys.modules, key)
    return monkeypatch


def _install(monkeypatch, name, pkg, subs):
    monkeypatch.setitem(sys.modules, name, pkg)
    for sub, mod in subs.items():
        monkeypatch.setitem(sys.modules, f"{name}.{sub}", mod)


def test_nothing_installed_resolves_to_none_without_raising(fresh):
    fresh.setattr(tkc.importlib.util, "find_spec", lambda name, *a: None)
    assert tkc.get_triton_kernels() is None
    assert tkc.matmul_ogs_available("cpu") is False


def test_top_level_install_wins(fresh):
    pkg, subs = _fake_package("triton_kernels")
    _install(fresh, "triton_kernels", pkg, subs)
    assert tkc.get_triton_kernels() is pkg


def test_incomplete_build_is_refused(fresh):
    pkg, subs = _fake_package("triton_kernels", complete = False)
    _install(fresh, "triton_kernels", pkg, subs)
    fresh.setattr(tkc, "_import_vllm_vendored", lambda: None)
    assert tkc.get_triton_kernels() is None


def test_vllm_copy_is_used_and_the_alias_does_not_outlive_the_import(fresh):
    name = "vllm.third_party.triton_kernels"
    pkg, subs = _fake_package(name, lazy_absolute = True)
    _install(fresh, name, pkg, subs)
    real_find = importlib.util.find_spec
    fresh.setattr(tkc.importlib.util, "find_spec",
                  lambda n, *a: None if n == "triton_kernels" else (object() if n == name else real_find(n, *a)))
    seen = []
    real_import = importlib.import_module

    def spy(n, *a):
        seen.append((n, "triton_kernels" in sys.modules))
        return real_import(n, *a)
    fresh.setattr(tkc.importlib, "import_module", spy)
    assert tkc.get_triton_kernels() is pkg
    # patch_gpt_oss keys its native (forward-only) path on `import triton_kernels`: it must still fail.
    assert "triton_kernels" not in sys.modules
    assert any(alias for n, alias in seen if n.startswith(name + "."))


def test_opt_out_env(fresh):
    pkg, subs = _fake_package("triton_kernels")
    _install(fresh, "triton_kernels", pkg, subs)
    fresh.setenv("UNSLOTH_MXFP4_MATMUL_OGS", "0")
    assert tkc.get_triton_kernels() is None


def test_self_check_failure_is_cached_false(fresh):
    calls = []

    def boom(device):
        calls.append(device)
        raise RuntimeError("launch failed")
    fresh.setattr(tkc, "_self_check", boom)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert tkc.matmul_ogs_available(dev) is False
    assert tkc.matmul_ogs_available(dev) is False
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "matmul_ogs needs a CUDA device")
def test_real_self_check_matches_exact_dequant_and_leaves_patch_gpt_oss_alone(fresh):
    if tkc.get_triton_kernels() is None:
        pytest.skip("no triton_kernels build (top-level or vLLM's) in this environment")
    assert tkc.matmul_ogs_available("cuda") is True
    assert "triton_kernels" not in sys.modules or importlib.util.find_spec("triton_kernels") is not None
    from unsloth_zoo.temporary_patches.gpt_oss import _check_triton_kernels_available
    assert _check_triton_kernels_available() == (importlib.util.find_spec("triton_kernels") is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "matmul_ogs needs a CUDA device")
def test_self_check_rejects_a_wrong_layout(fresh):
    if tkc.get_triton_kernels() is None:
        pytest.skip("no triton_kernels build (top-level or vLLM's) in this environment")
    real = tkc.mxfp4_ogs_weight
    # Experts rolled by one: every routed GEMM reads its neighbour's weights.
    fresh.setattr(tkc, "mxfp4_ogs_weight", lambda b, s: real(b.roll(1, 0), s.roll(1, 0)))
    assert tkc._self_check(torch.device("cuda")) is False


def test_no_cuda_answers_false_without_touching_cuda(fresh):
    fresh.setattr(torch.cuda, "is_available", lambda: False)

    def no_cuda(*args, **kwargs):
        raise AssertionError("queried CUDA on a host without it")

    fresh.setattr(torch.cuda, "current_device", no_cuda)
    assert tkc.matmul_ogs_available() is False
