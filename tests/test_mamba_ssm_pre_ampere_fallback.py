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

"""Tests patch_mamba_ssm_pre_ampere_fallback in temporary_patches/misc.py.

mamba_ssm's Triton kernels need sm_80+. On a T4 the package imports fine and
is_fast_path_available is True, so the layer routes into cuda_kernels_forward
and Triton fails once training starts. The patch must fire ONLY on pre-Ampere
NVIDIA CUDA: ROCm, XPU, MPS and CPU are left alone, Ampere+ keeps the fast path.

Extracted by AST so the test needs neither a GPU nor a transformers import.
"""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"
_SRC = MISC.read_text(encoding = "utf-8")


def _load():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_mamba_ssm_pre_ampere_fallback":
            ns = {"torch": torch}
            exec(ast.get_source_segment(_SRC, node), ns)
            return ns[node.name]
    raise AssertionError("patch_mamba_ssm_pre_ampere_fallback not found")


patch = _load()

MODEL_MOD = "transformers.models.granitemoehybrid.modeling_granitemoehybrid"


class _FakeCuda:
    def __init__(self, available = True, capability = (7, 5)):
        self._available, self._capability = available, capability
    def is_available(self): return self._available
    def get_device_capability(self, *a, **k): return self._capability


@pytest.fixture
def env(monkeypatch):
    """Pre-Ampere NVIDIA CUDA with mamba_ssm installed and a model imported."""
    monkeypatch.setattr(torch, "cuda", _FakeCuda(), raising = False)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)

    saved = {k: sys.modules.get(k) for k in ("mamba_ssm", MODEL_MOD,
                                             "transformers.utils.import_utils")}
    sys.modules["mamba_ssm"] = types.ModuleType("mamba_ssm")

    model_mod = types.ModuleType(MODEL_MOD)
    model_mod.is_fast_path_available = True
    model_mod.selective_state_update = lambda *a, **k: None
    model_mod.mamba_chunk_scan_combined = lambda *a, **k: None
    model_mod.mamba_split_conv1d_scan_combined = lambda *a, **k: None
    sys.modules[MODEL_MOD] = model_mod

    # transformers 5 also resolves these from the Hub, through
    # integrations.hub_kernels.lazy_load_kernel, which asks no predicate.
    hk = types.ModuleType("transformers.integrations.hub_kernels")
    hk._HUB_KERNEL_MAPPING = {
        "causal-conv1d": {"repo_id": "kernels-community/causal-conv1d"},
        "mamba-ssm": {"repo_id": "kernels-community/mamba-ssm"},
        "falcon_mamba-ssm": {"repo_id": "kernels-community/mamba-ssm"},
        "deep-gemm": {"repo_id": "kernels-community/deep-gemm"},
    }
    hk._KERNEL_MODULE_MAPPING = {"mamba-ssm": types.ModuleType("already_loaded")}
    for name in ("transformers.integrations", "transformers.integrations.hub_kernels"):
        saved[name] = sys.modules.get(name)
    integrations = types.ModuleType("transformers.integrations")
    integrations.__path__ = []
    integrations.hub_kernels = hk
    sys.modules["transformers.integrations"] = integrations
    sys.modules["transformers.integrations.hub_kernels"] = hk

    iu = types.ModuleType("transformers.utils.import_utils")
    iu.is_mamba_ssm_available = lambda: True
    iu.is_mamba_2_ssm_available = lambda: True
    sys.modules["transformers.utils.import_utils"] = iu
    # `import transformers.utils.import_utils as _iu` needs the parents too.
    for parent in ("transformers", "transformers.utils"):
        if parent not in sys.modules:
            m = types.ModuleType(parent)
            m.__path__ = []
            sys.modules[parent] = m
            saved.setdefault(parent, None)
    sys.modules["transformers"].utils = sys.modules["transformers.utils"]
    # `import a.b.c as x` resolves through getattr(a.b, "c"), so this attribute
    # outlives a sys.modules-only teardown and hands the stub to every later
    # `import transformers.utils.import_utils as iu` (test_vendor_fla.py:219,
    # fla_vendor.py:473). Restore it too.
    utils_mod = sys.modules["transformers.utils"]
    had_attr = hasattr(utils_mod, "import_utils")
    saved_attr = getattr(utils_mod, "import_utils", None)
    utils_mod.import_utils = iu

    yield model_mod, iu, hk

    if had_attr: utils_mod.import_utils = saved_attr
    elif hasattr(utils_mod, "import_utils"): del utils_mod.import_utils
    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v


def _no_local_wheel():
    """Make `import mamba_ssm` raise, for the not-installed branch.

    None in sys.modules does that even where the real package is installed,
    which popping the fake would not. Assigned rather than
    `monkeypatch.setitem`: monkeypatch undoes after the `env` teardown and
    would put the fake back, and a `types.ModuleType` has `__spec__` None, so
    a leaked one makes every later `importlib.util.find_spec("mamba_ssm")`
    raise ValueError. `env` owns the key and restores it.
    """
    sys.modules["mamba_ssm"] = None


def test_pre_ampere_disables_fast_path(env):
    model_mod, iu, _hk = env
    assert patch() is True
    assert model_mod.is_fast_path_available is False
    assert model_mod.selective_state_update is None
    assert model_mod.mamba_chunk_scan_combined is None
    assert model_mod.mamba_split_conv1d_scan_combined is None
    # Modules imported later must also see it as unavailable.
    assert iu.is_mamba_ssm_available() is False
    assert iu.is_mamba_2_ssm_available() is False


def test_ampere_keeps_fast_path(env, monkeypatch):
    model_mod, iu, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (8, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_hopper_keeps_fast_path(env, monkeypatch):
    model_mod, _, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (9, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_rocm_is_untouched(env, monkeypatch):
    model_mod, iu, _hk = env
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_no_cuda_is_untouched(env, monkeypatch):
    model_mod, _, _hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(available = False), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_no_mamba_ssm_is_untouched(env):
    model_mod, iu, _hk = env
    _no_local_wheel()
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


# ---- transformers 5's Hub kernels ----------------------------------------
#
# lazy_load_kernel resolves mamba-ssm from kernels-community when the `kernels`
# package is installed. It needs no local wheel and never calls
# is_mamba_ssm_available, so the predicates above cannot reach it.

def test_pre_ampere_unregisters_the_hub_kernels(env):
    _model_mod, _iu, hk = env
    patch()
    for name in ("mamba-ssm", "falcon_mamba-ssm", "causal-conv1d"):
        assert name not in hk._HUB_KERNEL_MAPPING
        # A module already resolved into the cache must stop being handed out.
        assert hk._KERNEL_MODULE_MAPPING[name] is None


def test_unrelated_hub_kernels_are_left_registered(env):
    _model_mod, _iu, hk = env
    patch()
    assert "deep-gemm" in hk._HUB_KERNEL_MAPPING


def test_hub_kernels_go_even_without_the_local_wheel(env):
    """The Hub path is the one that reaches a pre-Ampere GPU with no wheel
    installed, so unregistering has to precede the mamba_ssm import."""
    _model_mod, _iu, hk = env
    _no_local_wheel()
    assert patch() is None
    assert "mamba-ssm" not in hk._HUB_KERNEL_MAPPING


def test_ampere_keeps_the_hub_kernels(env, monkeypatch):
    _model_mod, _iu, hk = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (8, 0)), raising = False)
    assert patch() is None
    assert "mamba-ssm" in hk._HUB_KERNEL_MAPPING


def test_non_transformers_modules_are_left_alone(env):
    vllm = types.ModuleType("vllm.model_executor.layers.mamba")
    vllm.is_fast_path_available = True
    sys.modules["vllm.model_executor.layers.mamba"] = vllm
    try:
        patch()
        assert vllm.is_fast_path_available is True
    finally:
        sys.modules.pop("vllm.model_executor.layers.mamba", None)


def test_registered_as_a_temporary_patch():
    assert "TEMPORARY_PATCHES.append(patch_mamba_ssm_pre_ampere_fallback)" in _SRC


# Keep last: it checks what the `env` fixture left behind after teardown.
def test_the_fixture_leaves_no_stub_bound_on_transformers_utils():
    """The stub must not outlive the fixture. `import a.b.c as x` goes through
    getattr(a.b, "c"), so a leaked attribute is handed to every later
    `import transformers.utils.import_utils as iu` -- the form used by
    tests/test_vendor_fla.py:219 and unsloth_zoo/temporary_patches/fla_vendor.py:473."""
    utils_mod = sys.modules.get("transformers.utils")
    if utils_mod is None or getattr(utils_mod, "__file__", None) is None:
        pytest.skip("real transformers.utils was never imported in this session")
    import importlib
    real = importlib.import_module("transformers.utils.import_utils")
    assert getattr(utils_mod, "import_utils", real) is real
    import transformers.utils.import_utils as reimported
    assert reimported is real


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
