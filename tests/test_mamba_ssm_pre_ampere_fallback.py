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
    sys.modules["transformers.utils"].import_utils = iu

    yield model_mod, iu

    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v


def test_pre_ampere_disables_fast_path(env):
    model_mod, iu = env
    assert patch() is True
    assert model_mod.is_fast_path_available is False
    assert model_mod.selective_state_update is None
    assert model_mod.mamba_chunk_scan_combined is None
    assert model_mod.mamba_split_conv1d_scan_combined is None
    # Modules imported later must also see it as unavailable.
    assert iu.is_mamba_ssm_available() is False
    assert iu.is_mamba_2_ssm_available() is False


def test_ampere_keeps_fast_path(env, monkeypatch):
    model_mod, iu = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (8, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_hopper_keeps_fast_path(env, monkeypatch):
    model_mod, _ = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(capability = (9, 0)), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_rocm_is_untouched(env, monkeypatch):
    model_mod, iu = env
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


def test_no_cuda_is_untouched(env, monkeypatch):
    model_mod, _ = env
    monkeypatch.setattr(torch, "cuda", _FakeCuda(available = False), raising = False)
    assert patch() is None
    assert model_mod.is_fast_path_available is True


def test_no_mamba_ssm_is_untouched(env):
    model_mod, iu = env
    sys.modules.pop("mamba_ssm", None)
    assert patch() is None
    assert model_mod.is_fast_path_available is True
    assert iu.is_mamba_ssm_available() is True


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
