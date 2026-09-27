"""transformers 5.17 lazy_load_kernel never tries the local mamba_ssm / causal_conv1d packages.

"mamba-ssm" and "causal-conv1d" left _HUB_KERNEL_MAPPING, and lazy_load_kernel returns None for any name
missing from it, so remote modeling code that resolves its Mamba kernels this way (nvidia
Nemotron-3-Nano-Omni's modeling_nemotron_h.py) always runs the torch reference Mamba path even with
working wheels installed. The real-package tests need CUDA sm_80+ and importable mamba_ssm + causal_conv1d;
the stand-in package tests run anywhere.
"""
import importlib
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
if not hasattr(hub_kernels, "lazy_load_kernel"):
    # transformers 4.x: no lazy_load_kernel, so remote code of this shape cannot load and the patch is a no-op.
    pytest.skip("this transformers has no lazy_load_kernel", allow_module_level=True)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None
    or torch.cuda.get_device_capability() < (8, 0),
    reason="needs NVIDIA CUDA sm_80+",
)


def _local_ok():
    try:
        importlib.import_module("mamba_ssm.ops.triton.ssd_combined")
        importlib.import_module("causal_conv1d")
        return True
    except Exception:
        return False


@pytest.fixture
def patched(monkeypatch):
    import transformers.integrations as integrations
    saved = (hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel)
    monkeypatch.setattr(hub_kernels, "_KERNEL_MODULE_MAPPING", {})
    try:
        from unsloth_zoo.temporary_patches.misc import patch_lazy_load_kernel_local_packages
    except ImportError:
        patch_lazy_load_kernel_local_packages = None
    if patch_lazy_load_kernel_local_packages is not None:
        patch_lazy_load_kernel_local_packages()
    yield
    hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel = saved


@needs_cuda
@pytest.mark.skipif(not _local_ok(), reason="mamba_ssm / causal_conv1d not importable")
def test_remote_style_resolution_finds_local_kernels(patched):
    assert "mamba-ssm" not in hub_kernels._HUB_KERNEL_MAPPING
    # What modeling_nemotron_h.py does: `from transformers.integrations import lazy_load_kernel` at import,
    # then resolve the kernels inside the mixer __init__.
    ns = {}
    exec("from transformers.integrations import lazy_load_kernel", ns)
    from transformers.utils.import_utils import resolve_internal_import
    mamba = ns["lazy_load_kernel"]("mamba-ssm")
    conv = ns["lazy_load_kernel"]("causal-conv1d")
    assert isinstance(mamba, types.ModuleType) and mamba.__name__ == "mamba_ssm"
    assert isinstance(conv, types.ModuleType) and conv.__name__ == "causal_conv1d"
    for path in ("ops.triton.selective_state_update.selective_state_update",
                 "ops.triton.ssd_combined.mamba_chunk_scan_combined",
                 "ops.triton.ssd_combined.mamba_split_conv1d_scan_combined"):
        assert resolve_internal_import(mamba, chained_path=path) is not None, path
    assert getattr(conv, "causal_conv1d_fn", None) is not None
    assert getattr(conv, "causal_conv1d_update", None) is not None


@needs_cuda
@pytest.mark.skipif(not _local_ok(), reason="mamba_ssm / causal_conv1d not importable")
def test_cached_none_is_retried(patched):
    hub_kernels._KERNEL_MODULE_MAPPING["mamba-ssm"] = None
    assert isinstance(hub_kernels.lazy_load_kernel("mamba-ssm"), types.ModuleType)


@needs_cuda
def test_other_names_unchanged(patched):
    assert hub_kernels.lazy_load_kernel("unsloth-no-such-kernel") is None


def test_pre_ampere_and_kill_switch_keep_original(monkeypatch):
    import transformers.integrations as integrations
    saved = (hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel)
    try:
        from unsloth_zoo.temporary_patches.misc import patch_lazy_load_kernel_local_packages
    except ImportError:
        pytest.skip("patch not present")
    try:
        monkeypatch.setenv("UNSLOTH_LOCAL_MAMBA_KERNELS", "0")
        patch_lazy_load_kernel_local_packages()
        assert hub_kernels.lazy_load_kernel is saved[0]
        monkeypatch.delenv("UNSLOTH_LOCAL_MAMBA_KERNELS")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
        patch_lazy_load_kernel_local_packages()
        assert hub_kernels.lazy_load_kernel is saved[0]
    finally:
        hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel = saved


@pytest.fixture
def stand_in_packages(monkeypatch):
    """Tiny stand-ins for mamba_ssm / causal_conv1d on a pretend sm_90 GPU, so the resolution logic runs on CPU."""
    import sys
    import transformers.integrations as integrations
    if "mamba-ssm" in hub_kernels._HUB_KERNEL_MAPPING:
        pytest.skip("this transformers still maps mamba-ssm to a Hub kernel")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    fakes = {}
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.ssd_combined", "causal_conv1d"):
        fakes[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, fakes[name])
    fakes["causal_conv1d"].causal_conv1d_fn = lambda *a, **k: None
    monkeypatch.setattr(hub_kernels, "_KERNEL_MODULE_MAPPING", {})
    saved = (hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel)
    try:
        from unsloth_zoo.temporary_patches.misc import patch_lazy_load_kernel_local_packages
    except ImportError:
        patch_lazy_load_kernel_local_packages = None
    if patch_lazy_load_kernel_local_packages is not None:
        patch_lazy_load_kernel_local_packages()
    yield fakes
    hub_kernels.lazy_load_kernel, integrations.lazy_load_kernel = saved


def test_stand_in_packages_are_found(stand_in_packages):
    ns = {}
    exec("from transformers.integrations import lazy_load_kernel", ns)
    assert ns["lazy_load_kernel"]("mamba-ssm") is stand_in_packages["mamba_ssm"]
    assert ns["lazy_load_kernel"]("causal-conv1d") is stand_in_packages["causal_conv1d"]
    # Cached like a Hub kernel, and every other name still goes through transformers.
    assert hub_kernels._KERNEL_MODULE_MAPPING["mamba-ssm"] is stand_in_packages["mamba_ssm"]
    assert hub_kernels.lazy_load_kernel("unsloth-no-such-kernel") is None


def test_broken_install_falls_back_to_none(stand_in_packages, monkeypatch):
    import sys
    # mamba_ssm imports but its Triton submodule does not: keep transformers' None (torch path).
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", None)
    assert hub_kernels.lazy_load_kernel("mamba-ssm") is None
