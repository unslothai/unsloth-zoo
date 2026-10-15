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

"""Device gates must read every visible GPU (or the current one), never device 0 alone.

CPU-only: torch.cuda is spoofed per test. Covers bf16 for GC offload buffers, the
pre-Turing compile switch, tiled-MLP free memory, the per-device AMD arch / aiter
caches (set_device switch), the rocminfo mask fallback and the llama.cpp asset target.
"""
import ast
import importlib.machinery
import inspect
import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import device_type
from unsloth_zoo import gradient_checkpointing as gc
from unsloth_zoo import llama_cpp
from unsloth_zoo import tiled_mlp

_MISSING = object()
MIXED = [[(8, 0), (7, 5)], [(7, 5), (8, 0)]]


@pytest.fixture
def cuda_caps(monkeypatch):
    """Spoof N visible CUDA devices with the given capabilities; current device settable."""
    state = {"current": 0}

    def apply(caps, current = 0):
        state["current"] = current
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
        monkeypatch.setattr(
            torch.cuda, "get_device_capability",
            lambda device = None: caps[state["current"] if device is None else device],
        )
        return state

    return apply


# C1: bf16 offload buffers -------------------------------------------------------------

@pytest.fixture
def gc_dtype(monkeypatch, cuda_caps):
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    names = {n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names}
    saved = {name: getattr(gc, name, _MISSING) for name in names}
    real_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    def run(caps, current = 0):
        cuda_caps(caps, current)
        # The visible-device list comes from the spoof; the GC loop itself sees no device.
        listed = [tuple(c) for c in caps]
        monkeypatch.setattr(device_type, "_visible_cuda_capabilities", lambda: listed, raising = False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(gc, "DEVICE_TYPE", "cuda")
        monkeypatch.setattr(gc, "DEVICE_TYPE_TORCH", "cuda")
        gc.CPU_BUFFERS = []
        gc.initialize_unsloth_gradient_checkpointing()
        return gc.CPU_BUFFERS[0].dtype

    yield run
    for name, value in saved.items():
        if value is _MISSING:
            if hasattr(gc, name): delattr(gc, name)
        else:
            setattr(gc, name, value)


@pytest.mark.parametrize("caps", MIXED)
@pytest.mark.parametrize("current", [0, 1])
def test_gc_bf16_needs_every_visible_device(gc_dtype, caps, current):
    assert gc_dtype(caps, current) is torch.float16


def test_gc_bf16_homogeneous_ampere_unchanged(gc_dtype):
    assert gc_dtype([(8, 0), (8, 0)]) is torch.bfloat16


def test_visible_cuda_capabilities_lists_every_device(cuda_caps):
    cuda_caps([(8, 6), (6, 1), (9, 0)], current = 2)
    assert device_type._visible_cuda_capabilities() == [(8, 6), (6, 1), (9, 0)]


# C2: pre-Turing compile switch --------------------------------------------------------

def _old_cuda_arch_version(caps, current, cuda_caps):
    """Run compiler.py's module-level capability block against spoofed devices."""
    import unsloth_zoo.compiler as compiler
    tree = ast.parse(inspect.getsource(compiler))
    block = next(
        node for node in tree.body
        if isinstance(node, ast.If) and "OLD_CUDA_ARCH_VERSION" in ast.unparse(node)
        and "get_device_capability" in ast.unparse(node)
    )
    cuda_caps(caps, current)
    namespace = {"__name__": "unsloth_zoo.compiler", "__package__": "unsloth_zoo",
                 "torch": torch, "DEVICE_TYPE": "cuda"}
    exec(compile(ast.Module(body = [block], type_ignores = []), "compiler_block", "exec"), namespace)
    return namespace["OLD_CUDA_ARCH_VERSION"]


@pytest.mark.parametrize("current", [0, 1])
@pytest.mark.parametrize("caps", [[(8, 0), (6, 1)], [(6, 1), (8, 0)]])
def test_compile_switch_sees_any_old_device(cuda_caps, caps, current):
    assert _old_cuda_arch_version(caps, current, cuda_caps) is True


def test_compile_switch_homogeneous_unchanged(cuda_caps):
    assert _old_cuda_arch_version([(8, 0), (8, 0)], 1, cuda_caps) is False
    assert _old_cuda_arch_version([(7, 0)], 0, cuda_caps) is True


# Z1: tiled-MLP free memory ------------------------------------------------------------

def _fake_cuda():
    fake = mock.MagicMock()
    fake.is_available.return_value = True
    fake.mem_get_info.return_value = (8 * 1024 ** 3, 16 * 1024 ** 3)
    return fake


def test_tiled_mlp_reads_current_device_not_gpu0():
    fake = _fake_cuda()
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), mock.patch.object(torch, "cuda", fake):
        assert tiled_mlp._default_target_gb() == 4.0
    assert fake.mem_get_info.call_args.args in ((), (None,)), fake.mem_get_info.call_args


def test_tiled_mlp_reads_the_activation_device():
    fake = _fake_cuda()
    device = torch.device("cuda", 1)
    with mock.patch.object(tiled_mlp, "DEVICE_TYPE", "cuda"), mock.patch.object(torch, "cuda", fake):
        tiled_mlp._default_target_gb(device)
    assert fake.mem_get_info.call_args.args == (device,)


# Z3: per-device AMD arch / aiter caches -----------------------------------------------

def _clear_amd_caches():
    for fn in (device_type._detect_gfx_arch, device_type.get_amd_attention_implementation,
               device_type.get_amd_flash_attn_func):
        fn.cache_clear()


@pytest.fixture
def amd_host(monkeypatch):
    """Two HIP devices: gfx942 at 0, gfx1100 at 1, ROCm 7, aiter importable."""
    state = {"current": 0}
    archs = ["gfx942:sramecc+:xnack-", "gfx1100"]
    monkeypatch.setattr(device_type, "is_hip", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: state.__setitem__("current", i), raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(gcnArchName = archs[i]))
    monkeypatch.setattr(torch.version, "hip", "7.0.1", raising = False)
    aiter = types.ModuleType("aiter")
    aiter.__spec__ = importlib.machinery.ModuleSpec("aiter", None)
    aiter.flash_attn_func = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    _clear_amd_caches()
    yield aiter
    _clear_amd_caches()


def test_gfx_arch_follows_set_device(amd_host):
    assert device_type._detect_gfx_arch() == "gfx942"
    torch.cuda.set_device(1)
    assert device_type._detect_gfx_arch() == "gfx1100"
    torch.cuda.set_device(0)
    assert device_type._detect_gfx_arch() == "gfx942"


def test_aiter_choice_follows_set_device(amd_host):
    assert device_type.get_amd_attention_implementation() == "amd_aiter"
    assert device_type.get_amd_flash_attn_func() is amd_host.flash_attn_func
    torch.cuda.set_device(1)
    assert device_type.get_amd_attention_implementation() == "sdpa"
    assert device_type.get_amd_flash_attn_func() is None


def test_amd_caches_take_an_explicit_device(amd_host):
    assert device_type._detect_gfx_arch(1) == "gfx1100"
    assert device_type.get_amd_attention_implementation("cuda:1") == "sdpa"
    assert device_type.get_amd_attention_implementation(torch.device("cuda", 0)) == "amd_aiter"


# Z2: rocminfo fallback honours HIP masks ----------------------------------------------

ROCMINFO = """
*******
Agent 1
*******
  Name:                    AMD Ryzen AI Max+ 395
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1151
  Device Type:             GPU
  ISA 1: amdgcn-amd-amdhsa--gfx1151
*******
Agent 3
*******
  Name:                    gfx1100
  Device Type:             GPU
  ISA 1: amdgcn-amd-amdhsa--gfx1100
"""


@pytest.mark.parametrize("env, expected", [
    ({}, "gfx1151"),
    ({"HIP_VISIBLE_DEVICES": "1"}, "gfx1100"),
    ({"CUDA_VISIBLE_DEVICES": "1,0"}, "gfx1100"),
    ({"HIP_VISIBLE_DEVICES": ""}, None),
    ({"HIP_VISIBLE_DEVICES": "-1"}, None),
])
def test_rocminfo_fallback_honours_mask(monkeypatch, env, expected):
    for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import subprocess
    def run(command, **kwargs):
        out = ROCMINFO if command[0] == "rocminfo" else ""
        return types.SimpleNamespace(stdout = out, returncode = 0)
    monkeypatch.setattr(subprocess, "run", run)
    device_type._detect_gfx_arch.cache_clear()
    try:
        assert device_type._detect_gfx_arch() == expected
    finally:
        device_type._detect_gfx_arch.cache_clear()


# Z4: llama.cpp asset target -----------------------------------------------------------

def _fake_torch(caps = None, archs = None, current = 0):
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(caps or archs),
        current_device = lambda: current,
        get_device_capability = lambda i = None: caps[current if i is None else i],
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
    )
    version = types.SimpleNamespace(hip = "6.4" if archs else None, cuda = None if archs else "12.8")
    return types.SimpleNamespace(cuda = cuda, version = version)


@pytest.mark.parametrize("caps", MIXED)
@pytest.mark.parametrize("current", [0, 1])
def test_cuda_target_covers_every_visible_sm(monkeypatch, caps, current):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(caps = caps, current = current))
    assert llama_cpp._detect_gpu_target() == ("cuda", 75, "cuda12", (75, 80))


def test_cuda_target_homogeneous_unchanged(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(caps = [(8, 6), (8, 6)], current = 1))
    assert llama_cpp._detect_gpu_target() == ("cuda", 86, "cuda12")


def test_rocm_target_mixed_family_builds_from_source(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1100", "gfx1201"]))
    assert llama_cpp._detect_gpu_target() is None


def test_rocm_target_same_family_uses_current_arch(monkeypatch):
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(archs = ["gfx1100", "gfx1101:xnack-"], current = 1))
    assert llama_cpp._detect_gpu_target() == ("rocm", "gfx1101")


def _manifest():
    def art(name, sms, coverage = "narrow"):
        return {"asset_name": name, "install_kind": "linux-cuda", "runtime_line": "cuda12",
                "supported_sms": [str(s) for s in sms], "coverage_class": coverage,
                "min_sm": min(sms), "max_sm": max(sms), "rank": 0}
    artifacts = [art("sm80-only.tar.gz", [80, 86]), art("sm75-only.tar.gz", [75]),
                 art("turing-ampere.tar.gz", [75, 80, 86, 89]),
                 art("portable.tar.gz", [75, 80, 86, 89, 90, 100, 120], "portable")]
    return {"artifacts": artifacts}, {a["asset_name"]: f"https://x/{a['asset_name']}" for a in artifacts}


def test_asset_ranking_covers_every_visible_sm(monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    manifest, assets = _manifest()
    names = [n for n, _ in llama_cpp._select_gpu_assets(
        "b1", assets, manifest, target = ("cuda", 75, "cuda12", (75, 80)))]
    assert names == ["turing-ampere.tar.gz", "portable.tar.gz"]
    # Single device: narrowest covering bundle first, unchanged.
    names = [n for n, _ in llama_cpp._select_gpu_assets("b1", assets, manifest, target = ("cuda", 75, "cuda12"))]
    assert names == ["sm75-only.tar.gz", "portable.tar.gz"]
