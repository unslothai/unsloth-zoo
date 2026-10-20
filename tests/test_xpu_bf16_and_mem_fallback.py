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

"""Intel XPU free-memory fallback, native bf16 detection, and the Apple MLX bf16 answer.

torch.xpu.mem_get_info raises on some Intel GPUs (Arc B580, Lunar Lake), and
torch.xpu.is_bf16_supported() counts emulation so it is always True. Everything
here runs on a CPU box with spoofed torch.xpu / mlx.core.
"""
import ast
import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import device_type as dt

_GB = 1024 ** 3
_MISSING = object()


def _props(flag = _MISSING, total = 16 * _GB):
    props = types.SimpleNamespace(total_memory = total)
    if flag is not _MISSING:
        props.has_bfloat16_conversions = flag
    return props


def _fake_xpu(flags = (True,), mem_get_info = None, is_bf16 = True, props_raise = False):
    devices = [_props(flag) for flag in flags]

    def get_device_properties(index = None):
        if props_raise:
            raise RuntimeError("unreadable")
        return devices[0 if index is None else index]

    xpu = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(devices),
        current_device = lambda: len(devices) - 1,
        get_device_properties = get_device_properties,
        memory_reserved = lambda index = None: 3 * _GB,
        mem_get_info = mem_get_info,
    )
    if is_bf16 is not None:
        # torch's default including_emulation=True: always True whatever the card.
        xpu.is_bf16_supported = lambda including_emulation = True: is_bf16
    return xpu



# XPU and CUDA never run on macOS, where zoo imports down the MLX path; spoofing them there tests nothing.
not_on_macos = pytest.mark.skipif(sys.platform == "darwin", reason = "XPU / CUDA paths do not run on macOS")

@pytest.fixture
def on_xpu(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available = lambda: False)

    def install(xpu):
        monkeypatch.setattr(dt, "DEVICE_TYPE", "xpu")
        monkeypatch.setattr(torch, "cuda", fake_cuda)
        monkeypatch.setattr(torch, "xpu", xpu, raising = False)
    return install


# ---- device_is_bf16_supported on XPU ----

@not_on_macos
@pytest.mark.parametrize(
    "flags, expected",
    [
        ((True,), True),                 # single native card: unchanged
        ((True, True), True),            # homogeneous: unchanged
        ((False,), False),               # emulation only (torch still says True)
        ((True, False), False),          # mixed: weakest card decides
        ((False, True), False),
        ((_MISSING,), True),             # older torch without the attribute: unchanged
        ((_MISSING, _MISSING), True),
    ],
)
def test_xpu_bf16_uses_native_conversions_on_every_device(on_xpu, flags, expected):
    on_xpu(_fake_xpu(flags))
    assert dt.device_is_bf16_supported() is expected


@not_on_macos
def test_xpu_bf16_unreadable_properties_keep_todays_answer(on_xpu):
    on_xpu(_fake_xpu((False,), props_raise = True))
    assert dt.device_is_bf16_supported() is True


@not_on_macos
def test_xpu_bf16_partial_build_still_false(on_xpu):
    on_xpu(_fake_xpu((True,), is_bf16 = None))
    assert dt.device_is_bf16_supported() is False


@not_on_macos
def test_cuda_bf16_still_asks_torch_cuda(monkeypatch):
    for answer in (True, False):
        fake_cuda = types.SimpleNamespace(is_available = lambda: True, is_bf16_supported = lambda a = answer: a)
        monkeypatch.setattr(dt, "DEVICE_TYPE", "cuda")
        monkeypatch.setattr(torch, "cuda", fake_cuda)
        assert dt.device_is_bf16_supported() is answer


# ---- gradient checkpointing offload dtype on XPU ----

@not_on_macos
def test_gradient_checkpointing_xpu_buffers_follow_native_bf16(monkeypatch):
    from unsloth_zoo import gradient_checkpointing as gc
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    written = {n for node in ast.walk(tree) if isinstance(node, ast.Global) for n in node.names}
    saved = {name: getattr(gc, name, _MISSING) for name in written}
    real_empty = torch.empty

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(gc, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(gc, "DEVICE_TYPE_TORCH", "cuda")
    try:
        got = {}
        for flags in ((True,), (False,), (_MISSING,), (True, False)):
            monkeypatch.setattr(torch, "xpu", _fake_xpu(flags), raising = False)
            gc.CPU_BUFFERS = []
            gc.initialize_unsloth_gradient_checkpointing()
            got[flags] = gc.CPU_BUFFERS[0].dtype
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(gc, name):
                    delattr(gc, name)
            else:
                setattr(gc, name, value)
    assert got[(True,)] is torch.bfloat16
    assert got[(_MISSING,)] is torch.bfloat16
    assert got[(False,)] is torch.float16
    assert got[(True, False)] is torch.float16


# ---- vllm_utils ----

@pytest.fixture
def vu():
    try:
        from unsloth_zoo import vllm_utils
    except ModuleNotFoundError as e:
        pytest.skip(f"vllm_utils needs {e.name}")
    return vllm_utils


@not_on_macos
def test_get_mem_info_xpu_falls_back_when_mem_get_info_raises(monkeypatch, vu):
    def boom(*a, **k):
        raise RuntimeError("The device (Intel(R) Arc(TM) B580 Graphics) doesn't support querying the available free memory")
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(torch, "xpu", _fake_xpu((True, True), mem_get_info = boom), raising = False)
    assert vu.get_mem_info() == (13 * _GB, 16 * _GB)


@not_on_macos
def test_get_mem_info_xpu_uses_mem_get_info_when_it_works(monkeypatch, vu):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(torch, "xpu", _fake_xpu(mem_get_info = lambda: (5, 7)), raising = False)
    assert vu.get_mem_info() == (5, 7)


@not_on_macos
def test_get_mem_info_cuda_unchanged(monkeypatch, vu):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (11, 22))
    assert vu.get_mem_info() == (11, 22)


@not_on_macos
def test_vllm_xpu_dtype_asks_native_bf16_and_old_gpu_message_is_actionable(vu):
    source = inspect.getsource(vu.load_vllm)
    assert "_xpu_bf16_supported()" in source
    assert "fast_inference = False" in source
    assert "compute capability 7.0" in source
    assert "{major_version}.{minor_version}" in source


# ---- Apple MLX ----

@pytest.mark.parametrize(
    "info, expected",
    [
        ({"architecture": "applegpu_g13s", "device_name": "Apple M1 Max"}, False),
        ({"architecture": "applegpu_g14g", "device_name": "Apple M2"}, False),
        ({"architecture": "applegpu_g15p", "device_name": "Apple M3 Pro"}, True),
        ({"architecture": "applegpu_g16s", "device_name": "Apple M4 Max"}, True),
        ({"architecture": "applegpu_g17g", "device_name": "Apple M5"}, True),
        # Unparsable architecture: fall back to unsloth's name check.
        ({"architecture": "", "device_name": "Apple M2 Ultra"}, False),
        ({"architecture": "", "device_name": "Apple M4 Pro"}, True),
        ({}, True),
        (RuntimeError("no Metal"), True),
    ],
)
def test_mlx_bf16_matches_unsloth_apple_answer(monkeypatch, info, expected):
    def device_info():
        if isinstance(info, Exception):
            raise info
        return info
    core = types.ModuleType("mlx.core")
    core.device_info = device_info
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(dt, "DEVICE_TYPE", "mlx")
    assert dt.device_is_bf16_supported() is expected
